# Production RAG Implementation Guide

This guide is a full blueprint you can follow to implement your notebook flow as production-ready Python modules.

Scope covered:
- Separate retriever files (dense/simple + hybrid)
- Separate reranking file(s)
- Separate prompts
- Separate LLM initialization
- Separate CSDC and CRC composition
- One pipeline entrypoint

No implementation code is included here; this is the build contract.

## 1) Target Architecture

Flow:
1. Load settings
2. Initialize embeddings + vector store retriever (dense)
3. Build sparse retriever + hybrid retriever
4. Initialize LLM
5. Build rerank chain (PromptTemplate + LLM + parser)
6. Build rerank-aware retriever runnable
7. Build CSDC and CRC
8. Run query and return answer + trace metadata

## 2) Folder Structure To Create

Create exactly this structure under project root:

```text
RAG/
  src/
    rag_app/
      __init__.py
      config/
        __init__.py
        settings.py
      llm/
        __init__.py
        init_llm.py
      prompts/
        __init__.py
        answer_prompt.py
        rerank_prompt.py
      retrieval/
        __init__.py
        dense_retriever.py
        hybrid_retriever.py
      reranking/
        __init__.py
        rerank_parser.py
        rerank_service.py
      chains/
        __init__.py
        answer_chain.py
        rerank_retrieval_chain.py
      schemas/
        __init__.py
        rerank_schema.py
        response_schema.py
      pipeline/
        __init__.py
        query_pipeline.py
  scripts/
    run_query.py
  tests/
    test_settings.py
    test_retrieval.py
    test_rerank_parser.py
    test_rerank_service.py
    test_pipeline_smoke.py
```

## 3) File-by-File Contract

### src/rag_app/config/settings.py
Purpose:
- Central source of runtime configuration.

Create:
- `class AppSettings` (dataclass or pydantic model)

Fields:
- `openai_api_key: str`
- `chat_model: str` (default: `gpt-5.4-nano`)
- `embedding_model: str` (default: `text-embedding-3-large`)
- `temperature: float` (default: `0.0`)
- `max_tokens: int` (default: `512`)
- `collection_name: str` (default: `bookstore`)
- `persist_directory: str` (default: `chroma_store/bookstore`)
- `top_k_retrieve: int` (default: `10`)
- `top_k_rerank: int` (default: `5`)
- `source_id_filter: str | None` (default: `bhagavadgita`)
- `dense_weight: float` (default: `0.5`)
- `sparse_weight: float` (default: `0.5`)

Methods:
- `@classmethod from_env(cls) -> AppSettings`
  - Reads env vars
  - Validates required fields
  - Applies defaults

Boilerplate:
```python
from dataclasses import dataclass
import os


@dataclass(frozen=True, slots=True, kw_only=True)
class AppSettings:
    openai_api_key: str
    chat_model: str = "gpt-5.4-nano"
    embedding_model: str = "text-embedding-3-large"
    temperature: float = 0.0
    max_tokens: int = 512
    collection_name: str = "bookstore"
    persist_directory: str = "chroma_store/bookstore"
    top_k_retrieve: int = 10
    top_k_rerank: int = 5
    dense_weight: float = 0.5
    sparse_weight: float = 0.5

    def __post_init__(self) -> None:
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required.")
        if not (0.0 <= self.temperature <= 1.0):
            raise ValueError("temperature must be between 0.0 and 1.0.")
        if self.top_k_rerank < 1:
            raise ValueError("top_k_rerank must be >= 1.")
        if self.top_k_retrieve < self.top_k_rerank:
            raise ValueError("top_k_retrieve must be >= top_k_rerank.")
        if (self.dense_weight + self.sparse_weight) <= 0:
            raise ValueError("dense_weight + sparse_weight must be > 0.")

    @classmethod
    def from_env(cls) -> "AppSettings":
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            chat_model=os.getenv("CHAT_MODEL", "gpt-5.4-nano"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
            temperature=float(os.getenv("TEMPERATURE", "0.0")),
            max_tokens=int(os.getenv("MAX_TOKENS", "512")),
            collection_name=os.getenv("COLLECTION_NAME", "bookstore"),
            persist_directory=os.getenv("PERSIST_DIRECTORY", "chroma_store/bookstore"),
            top_k_retrieve=int(os.getenv("TOP_K_RETRIEVE", "10")),
            top_k_rerank=int(os.getenv("TOP_K_RERANK", "5")),
            dense_weight=float(os.getenv("DENSE_WEIGHT", "0.5")),
            sparse_weight=float(os.getenv("SPARSE_WEIGHT", "0.5")),
        )
```

Validation rules:
- `0.0 <= temperature <= 1.0`
- `top_k_retrieve >= top_k_rerank >= 1`
- `dense_weight + sparse_weight > 0`

---

### src/rag_app/llm/init_llm.py
Purpose:
- Single location for chat model initialization.

Create functions:
- `def create_chat_llm(settings: AppSettings):`
  - Uses `init_chat_model(...)` from `langchain.chat_models.base`
  - Sets model, api key, temperature, max tokens

- `def create_embeddings(settings: AppSettings):`
  - Uses `OpenAIEmbeddings(model=settings.embedding_model)`

Internal policy:
- No prompt logic in this file.
- No retrieval logic in this file.

Boilerplate:
```python
from langchain.chat_models.base import init_chat_model
from langchain_openai import OpenAIEmbeddings


def create_chat_llm(settings):
    return init_chat_model(
        model=settings.chat_model,
        api_key=settings.openai_api_key,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )


def create_embeddings(settings):
    return OpenAIEmbeddings(model=settings.embedding_model)
```

---

### src/rag_app/prompts/answer_prompt.py
Purpose:
- Final answer prompt for CSDC.

Create function:
- `def build_answer_prompt() -> ChatPromptTemplate`

Prompt requirements:
- System role: answer only from context
- If missing answer: return `I don't know.`
- Includes placeholder: `{context}`
- Human message placeholder: `{input}`

---

### src/rag_app/prompts/rerank_prompt.py
Purpose:
- Strict JSON rerank contract.

Create function:
- `def build_rerank_prompt() -> PromptTemplate`

Prompt requirements:
- Inputs: `{question}`, `{documents}`
- Output: ONLY JSON
- Schema:
  - `{"ranked": [{"rank": 1, "doc_index": 2, "reason": "..."}]}`
- Rules:
  - exact top-N
  - unique ranks
  - unique doc indexes
  - short reasons

Important:
- Escape literal braces correctly in template text.

Boilerplate:
```python
from langchain_core.prompts import PromptTemplate


def build_rerank_prompt() -> PromptTemplate:
    return PromptTemplate.from_template(
        """
You are a reranking assistant.

Task: Rank the MOST relevant documents to the user question.

User Question: {question}

Documents (numbered 1..N):
{documents}

Return ONLY valid JSON in this exact schema:
{{
  "ranked": [
    {{"rank": 1, "doc_index": 2, "reason": "short reason"}}
  ]
}}

Rules:
- Return exactly top 5 items.
- rank must be unique integers 1..5.
- doc_index must reference the numbered documents above (1-based index).
- Include each document at most once.
- reason must be <= 20 words.
- Do not include any text outside the JSON object.
"""
    )
```

---

### src/rag_app/retrieval/dense_retriever.py
Purpose:
- Build vector store and dense retriever only.

Create functions:
- `def create_vector_store(settings: AppSettings, embeddings):`
  - Initializes Chroma with persist dir + collection

- `def build_dense_retriever(vector_store, settings: AppSettings):`
  - Returns `vector_store.as_retriever(...)`
  - Applies `k=settings.top_k_retrieve`
  - Applies source filter when configured

- `def ensure_vector_index(vector_store, chunks: list) -> int`
  - If collection empty, add chunks in batches
  - Return inserted count

Internal methods:
- `_build_search_kwargs(settings: AppSettings) -> dict`

Boilerplate:
```python
from pathlib import Path
from langchain_chroma import Chroma


def create_vector_store(settings, embeddings):
    persist_directory = Path(settings.persist_directory)
    persist_directory.mkdir(parents=True, exist_ok=True)

    return Chroma(
        persist_directory=str(persist_directory),
        embedding_function=embeddings,
        collection_name=settings.collection_name,
    )


def build_dense_retriever(vector_store, settings):
    search_kwargs = {
        "k": settings.top_k_retrieve,
        "filter": {"source_id": settings.source_id_filter},
    }
    return vector_store.as_retriever(search_kwargs=search_kwargs)
```

---

### src/rag_app/retrieval/hybrid_retriever.py
Purpose:
- Build sparse retriever and combine with dense retriever.

Create functions:
- `def build_sparse_retriever(chunks: list, k: int):`
  - Uses `BM25Retriever.from_documents(...)`

- `def build_hybrid_retriever(dense_retriever, sparse_retriever, settings: AppSettings):`
  - Uses `EnsembleRetriever(...)`
  - Weights from settings

Internal checks:
- Assert both retrievers exist
- Normalize weights if needed

---

### src/rag_app/schemas/rerank_schema.py
Purpose:
- Type contracts for reranking.

Create:
- `class RerankItem`
  - `rank: int`
  - `doc_index: int` (1-based)
  - `reason: str`

- `class RerankPayload`
  - `ranked: list[RerankItem]`

Methods:
- `validate_top_n(top_n: int)`
- `validate_uniqueness()`

Boilerplate:
```python
from dataclasses import dataclass


@dataclass
class RerankItem:
    rank: int
    doc_index: int
    reason: str


@dataclass
class RerankPayload:
    ranked: list[RerankItem]
```

---

### src/rag_app/schemas/response_schema.py
Purpose:
- Output shape for pipeline responses.

Create:
- `class SourceRef`
  - `source: str | None`
  - `chunk_index: int | None`
  - `page_number: int | None`

- `class QueryResponse`
  - `question: str`
  - `answer: str`
  - `sources: list[SourceRef]`
  - `retrieved_count: int`
  - `reranked_count: int`

---

### src/rag_app/reranking/rerank_parser.py
Purpose:
- Parse and validate reranker JSON output.

Create functions:
- `def parse_rerank_output(raw_text: str) -> RerankPayload | None`
  - Strip fences and extra text if present
  - JSON parse
  - Map into schema
  - Return `None` on failure

- `def sanitize_ranked_items(items: list[RerankItem], top_n: int, max_doc_index: int) -> list[RerankItem]`
  - Drop invalid indices
  - Enforce unique `rank` and `doc_index`
  - Sort by rank
  - Trim to `top_n`

Internal methods:
- `_extract_json_blob(raw_text: str) -> str`
- `_safe_int(value, default)`

---

### src/rag_app/reranking/rerank_service.py
Purpose:
- Build rerank chain and map ranked indices to documents.

Boilerplate:
```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence

from ..prompts.rerank_prompt import build_rerank_prompt


def build_rerank_chain(llm):
    return build_rerank_prompt() | llm | StrOutputParser()
```

---

Create functions:
- `def build_rerank_chain(llm):`
  - `build_rerank_prompt() | llm | StrOutputParser()`

- `def render_numbered_documents(candidate_docs: list) -> str`
  - `"1. ...\n2. ..."`

- `def rerank_documents(question: str, candidate_docs: list, rerank_chain, top_n: int) -> tuple[list, list[RerankItem]]`
  - Invoke rerank chain
  - Parse payload
  - Map `doc_index -> doc`
  - Fallback to first `top_n` on parse failure
  - Return `(reranked_docs, ranked_items)`

Internal methods:
- `_map_ranked_items_to_docs(...)`
- `_fallback_top_n(...)`

---

### src/rag_app/chains/answer_chain.py
Purpose:
- Build CSDC from answer prompt + llm.

Boilerplate:
```python
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from ..prompts.answer_prompt import build_answer_prompt


def build_csdc(llm):
    return create_stuff_documents_chain(llm, build_answer_prompt())
```

---

Create function:
- `def build_csdc(llm):`
  - Uses `create_stuff_documents_chain(llm, build_answer_prompt())`

---

### src/rag_app/chains/rerank_retrieval_chain.py
Purpose:
- Compose rerank-aware retriever runnable and CRC.

Create function:
- `def build_rerank_crc(hybrid_retriever, llm, settings: AppSettings):`

Inside this function:
1. Build rerank chain
2. Define `rerank_retrieve(inputs: dict)` internal function:
   - `question = inputs["input"]`
   - `candidate_docs = hybrid_retriever.invoke(question)`
   - rerank with `rerank_documents(...)`
   - return reranked docs list
3. Wrap with `RunnableLambda(rerank_retrieve)`
4. Build CSDC via `build_csdc(llm)`
5. Build CRC via `create_retrieval_chain(rerank_retriever, csdc)`
6. Return CRC runnable

---

### src/rag_app/pipeline/query_pipeline.py
Purpose:
- Orchestrate all modules into one callable query API.

Boilerplate:
```python
from ..config.settings import AppSettings


def run_query(question: str, settings: AppSettings | None = None):
    if settings is None:
        settings = AppSettings.from_env()

    # build runtime here
    # invoke chain here
    return {"question": question, "answer": ""}
```

---

Create functions:
- `def load_chunks_from_data(data_dir: str) -> list`
  - Load PDFs via `PyMuPDFLoader`
  - Split via `RecursiveCharacterTextSplitter`
  - Preserve metadata (`source`, `chunk_index`, `page_number`)

- `def build_runtime(settings: AppSettings):`
  - embeddings
  - vector store
  - chunks
  - dense retriever
  - sparse retriever
  - hybrid retriever
  - llm
  - rerank CRC
  - return runtime object/dict

- `def run_query(question: str, settings: AppSettings | None = None) -> QueryResponse`
  - Build/load runtime
  - Invoke CRC with `{ "input": question }`
  - Convert output to `QueryResponse`

Internal methods:
- `_collect_source_refs(docs: list) -> list[SourceRef]`
- `_validate_question(question: str)`

---

### scripts/run_query.py
Purpose:
- Command-line runner for end-to-end query.

Create:
- `def main():`
  - argparse for question
  - call `run_query(question)`
  - print answer
  - print sources and counts

CLI behavior:
- Exit code `0` on success
- Exit code `1` with readable message on failure

---

## 4) Construction Order (Important)

Implement in this order:
1. `settings.py`
2. `init_llm.py`
3. `answer_prompt.py`, `rerank_prompt.py`
4. `dense_retriever.py`
5. `hybrid_retriever.py`
6. `rerank_schema.py`, `response_schema.py`
7. `rerank_parser.py`
8. `rerank_service.py`
9. `answer_chain.py`
10. `rerank_retrieval_chain.py`
11. `query_pipeline.py`
12. `run_query.py`
13. tests

Reason:
- Each module depends only on earlier layers.

## 5) Tests To Add (Minimum)

### tests/test_settings.py
- env parsing defaults
- required API key validation
- invalid `top_k` constraints

### tests/test_retrieval.py
- dense retriever creation
- hybrid retriever creation
- weight behavior sanity

### tests/test_rerank_parser.py
- valid JSON parse
- malformed JSON returns fallback signal
- uniqueness and top_n enforcement

### tests/test_rerank_service.py
- rerank maps indices correctly
- parse failure returns top_n fallback

### tests/test_pipeline_smoke.py
- one known query returns non-empty answer
- response includes source refs

## 6) Production Rules

- Keep prompts pure (no business logic in prompt files).
- Keep LLM init centralized in one module.
- Keep the first version simple: use one source/collection unless you explicitly need multi-source splitting.
- Never trust reranker output without validation.
- Always preserve trace metadata from docs.
- Never allow reranking failure to break final answer flow.
- Keep notebook for experimentation only; src package is runtime truth.

## 7) Mapping From Your Notebook To Modules

Notebook section -> module target:
- PromptTemplate rerank block -> `prompts/rerank_prompt.py` + `reranking/rerank_service.py`
- ChatPromptTemplate answer block -> `prompts/answer_prompt.py`
- Dense retriever block -> `retrieval/dense_retriever.py`
- BM25 + Ensemble block -> `retrieval/hybrid_retriever.py`
- CSDC + CRC block -> `chains/answer_chain.py` + `chains/rerank_retrieval_chain.py`
- Final invoke block -> `pipeline/query_pipeline.py` + `scripts/run_query.py`

## 8) First Run Checklist

1. Create all files exactly as listed.
2. Fill contracts in construction order.
3. Set `.env` values (`OPENAI_API_KEY`, model names, k values).
4. Run tests.
5. Run `python scripts/run_query.py "What did Krishna say about the Mahabharat battle?"`.
6. Compare output with notebook behavior.

If you want, next I can review your first implementation file-by-file against this guide and mark each file as pass/fix with exact corrections.