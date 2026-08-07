# RAG Learning Summary (So Far)

This file summarizes the concepts already covered in this project from notebooks, experiments, and modular code drafts.

## 1) RAG Foundations

- What Retrieval-Augmented Generation (RAG) is and why retrieval improves LLM answers.
- End-to-end RAG flow:
  1. Ingest data
  2. Chunk/split text
  3. Create embeddings
  4. Store vectors + metadata
  5. Retrieve relevant chunks
  6. Generate answer with context

## 2) Data Ingestion (Multi-source)

Covered ingestion patterns for:

- PDF
- Text files
- CSV / Excel
- JSON / JSONL
- SQL/database-style sources

Key learning:
- Normalize documents into a common `Document` shape with text + metadata.

## 3) Text Splitting / Chunking

- Basic chunking (size + overlap style splitting)
- Semantic chunking experiments

Key learning:
- Chunking quality strongly affects retrieval recall and answer quality.

## 4) Embeddings + Vector Stores

- OpenAI embedding usage
- Local vector storage with Chroma
- MongoDB Atlas vector field design for hybrid workloads

Key learning:
- Embedding model choice and chunk granularity impact semantic search behavior.

## 5) Retriever Types Practiced

### Dense retrieval
- Vector similarity search
- Top-k nearest chunks

### Sparse / lexical retrieval
- TF-IDF
- BM25
- MongoDB Atlas full-text search

### Hybrid retrieval
- Combine dense + sparse outputs
- Ensemble-style retrieval and rank fusion mindset

Key learning:
- Hybrid retrieval is usually more robust than dense-only for misspellings, exact terms, and domain keywords.

## 6) Ranking Improvements

- Reranking with Cross-Encoder
- MMR for diversity-aware selection
- Merge/ensemble retrieval exploration

Key learning:
- First retrieval gets candidates; reranking improves precision for final LLM context.

## 7) Query Understanding Enhancements

- Multi-query retriever (query expansion)
- Self-query retriever (intent + metadata style filtering)
- History-aware retriever for multi-turn chat

Key learning:
- Query reformulation and conversation memory improve recall and user experience.

## 8) Chain Composition in LangChain

- Stuff Document Chain creation
- Retrieval Chain creation
- Moving toward reusable pipeline composition

Key learning:
- Keep retrieval, prompting, and LLM orchestration separate for maintainability.

## 9) Caching Concepts Explored

- MongoDB LLM cache
- MongoDB semantic caching
- Redis semantic caching

Key learning:
- Caching reduces latency and cost for repeated or semantically similar requests.

## 10) Metadata, Indexing, and Search Ops

- Source/page metadata tracking
- Chunk IDs and traceability
- Atlas index setup (vector index + full-text index)

Key learning:
- Good metadata and index design are essential for filtering, debugging, and observability.

## 11) Architecture Maturity Progress

Current trajectory in repo:

- Notebook-first experimentation ✅
- Split modules in `src_2/` and `src_3/` ✅
- Production-oriented blueprint documented ✅

Key architectural concept learned:
- Move from exploratory notebooks to modular services (config, ingestion, embedding, retrieval, reranking, chains, pipeline).

## 12) Practical Outcome Achieved

You now have the building blocks to create:

- Single-retriever RAG
- Hybrid RAG (dense + sparse)
- Hybrid + rerank RAG
- History-aware conversational RAG
- Cache-enabled RAG

---

## Suggested Next Learning Milestones

1. Add evaluation metrics (Recall@k, MRR, latency, cost)
2. Build reproducible benchmark dataset from `data/`
3. Convert one end-to-end notebook path into a tested `src/rag_app` package
4. Add API + simple UI for demo-ready deployment

---

## One-line Summary

This project already covers the full practical RAG journey: **ingestion → chunking → embeddings → dense/sparse/hybrid retrieval → reranking → conversational retrieval → caching → modular production design**.
