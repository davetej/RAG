# RAG Module Plan

This document describes a clean starting structure for building a MongoDB-based hybrid retriever and a basic RAG flow.

## 1. Core idea

Split the project into small modules so each one has one responsibility.

## 2. Suggested module layout

- `llm.py` -> LLM service class
  - initialize LLM
  - initialize chat model

- `loadpdf.py` -> document loading
  - load PDF files
  - extract text content

- `splitting.py` -> text splitting
  - split large documents into chunks

- `semanticchunking.py` -> semantic chunking
  - create smarter chunks when needed

- `embedding.py` -> embedding service
  - initialize embedding model
  - create embeddings
  - load embeddings

- `vectorstore.py` -> vector store service
  - create vector store
  - load vector store
  - save documents
  - load documents

- `retriever.py` -> retrieval service
  - create basic retriever
  - load basic retriever
  - load ensemble/hybrid retriever

- `create_stuff_document_chain.py` -> document chain service
  - create a chain that combines retrieved documents with the prompt

- `retrieval_chain.py` -> retrieval chain service
  - create the full retrieval chain

## 3. Recommended class structure

### LLM service
- `LLMService`
- methods:
  - `initialize_llm()`
  - `initialize_chat_model()`

### Ingestion service
- `IngestionService`
- methods:
  - `load_pdf()`
  - `load_documents()`

### Chunking service
- `ChunkingService`
- methods:
  - `split_text()`
  - `semantic_chunk()`

### Embedding service
- `EmbeddingService`
- methods:
  - `initialize_embedding_model()`
  - `create_embeddings()`
  - `load_embeddings()`

### Vector store service
- `VectorStoreService`
- methods:
  - `create_vectorstore()`
  - `load_vectorstore()`
  - `save_documents()`
  - `load_documents()`

### Retriever service
- `BasicRetrieverService`
- `HybridRetrieverService`
- methods:
  - `create_retriever()`
  - `load_retriever()`
  - `load_ensemble_retriever()`

### Chain service
- `DocumentChainService`
- `RetrievalChainService`
- methods:
  - `create_stuff_document_chain()`
  - `create_retrieval_chain()`

## 4. Suggested development order

1. Config
2. Ingestion
3. Chunking
4. Embedding
5. Vector store
6. Basic retriever
7. Hybrid retriever
8. Retrieval chain
9. App runner / entrypoint

## 5. Simple production advice

- Keep each module focused on one responsibility.
- Use classes instead of many disconnected functions.
- Pass configuration through `self` or constructor arguments.
- Keep the retrieval logic separate from the prompt logic.
- Build basic retrieval first, then add hybrid retrieval.

## 6. First milestone

The first working version should support:

- load PDF
- split text
- create embeddings
- store documents
- retrieve relevant chunks
- pass them to an LLM

That is enough to build a working RAG flow before adding too much complexity.
