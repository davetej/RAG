from pathlib import Path

from langchain_chroma import Chroma

from ..config.settings import AppSettings


def create_vector_store(settings: AppSettings, embeddings):
    persist_directory = Path(settings.persist_directory)
    persist_directory.mkdir(parents=True, exist_ok=True)

    return Chroma(
        persist_directory=str(persist_directory),
        embedding_function=embeddings,
        collection_name=settings.collection_name,
    )

def build_dense_retriever(vector_store, settings):
    search_kwargs = {
        "k": settings.top_k_retrieve
    }
    return vector_store.as_retriever(search_kwargs=search_kwargs)