from config import AppSettings
from vectorstore import load_vectorstore, VectorStore
from langchain_core.retrievers import BaseRetriever
from langchain_community.retrievers import BM25Retriever

#  building retriever part now

def init_basic_retriever(vectorstore: VectorStore, settings: AppSettings) -> BaseRetriever:
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.k}
    )
