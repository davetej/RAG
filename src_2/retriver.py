from config import AppSettings
from vectorStore import load_vectorstore
from langchain_core.retrievers import BaseRetriever
#  building retriever part now

def init_retriever(settings: AppSettings) -> BaseRetriever:
    return load_vectorstore(settings).as_retriever(search_kwargs={"k": settings.k})
