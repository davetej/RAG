from langchain_text_splitters import RecursiveCharacterTextSplitter
from RAG.archive.src_2.semanticChunking import init_semantic_chunker
from RAG.archive.src_2.config import AppSettings

def split_documents(documents,settings:AppSettings)->list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunck_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return splitter.split_documents(documents)


def semantic_split_documents(documents, settings: AppSettings) -> list:
    
    chunker = init_semantic_chunker(settings)
    return chunker.split_documents(documents)