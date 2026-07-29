from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import AppSettings

def split_documents(documents,settings:AppSettings)->list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunck_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return splitter.split_documents(documents)
