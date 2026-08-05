from typing import List, Optional
from llm import LLMService
from config import AppSettings
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from logging_utils import get_logger


logger = get_logger(__name__)


class DocumentSplitter:
    def __init__(self, settings: AppSettings, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None):
        self.settings = settings
        self.chunk_size = chunk_size if chunk_size is not None else settings.chunk_size
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
        logger.info(
            "Initializing document splitter chunk_size=%s chunk_overlap=%s",
            self.chunk_size,
            self.chunk_overlap,
        )

        self.text_splitter = SemanticChunker(
            embeddings=LLMService(self.settings)._init_embedding_model(),
            min_chunk_size=self.chunk_size,
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        logger.info("Splitting documents count=%s", len(documents))
        chunks = self.text_splitter.split_documents(documents)
        logger.info("Document splitting completed chunks=%s", len(chunks))
        return chunks


if __name__ == "__main__":
    from loadPdf import PDFLoader
    from config import AppSettings

    settings = AppSettings.from_env()

     
    loader = PDFLoader(settings)
    docs = loader.load_pdf_file("D:\\2027\\Projects\\RAG\\data\\BhagavadGita.pdf", user_id="test_user")

    splitter_obj = DocumentSplitter(settings=settings, chunk_size=settings.chunk_size, 
                                chunk_overlap=settings.chunk_overlap)
    chunks = splitter_obj.split_documents(docs)

    logger.info("Total chunks created: %s", len(chunks))