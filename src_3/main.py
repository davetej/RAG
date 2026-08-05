from config import AppSettings
from typing import Any, Dict, Optional
from retriverChain import RetriverService
from loadPdf import PDFLoader
from spilitting import DocumentSplitter
from vectorStore import VectorStoreService
from stuffDocumentChain import StuffDocumentChain
from logging_utils import configure_logging, get_logger


logger = get_logger(__name__)

class RagHandler:
    def __init__(self, settings: AppSettings):
        configure_logging(settings.log_level)
        self.settings = settings
        logger.info("Initializing RAG handler")
        self.retriever_service = RetriverService(settings)
        self.vector_store_service = VectorStoreService(settings)
        infra = self.vector_store_service.ensure_infrastructure()
        logger.info(
            "Infrastructure ready db=%s db_created=%s collection=%s created=%s created_indexes=%s",
            infra.get("db_name"),
            infra.get("database_created"),
            infra.get("collection"),
            infra.get("collection_created"),
            infra.get("created_indexes"),
        )

    def ensureStorage(self) -> Dict[str, Any]:
        logger.info("Manual storage ensure requested")
        return self.vector_store_service.ensure_infrastructure()

    # Below method ius to run the query flow
    def queryRag(self, user_id: str, query: str):
        logger.info("Running query user_id=%s query_len=%s", user_id, len(query or ""))
        retriver_chain = self.retriever_service.retrival_chain(user_id=user_id)
        response = retriver_chain.invoke({"input": query})
        logger.info("Query completed user_id=%s", user_id)
        return response['answer']

    # Below method is to process the pdf file and store the embeddings in the vector store
    def processFile(self, file_path: str, user_id: str):
        logger.info("Starting ingestion file_path=%s user_id=%s", file_path, user_id)
        documents = PDFLoader(self.settings).load_pdf_file(file_path=file_path, user_id=user_id) #loading
        logger.info("Loaded documents count=%s", len(documents))

        # splitting the documents into chunks
        splitter_obj = DocumentSplitter(settings=self.settings)
        chunks = splitter_obj.split_documents(documents)
        logger.info("Split documents into chunks count=%s", len(chunks))


        # Embedding and storing the chunks in the vector store
        result = self.vector_store_service.create_vector_store(documents=chunks)
        logger.info(
            "Vector store write completed attempted=%s inserted=%s acknowledged=%s",
            result.get("attempted"),
            result.get("inserted"),
            result.get("acknowledged"),
        )

    def truncateVectorCollection(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        logger.warning("Truncate requested user_id=%s", user_id if user_id else "ALL")
        result = self.vector_store_service.truncate_vector_collection(user_id=user_id)
        logger.info(
            "Truncate completed scope=%s deleted=%s acknowledged=%s",
            result.get("scope"),
            result.get("deleted_count"),
            result.get("acknowledged"),
        )
        return result

       

if __name__ == "__main__":
    settings = AppSettings.from_env()


    rag_handler = RagHandler(settings)
    # rag_handler.processFile(file_path="D:\\2027\\Projects\\RAG\\data\\atomic_habit.pdf", user_id="test_user")

    
    query = "What is Habit Stacking?"
    answer = rag_handler.queryRag(user_id="test_user", query=query)
    logger.info("Answer: %s", answer)   