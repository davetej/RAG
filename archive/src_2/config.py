
from dataclasses import dataclass
import os


from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True, slots=True)
class AppSettings:
    openai_api_key: str
    chat_model: str = 'gpt-5.4-nano'
    embedding_model: str = "text-embedding-3-large"
    collection_name: str = "book_store"
    data_path: str = "data"
    temperature: float = 0.0
    max_tokens: int = 512
    chunck_size: int = 1000
    chunk_overlap: int = 200
    k: int = 5
    vectorestore_path: str = "vectorstore"
    vectorstore_collection_name: str = "book_store"
    ingestion_batch_size: int = 500  

    @classmethod
    def from_env(cls) -> "AppSettings":
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            chat_model=os.getenv("CHAT_MODEL", 'gpt-5.4-nano'),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
            collection_name=os.getenv("COLLECTION_NAME", "book_store"),
            data_path=os.getenv("DATA_PATH", "data"),
            temperature=float(os.getenv("TEMPERATURE", "0.0")),
            max_tokens=int(os.getenv("MAX_TOKENS", "512")),
            k=int(os.getenv("TOP_K", "5")),
            chunck_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
            vectorestore_path=os.getenv("VECTORSTORE_PATH", "vectorstore"),
            vectorstore_collection_name=os.getenv("VECTORSTORE_COLLECTION_NAME", "book_store"),
            ingestion_batch_size=int(os.getenv("INGESTION_BATCH_SIZE", "500"))
        )

    