from dataclasses import dataclass
import os

@dataclass(frozen=True, slots=True, kw_only=True)
class AppSettings:
    openai_api_key: str
    chat_model: str = "gpt-5.4-nano"
    embedding_model: str = "text-embedding-3-large"
    temperature: float = 0.0
    max_tokens: int = 512
    collection_name: str = "bookstore"
    persist_directory: str = "chroma_store/bookstore"
    top_k_retrieve: int = 10
    top_k_rerank: int = 5
    dense_weight: float = 0.5
    sparse_weight: float = 0.5

    def __post_init__(self) -> None:
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required.")
        if not (0.0 <= self.temperature <= 1.0):
            raise ValueError("temperature must be between 0.0 and 1.0.")
        if self.top_k_rerank < 1:
            raise ValueError("top_k_rerank must be >= 1.")
        if self.top_k_retrieve < self.top_k_rerank:
            raise ValueError("top_k_retrieve must be >= top_k_rerank.")
        if (self.dense_weight + self.sparse_weight) <= 0:
            raise ValueError("dense_weight + sparse_weight must be > 0.")

    @classmethod
    def from_env(cls) -> "AppSettings":
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            chat_model=os.getenv("CHAT_MODEL", "gpt-5.4-nano"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
            temperature=float(os.getenv("TEMPERATURE", "0.0")),
            max_tokens=int(os.getenv("MAX_TOKENS", "512")),
            collection_name=os.getenv("COLLECTION_NAME", "bookstore"),
            persist_directory=os.getenv("PERSIST_DIRECTORY", "chroma_store/bookstore"),
            top_k_retrieve=int(os.getenv("TOP_K_RETRIEVE", "10")),
            top_k_rerank=int(os.getenv("TOP_K_RERANK", "5")),
            dense_weight=float(os.getenv("DENSE_WEIGHT", "0.5")),
            sparse_weight=float(os.getenv("SPARSE_WEIGHT", "0.5")),
        )