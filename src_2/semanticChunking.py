from config import AppSettings
from langchain_experimental.text_splitter import SemanticChunker
from llm import get_embedding_model

def init_semantic_chunker(app_settings: AppSettings) -> SemanticChunker:
    embedding_model = get_embedding_model(app_settings)
    chunker = SemanticChunker(
        embeddings=embedding_model,
        min_chunk_size=app_settings.chunck_size,
    )
    return chunker
