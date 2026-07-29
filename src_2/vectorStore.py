from pathlib import Path
from config import AppSettings
from langchain_chroma import Chroma
from llm import get_embedding_model




def _resolve_persist_dir(settings: AppSettings) -> str:
    path = Path(settings.vectorestore_path)
    if not path.is_absolute():
        path = Path(__file__).parent.parent / settings.vectorestore_path
    return str(path)


def create_vectorstore(settings: AppSettings, documents) -> Chroma:
    """First-time ingestion: embeds and persists documents in batches."""
    embeddings = get_embedding_model(settings)
    store = Chroma(
        persist_directory=_resolve_persist_dir(settings),
        embedding_function=embeddings,
        collection_name=settings.vectorstore_collection_name,
    )
    for i in range(0, len(documents), settings.ingestion_batch_size):
        batch = documents[i : i + settings.ingestion_batch_size]
        store.add_documents(batch)
        print(f"Ingested {min(i + settings.ingestion_batch_size, len(documents))} / {len(documents)}")

    return store


def load_vectorstore(settings: AppSettings) -> Chroma:
    """Load existing store from disk for querying."""
    return Chroma(
        persist_directory=_resolve_persist_dir(settings),
        embedding_function=get_embedding_model(settings),
        collection_name=settings.vectorstore_collection_name,
    )

    