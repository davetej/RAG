from pathlib import Path
from hashlib import sha256
from datetime import datetime, timezone
from typing import List

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document

from config import AppSettings
from logging_utils import get_logger


logger = get_logger(__name__)


class PDFLoader:
    def __init__(self, settings: AppSettings):
        self.settings = settings

    def _normalize_name(self, text: str) -> str:
        """Create a safe identifier for a document source."""
        return "".join(ch if ch.isalnum() else "_" for ch in text.lower()).strip("_")

    def _file_sha256(self, path: Path, chunk_size: int = 1024 * 1024) -> str:
        """Compute a stable checksum for the source PDF."""
        hasher = sha256()
        with path.open("rb") as handle:
            while True:
                block = handle.read(chunk_size)
                if not block:
                    break
                hasher.update(block)
        return hasher.hexdigest()

    def load_pdf_file(self, file_path: str, user_id :str, category: str = "research") -> List[Document]:
        """Load a PDF and enrich each chunk metadata for downstream vector storage."""
        path = Path(file_path).expanduser().resolve()
        logger.info("Loading PDF file path=%s user_id=%s category=%s", str(path), user_id, category)
        if not path.exists():
            logger.error("PDF file not found path=%s", str(path))
            raise FileNotFoundError(f"PDF file not found: {path}")

        loader = PyMuPDFLoader(str(path))
        documents = loader.load()

        source_checksum = self._file_sha256(path)
        source_id = self._normalize_name(path.stem)
        ingested_at = datetime.now(timezone.utc).isoformat()

        enriched_documents: List[Document] = []
        for index, doc in enumerate(documents):
            chunk_text = (doc.page_content or "").strip()
            if not chunk_text:
                continue

            page_number = (getattr(doc, "metadata", {}) or {}).get("page", index)
            metadata = {
                "source": path.name,
                "page_number": page_number + 1,
                "category": category,
                "source_path": str(path),
                "source_id": source_id,
                "source_checksum": source_checksum,
                "chunk_index": index,
                "chunk_size": len(chunk_text),
                "created_at": ingested_at,
                "user_id": user_id,
            }

            if hasattr(doc, "metadata") and isinstance(doc.metadata, dict):
                doc.metadata.update(metadata)
            else:
                doc.metadata = metadata

            enriched_documents.append(doc)

        logger.info(
            "PDF load complete pages_loaded=%s chunks_retained=%s source=%s",
            len(documents),
            len(enriched_documents),
            path.name,
        )

        return enriched_documents


if __name__ == "__main__":
    settings = AppSettings.from_env()
    loader = PDFLoader(settings)
    docs = loader.load_pdf_file(r"D:\2027\Projects\RAG\data\BhagavadGita.pdf", user_id="test_user")
    logger.info("Loaded %s chunks", len(docs))
