from config import AppSettings
from datetime import datetime, timezone
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from fileSha import file_sha256, stable_filename
from uuid import uuid4
from splitter import split_documents
from vectorStore import create_vectorstore

def processPdf(settings: AppSettings, pdf_name: str):
    

    # resolve relative paths against the project root, not the shell cwd
    data_dir = Path(settings.data_path)
    if not data_dir.is_absolute():
        data_dir = Path(__file__).parent.parent / settings.data_path
    pdf_file_path = data_dir / pdf_name
    print(f"Looking for: {pdf_file_path}")

    if not pdf_file_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_file_path}")
    print(f"Found PDF file: {pdf_file_path}. Processing...")

    source_checksum = file_sha256(pdf_file_path)
    source_filename = stable_filename(pdf_file_path, ext=".pdf")

    print('Loading PDF file using PyMuPDFLoader...')
    loader = PyMuPDFLoader(str(pdf_file_path))
    documents = loader.load()

    ingested_at = datetime.now(timezone.utc).isoformat()
    for index, doc in enumerate(documents):
        doc.metadata["source"] = source_filename
        doc.metadata["source_checksum"] = source_checksum
        doc.metadata["page_number"] = index + 1
        doc.metadata["created_at"] = ingested_at

    print(f"Loaded {len(documents)} pages from {pdf_file_path}.")

    chunks = split_documents(documents, settings)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = str(uuid4())
        chunk.metadata["chunk_index"] = index
        chunk.metadata["chunk_size"] = len(chunk.page_content)

    vectorstore = create_vectorstore(settings, chunks) 

    print(f"Vectorstore created and persisted at: {settings.vectorestore_path}")

    return vectorstore