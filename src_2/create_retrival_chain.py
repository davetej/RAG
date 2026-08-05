from langchain_classic.chains import create_retrieval_chain
from retriver import intit_basic_retriver
from create_stuff_doc_chain import create_stuff_doc_chain
from config import AppSettings

def build_retrieval_pipeline(settings: AppSettings):
    retriever = intit_basic_retriver(settings)
    stuff_doc_chain = create_stuff_doc_chain(settings)
    return create_retrieval_chain(retriever=retriever, combine_docs_chain=stuff_doc_chain)