from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from RAG.archive.src_2.llm import get_chat_model
from RAG.archive.src_2.config import AppSettings
from RAG.archive.src_2.prompt import get_rag_final_stage_prompt


def create_stuff_doc_chain(settings: AppSettings):
    return create_stuff_documents_chain(llm=get_chat_model(settings), prompt=get_rag_final_stage_prompt())


    