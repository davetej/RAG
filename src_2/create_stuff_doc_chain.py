from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from llm import get_chat_model
from config import AppSettings
from prompt import get_rag_final_stage_prompt


def create_stuff_doc_chain(settings: AppSettings):
    return create_stuff_documents_chain(llm=get_chat_model(settings), prompt=get_rag_final_stage_prompt())


    