from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from config import AppSettings
from llm import LLMService
from  promptTemplates import PromptTemplates
from langchain_core.runnables import Runnable
from logging_utils import get_logger


logger = get_logger(__name__)

class StuffDocumentChain:
    def __init__(self, settings: AppSettings ):
        self.settings = settings

    def create_stuff_documents_chain(self) -> Runnable:
        logger.info("Creating stuff documents chain")
        llm = LLMService(self.settings).init_chat_model()        
        prompt = PromptTemplates(self.settings).get_rag_answer_prompt() 
        return create_stuff_documents_chain(
            llm=llm,
            document_variable_name="context",
            prompt=prompt
        )
    