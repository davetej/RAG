from config import AppSettings
from langchain.chat_models.base import BaseChatModel
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from logging_utils import get_logger


logger = get_logger(__name__)


class LLMService:
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.openai_api_key = settings.openai_api_key
        self.chat_model = settings.chat_model
        

    def init_chat_model(self) -> BaseChatModel:

        logger.info("Initializing chat model model=%s", self.chat_model)
        return init_chat_model(
            model=self.chat_model,
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_tokens,
        )

    def _init_embedding_model(self) -> OpenAIEmbeddings:
        return OpenAIEmbeddings(
            model=self.settings.embedding_model,
            api_key=self.settings.openai_api_key,
        )