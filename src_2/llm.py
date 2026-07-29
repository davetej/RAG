from langchain.chat_models.base import init_chat_model, BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from config import AppSettings

def get_chat_model(config: AppSettings) -> BaseChatModel:
    return init_chat_model(
        model=config.chat_model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )


def get_embedding_model(config: AppSettings) -> Embeddings:
    return OpenAIEmbeddings(
        model=config.embedding_model,
        api_key=config.openai_api_key,
    )