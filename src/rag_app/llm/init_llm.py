from langchain.chat_models import ChatOpenAI
from ..config.settings import AppSettings

def create_chat_llm(settings: AppSettings):
    return ChatOpenAI(
        model_name=settings.chat_model,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        openai_api_key=settings.openai_api_key,
    )

def create_embeddings(settings: AppSettings):
    from langchain.embeddings import OpenAIEmbeddings
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
    )