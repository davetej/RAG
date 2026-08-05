from config import AppSettings
from llm import LLMService
class EmbeddingsService:

    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.llm_service = LLMService(settings)

    def get_embedding_model(self):
        return self.llm_service._init_embedding_model()