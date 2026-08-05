import os
from sentence_transformers import CrossEncoder
from config import AppSettings


class Reranker:
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.model_name = settings.reranker_model
        self.reranker = self._load_reranker(self.model_name)

    def _load_reranker(self, model_name: str):
        candidate_paths = []

        if os.path.isabs(model_name):
            candidate_paths.append(model_name)
        else:
            candidate_paths.append(model_name)
            candidate_paths.append(os.path.join(os.getcwd(), model_name))
            candidate_paths.append(os.path.join(os.path.dirname(__file__), model_name))

            if "/" in model_name:
                repo_name = model_name.split("/")[-1]
                candidate_paths.append(os.path.join(os.getcwd(), repo_name))
                candidate_paths.append(os.path.join(os.path.dirname(__file__), repo_name))

        for path in candidate_paths:
            if os.path.isdir(path):
                print(f"[Rerank] Loading local reranker model from: {path}")
                return CrossEncoder(path, local_files_only=True)

        print(f"[Rerank] Loading reranker model from Hugging Face Hub: {model_name}")
        return CrossEncoder(model_name)

    def rerank_documents(self, query: str, documents: list, reranked_doc = 10) -> list:
        """Rerank documents based on the query using a cross-encoder model."""
        if not documents:
            return []

        reranked_doc = max(1, int(reranked_doc))

        # Prepare pairs of (query, document_text) for scoring
        pairs = [(query, doc.page_content) for doc in documents]
        
        # Get scores from the reranker model
        scores = self.reranker.predict(pairs)
        
        # Combine documents with their scores and sort by score in descending order
        scored_documents = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
        
        # Return only the documents, now sorted by relevance
        return [doc for score, doc in scored_documents[:reranked_doc]]