
from config import AppSettings
from langchain_mongodb import MongoDBAtlasVectorSearch
from vectorStore import VectorStoreService
from embedding import EmbeddingsService
from langchain_core.documents import Document
from langchain_mongodb.retrievers import MongoDBAtlasHybridSearchRetriever
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict
from typing import List
from langchain_classic.chains import create_retrieval_chain
from stuffDocumentChain import StuffDocumentChain
from logging_utils import get_logger
from reranking import Reranker


logger = get_logger(__name__)


class RerankedRetriever(BaseRetriever):
    """Wrap an existing retriever and rerank documents before they reach the LLM."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    base_retriever: BaseRetriever
    reranker: Reranker

    def _get_relevant_documents(self, query: str):
        # Step 1: Get the initial documents from the base retriever.
        docs = self.base_retriever.invoke(query)
        print(f"[Rerank] Fetched {len(docs)} documents before reranking")

        # Step 2: Rerank documents based on the query.
        reranked_docs = self.reranker.rerank_documents(query, docs)
        print(f"[Rerank] Kept {len(reranked_docs)} documents after reranking")

        # Step 3: Return the reranked documents.
        return reranked_docs


# def _print_retrieval_summary(label: str, docs: List[Document], limit: int = 5) -> None:
#     print(f"\n=== {label} (top {min(limit, len(docs))}/{len(docs)}) ===")
#     for rank, doc in enumerate(docs[:limit], start=1):
#         metadata = doc.metadata if isinstance(doc.metadata, dict) else {}
#         nested_metadata = metadata.get("metadata") if isinstance(metadata.get("metadata"), dict) else metadata

#         chunk_id = nested_metadata.get("chunk_id", metadata.get("_id", "n/a"))
#         source = nested_metadata.get("source", "unknown")
#         page = nested_metadata.get("page_number", nested_metadata.get("page", "n/a"))
#         vector_score = metadata.get("vector_score", "n/a")
#         fulltext_score = metadata.get("fulltext_score", "n/a")
#         search_score = metadata.get("score", "n/a")

#         print(
#             f"#{rank} chunk_id={chunk_id} | source={source} | page={page} | "
#             f"vector_score={vector_score} | fulltext_score={fulltext_score} | score={search_score}"
#         )

class RetriverService:
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.vector_store_service = VectorStoreService(settings)
        self.embeddings_service = EmbeddingsService(settings)
        logger.info("Retriever service initialized")

    def _get_collection(self):
        return self.vector_store_service.mongo_client_instance[self.settings.db_name][self.settings.collection_name]

    def _get_vector_store(self) -> MongoDBAtlasVectorSearch:
        logger.debug("Creating MongoDBAtlasVectorSearch index=%s", self.settings.dense_index_name)
        return MongoDBAtlasVectorSearch(
            collection=self._get_collection(),
            embedding=self.embeddings_service.get_embedding_model(),
            index_name=self.settings.dense_index_name,
            text_key="text",
            embedding_key="embedding",
        )

    # def _dense_retriever(self) -> BaseRetriever:
    #     k = self.settings.k
    #     logger.debug("Creating dense retriever k=%s", k)
    #     return self._get_vector_store().as_retriever(search_kwargs={"k": k})

    # def _sparse_retriever_debug(self, query: str,user_id:str) -> List[Document]:
    #     k = self.settings.k
    #     logger.info("Running sparse debug retrieval user_id=%s k=%s", user_id, k)
       
    #     pipeline = [
    #                     {
    #                         "$search": {
    #                             "index": self.settings.sparse_index_name,
    #                             "compound": {
    #                                 "must": [
    #                                     {"text": {"query": query, "path": "text"}},
    #                                     {"equals": {"path": "metadata.user_id", "value": user_id}},
    #                                 ]
    #                             },
    #                         }
    #                     },
    #                     {"$limit": k},
    #                     {"$project": {"_id": 0, "text": 1, "metadata": 1, "score": {"$meta": "searchScore"}}},
    #             ]
    #     results = []
    #     collection = self._get_collection()
    #     for row in collection.aggregate(pipeline):
    #         results.append(
    #             Document(
    #                 page_content=row.get("text", ""),
    #                 metadata={**row.get("metadata", {}), "score": row.get("score")},
    #             )
    #         )
    #     logger.info("Sparse debug retrieval complete docs=%s", len(results))
    #     return results

    def _ensembled_retriever(self, user_id: str) -> MongoDBAtlasHybridSearchRetriever:
        k = self.settings.k
        logger.info("Creating hybrid retriever user_id=%s k=%s", user_id, k)
        return MongoDBAtlasHybridSearchRetriever(
            vectorstore=self._get_vector_store(),
            search_index_name=self.settings.sparse_index_name,
            k=k,
            vector_penalty=60.0,
            fulltext_penalty=60.0,
            vector_weight=0.7,
            fulltext_weight=0.3,
           pre_filter={"metadata.user_id": user_id},
        )


    # def _retrieve_hybrid(self, query: str, user_id: str) -> List[Document]:
    #     k = self.settings.k
    #     logger.info("Running hybrid retrieval user_id=%s k=%s query_len=%s", user_id, k, len(query or ""))
    #     docs = self._ensembled_retriever(user_id=user_id).invoke(query)
    #     logger.info("Hybrid retrieval complete docs=%s", len(docs))
    #     return docs
    
    def retrival_chain(self, user_id: str) -> create_retrieval_chain:
        k = self.settings.k
        logger.info("Building retrieval chain user_id=%s k=%s", user_id, k)

        # Step 1: Create the original hybrid retriever.
        retriever = self._ensembled_retriever(user_id=user_id)

        # Step 2: Wrap it with a reranking layer.
        # This makes the chain retrieve documents first, then rerank them,
        # and only then pass the best documents to the LLM/combine step.
        reranked_retriever = RerankedRetriever(
            base_retriever=retriever,
            reranker=Reranker(self.settings),
        )

        csdc = StuffDocumentChain(settings=self.settings).create_stuff_documents_chain()
        return create_retrieval_chain(retriever=reranked_retriever, combine_docs_chain=csdc)

if __name__ == "__main__":
    settings = AppSettings.from_env()
    retriever_service = RetriverService(settings)
    query = "What is Habit stacking?"

    # dense_docs = retriever_service._dense_retriever().invoke(query)
    # sparse_docs = retriever_service._sparse_retriever_debug(query=query, user_id="test_user")
    # hybrid_docs = retriever_service._retrieve_hybrid(query=query, user_id="test_user")

    # _print_retrieval_summary("Dense Retriever", dense_docs, limit=5)
    # _print_retrieval_summary("Sparse Retriever (Debug)", sparse_docs, limit=5)
    # _print_retrieval_summary("Hybrid Retriever", hybrid_docs, limit=5)

    #  Below documents will be returned from mongo db based on query input.
    retriver_chain = retriever_service.retrival_chain(user_id="test_user")
    response = retriver_chain.invoke({"input": query})

    print("\n=== Final answer ===")
    if isinstance(response, dict):
        print(response.get("answer", response))
    else:
        print(response)
    