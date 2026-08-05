from typing import Any, Dict, Iterable, List, Optional

from config import AppSettings
from pymongo import MongoClient
from pymongo.operations import InsertOne
from pymongo.server_api import ServerApi
from llm import LLMService
from pymongo.operations import SearchIndexModel
from logging_utils import get_logger


logger = get_logger(__name__)

class VectorStoreService:
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.embedding_model: Optional[Any] = None
        self.mongo_client_instance = self.mongo_client()

    def _get_embedding_model(self) -> Any:
        if self.embedding_model is None:
            logger.debug("Initializing embedding model")
            self.embedding_model = LLMService(self.settings)._init_embedding_model()
        return self.embedding_model

    def mongo_client(self) -> MongoClient:
        logger.info("Connecting to MongoDB")
        client = MongoClient(self.settings.mongodb_uri, server_api=ServerApi('1'))
        logger.info(
            "Connected to MongoDB db=%s collection=%s",
            self.settings.db_name,
            self.settings.collection_name,
        )
        return client

    def _get_collection(self):
        db = self.mongo_client_instance[self.settings.db_name]
        return db[self.settings.collection_name]

    def ensure_database_and_collection(self) -> Dict[str, Any]:
        db_name = self.settings.db_name
        collection_name = self.settings.collection_name

        existing_databases = set(self.mongo_client_instance.list_database_names())
        database_created = db_name not in existing_databases

        db = self.mongo_client_instance[db_name]
        existing_collections = set(db.list_collection_names())

        collection_created = False
        if collection_name not in existing_collections:
            db.create_collection(collection_name)
            collection_created = True

        logger.info(
            "Database/collection ensured db=%s database_created=%s collection=%s collection_created=%s",
            db_name,
            database_created,
            collection_name,
            collection_created,
        )

        return {
            "db_name": db_name,
            "database_created": database_created,
            "collection": collection_name,
            "collection_created": collection_created,
        }

    def ensure_infrastructure(self) -> Dict[str, Any]:
        logger.info("Ensuring Mongo infrastructure")
        storage_summary = self.ensure_database_and_collection()
        db = self.mongo_client_instance[self.settings.db_name]
        collection_name = self.settings.collection_name
        collection = db[collection_name]

        try:
            existing_search_indexes = {
                idx.get("name")
                for idx in collection.list_search_indexes()
                if idx.get("name")
            }
        except Exception:
            existing_search_indexes = set()

        created_indexes: List[str] = []

        if self.settings.dense_index_name not in existing_search_indexes:
            self.create_vector_index(vector_index_name=self.settings.dense_index_name)
            created_indexes.append(self.settings.dense_index_name)

        if self.settings.sparse_index_name not in existing_search_indexes:
            self.create_sparse_index(sparse_index_name=self.settings.sparse_index_name)
            created_indexes.append(self.settings.sparse_index_name)

        logger.info(
            "Infrastructure ensured db=%s db_created=%s collection=%s created=%s new_indexes=%s",
            storage_summary.get("db_name"),
            storage_summary.get("database_created"),
            collection_name,
            storage_summary.get("collection_created"),
            created_indexes,
        )

        return {
            "db_name": storage_summary.get("db_name"),
            "database_created": storage_summary.get("database_created"),
            "collection": collection_name,
            "collection_created": storage_summary.get("collection_created"),
            "created_indexes": created_indexes,
            "dense_index_name": self.settings.dense_index_name,
            "sparse_index_name": self.settings.sparse_index_name,
        }

    def create_vector_store(self, documents: Iterable[Any]) -> Dict[str, Any]:
        # Generate embeddings in batches to reduce API round trips.
        collection = self._get_collection()
        embedding_model = self._get_embedding_model()
        logger.info("Creating vector store records")

        docs = list(documents)
        if not docs:
            logger.warning("No documents to insert into vector store")
            return {
                "collection": self.settings.collection_name,
                "attempted": 0,
                "inserted": 0,
                "acknowledged": True,
            }

        batch_size = max(1, self.settings.ingestion_batch_size)
        logger.info("Embedding documents in batches batch_size=%s total_docs=%s", batch_size, len(docs))

        write_requests = []
        for start in range(0, len(docs), batch_size):
            batch_docs = docs[start : start + batch_size]
            batch_texts = [doc.page_content for doc in batch_docs]
            batch_embeddings = embedding_model.embed_documents(batch_texts)

            if len(batch_embeddings) != len(batch_docs):
                raise ValueError(
                    f"Embedding response size mismatch: expected {len(batch_docs)}, got {len(batch_embeddings)}"
                )

            for doc, embedding in zip(batch_docs, batch_embeddings):
                doc_with_embedding = {
                    "text": doc.page_content,
                    "embedding": embedding,
                    "metadata": doc.metadata if hasattr(doc, 'metadata') else {}
                }
                write_requests.append(InsertOne(doc_with_embedding))

            logger.debug(
                "Embedded batch start=%s size=%s",
                start,
                len(batch_docs),
            )

        if not write_requests:
            logger.warning("No documents to insert into vector store")
            return {
                "collection": self.settings.collection_name,
                "attempted": 0,
                "inserted": 0,
                "acknowledged": True,
            }

        if len(write_requests) != len(docs):
            logger.warning(
                "Prepared requests count does not match input docs input=%s prepared=%s",
                len(docs),
                len(write_requests),
            )

        result = collection.bulk_write(write_requests, ordered=False)
        logger.info("Inserted documents into vector store count=%s", result.inserted_count)

        return {
            "collection": self.settings.collection_name,
            "attempted": len(write_requests),
            "inserted": result.inserted_count,
            "acknowledged": result.acknowledged,
        }
    
    def create_vector_index(self, vector_index_name: str = "vector_search_index") -> List[str]:
        logger.info("Creating vector index name=%s", vector_index_name)

        # Implement the logic to create a search index for the vector store
        collection = self._get_collection()
        
        # Create an index on the embedding field for efficient vector search
        if self.settings.embedding_dimensions <= 0:
            raise ValueError("embedding_dimensions must be a positive integer")

        vector_index = SearchIndexModel(
            definition={
                    "fields": [
                        {
                            "type": "vector",
                            "path": "embedding",
                            "numDimensions": self.settings.embedding_dimensions,
                            "similarity": "cosine",
                        },
                         {
                            "type": "filter",
                            "path": "metadata.user_id",
                        },
                    ]
                },
                name=vector_index_name,
                type="vectorSearch",
        )

        return collection.create_search_indexes([vector_index])


    def create_sparse_index(self, sparse_index_name: str = "sparse_search_index") -> List[str]:
        logger.info("Creating sparse index name=%s", sparse_index_name)
        # Implement the logic to create a sparse index for the vector store
        collection = self._get_collection()
        
        # Create a sparse index on the text field for efficient keyword search.
        sparse_index = SearchIndexModel(
                definition={
                    "mappings": {
                        "dynamic": False,
                        "fields": {
                            "text": {
                                "type": "string",
                                "analyzer": "lucene.standard"
                            },
                            "metadata": {
                                "type": "document",
                                "fields": {
                                    "user_id": {"type": "token"}
                                }
                                }
                            }
                        }
                },
                name=sparse_index_name,
                type="search"
)
        return collection.create_search_indexes([sparse_index])

    def truncate_vector_collection(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        collection = self._get_collection()
        filter_query: Dict[str, Any] = {}
        scope = "all"

        if user_id:
            filter_query = {"metadata.user_id": user_id}
            scope = f"user:{user_id}"

        count_before = collection.count_documents(filter_query)
        result = collection.delete_many(filter_query)

        logger.warning(
            "Truncated vector collection scope=%s deleted=%s",
            scope,
            result.deleted_count,
        )

        return {
            "collection": self.settings.collection_name,
            "scope": scope,
            "count_before": count_before,
            "deleted_count": result.deleted_count,
            "acknowledged": result.acknowledged,
        }
