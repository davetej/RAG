
from dataclasses import dataclass
import os
import re

from dotenv import load_dotenv




load_dotenv()


def _required_env(name: str) -> str:
	value = os.getenv(name)
	if value is None or value.strip() == "":
		raise ValueError(f"Missing required environment variable: {name}")
	return value.strip()


def _required_int(name: str) -> int:
	value = _required_env(name)
	try:
		return int(value)
	except ValueError as exc:
		raise ValueError(f"Environment variable {name} must be an integer") from exc


def _required_float(name: str) -> float:
	value = _required_env(name)
	try:
		return float(value)
	except ValueError as exc:
		raise ValueError(f"Environment variable {name} must be a float") from exc


def _required_bool(name: str) -> bool:
	value = _required_env(name).lower()
	if value in {"1", "true", "yes", "on"}:
		return True
	if value in {"0", "false", "no", "off"}:
		return False
	raise ValueError(f"Environment variable {name} must be one of 1,0,true,false,yes,no,on,off")


@dataclass(frozen=True, slots=True)
class AppSettings:
	openai_api_key: str
	chat_model: str
	embedding_model: str
	embedding_dimensions: int
	dense_index_name: str
	sparse_index_name: str
	collection_name: str
	data_path: str
	temperature: float
	max_tokens: int
	chunk_size: int
	chunk_overlap: int
	k: int
	vectorestore_path: str
	vectorstore_collection_name: str
	ingestion_batch_size: int
	mongodb_uri: str
	db_name: str
	user_filter_field: str
	enforce_user_filter: bool
	log_level: str
	reranker_model: str
	rerank_top_k: int

	@classmethod
	def from_env(cls) -> "AppSettings":
		return cls(
			openai_api_key=_required_env("OPENAI_API_KEY"),
			chat_model=_required_env("CHAT_MODEL"),
			embedding_model=_required_env("EMBEDDING_MODEL"),
			embedding_dimensions=_required_int("EMBEDDING_DIMENSIONS"),
			dense_index_name=_required_env("DENSE_INDEX_NAME"),
			sparse_index_name=_required_env("SPARSE_INDEX_NAME"),
			collection_name=_required_env("COLLECTION_NAME"),
			data_path=_required_env("DATA_PATH"),
			temperature=_required_float("TEMPERATURE"),
			max_tokens=_required_int("MAX_TOKENS"),
			k=_required_int("TOP_K"),
			chunk_size=_required_int("CHUNK_SIZE"),
			chunk_overlap=_required_int("CHUNK_OVERLAP"),
			vectorestore_path=_required_env("VECTORSTORE_PATH"),
			vectorstore_collection_name=_required_env("VECTORSTORE_COLLECTION_NAME"),
			ingestion_batch_size=_required_int("INGESTION_BATCH_SIZE"),
			mongodb_uri=_required_env("MONGODB_URI"),
			db_name=_required_env("DB_NAME"),
			user_filter_field=_required_env("USER_FILTER_FIELD"),
			enforce_user_filter=_required_bool("ENFORCE_USER_FILTER"),
			log_level=_required_env("LOG_LEVEL"),
			reranker_model=_required_env("RERANKER_MODEL"),
			rerank_top_k=_required_int("RERANK_TOP_K"),
		)

    