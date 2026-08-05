from config import AppSettings
from langchain_core.prompts import ChatPromptTemplate
from typing import Any, Iterable


class PromptTemplates:

    def __init__(self, settings: AppSettings):
        self.settings = settings
        self._system_prompt = (
            "You are a helpful RAG assistant. "
            "Answer only from the provided context. "
            "If the context is insufficient, say you do not have enough information. "
            "Be concise and include source references when available."
        )

    def get_rag_system_prompt(self) -> str:
        """System guardrails for grounded answering."""
        return self._system_prompt

    def get_rag_user_template(self) -> str:
        """User-facing prompt template with placeholders."""
        return (
            "Question:\n{input}\n\n"
            "Retrieved Context:\n{context}\n\n"
            "Instructions:\n"
            "1) Use only the retrieved context.\n"
            "2) If unsure, say: I do not have enough information in the provided context.\n"
            "3) Keep answer clear and concise.\n"
            "4) Add a short Sources section from metadata/source titles if provided."
        )

    def get_rag_answer_prompt(self) -> ChatPromptTemplate:
        """Combined chat prompt for answer generation."""
        return ChatPromptTemplate.from_messages(
            [
                ("system", self.get_rag_system_prompt()),
                ("human", self.get_rag_user_template()),
            ]
        )

    def format_context(
        self,
        chunks: Iterable[Any],
        max_chars_per_chunk: int = 1200,
        max_total_chars: int = 6000,
    ) -> str:
        """Create a bounded, traceable context block from retrieved chunks."""
        formatted_lines = []
        total_chars = 0

        for idx, chunk in enumerate(chunks, start=1):
            if isinstance(chunk, dict):
                text = chunk.get("page_content") or chunk.get("content") or chunk.get("text") or ""
                metadata = chunk.get("metadata") or {}
            else:
                text = getattr(chunk, "page_content", "") or getattr(chunk, "content", "") or ""
                metadata = getattr(chunk, "metadata", {}) or {}

            if not isinstance(metadata, dict):
                metadata = {}

            cleaned_text = " ".join(str(text).split())
            if not cleaned_text:
                continue

            if max_chars_per_chunk > 0 and len(cleaned_text) > max_chars_per_chunk:
                cleaned_text = cleaned_text[: max_chars_per_chunk - 3].rstrip() + "..."

            source = metadata.get("source", "unknown")
            page = metadata.get("page_number", metadata.get("page", "n/a"))
            line = f"[{idx}] source={source}, page={page}\n{cleaned_text}"

            projected_total = total_chars + len(line) + (2 if formatted_lines else 0)
            if max_total_chars > 0 and projected_total > max_total_chars:
                break

            formatted_lines.append(line)
            total_chars = projected_total

        if not formatted_lines:
            return "No relevant context was retrieved."

        return "\n\n".join(formatted_lines).strip()