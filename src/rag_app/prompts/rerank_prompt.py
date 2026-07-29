from langchain_core.prompts import PromptTemplate



def build_rerank_prompt() -> PromptTemplate:

    rerank_prompt = PromptTemplate.from_template("""
                You are a reranking assistant.

                Task: Rank the MOST relevant documents to the user question.

                User Question: {question}

                Documents (numbered 1..N):
                {documents}

                Return ONLY valid JSON in this exact schema:
                {{
                "ranked": [
                    {{"rank": 1, "doc_index": 2, "reason": "short reason"}}
                ]
                }}

                Rules:
                - Return exactly top 5 items.
                - rank must be unique integers 1..5.
                - doc_index must reference the numbered documents above (1-based index).
                - Include each document at most once.
                - reason must be <= 20 words.
                - Do not include any text outside the JSON object.
                """)
    return rerank_prompt