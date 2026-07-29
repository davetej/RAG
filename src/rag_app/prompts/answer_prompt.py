from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)


def build_answer_prompt() -> ChatPromptTemplate:
    """Build the final answer prompt for the CSDC chain."""
    return ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                "You answer only using the provided context. "
                "If the answer cannot be found in the context, respond with 'I don't know.'.\n\n"
                "Context:\n{context}"
            ),
            HumanMessagePromptTemplate.from_template("{input}"),
        ]
    )
