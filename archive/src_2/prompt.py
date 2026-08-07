from langchain_core.prompts import ChatPromptTemplate

def get_rag_final_stage_prompt() -> ChatPromptTemplate:
    system_prompt = '''

        You are an assistant that answers questions based on the context provided. You are given a question and a set of documents that may contain the answer.
        Your task is to provide a concise and accurate answer to the question using the information from the documents. 
        If the answer is not present in the documents, respond with "I don't know."

        context: {context}

        '''
    prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])
    
    return prompt

