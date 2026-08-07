from RAG.archive.src_2.config import AppSettings
from RAG.archive.src_2.llm import get_chat_model
from RAG.archive.src_2.retriver import init_basic_retriever
from vectorstore import load_vectorstore

# llm = get_chat_model(AppSettings.from_env())

# input_text = "What is Krishna saying about the nature of the soul in the Bhagavad Gita?"

# response = llm.invoke(input=input_text)
# print(f"AI: {response.content}")

# data ingestion flow test
from RAG.archive.src_2.pdfProcessor import processPdf

processPdf(AppSettings.from_env(),pdf_name= 'BhagavadGita.pdf', semantic_chunker=True)


# retriever = init_retriever(AppSettings.from_env())
# response = retriever.invoke(input='what is Krishna saying about the nature of the soul in the Bhagavad Gita?')
# print(response)

from RAG.archive.src_2.create_retrival_chain import build_retrieval_pipeline

# retrieval_chain = build_retrieval_pipeline(AppSettings.from_env())
# response = retrieval_chain.invoke({"input": 'what is Krishna saying about the nature of the soul in the Bhagavad Gita?'})
# print(response['answer'])


settings = AppSettings()

vectorstore = load_vectorstore(settings)

retriver = init_basic_retriever(vectorstore, settings)
