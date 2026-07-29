from config import AppSettings
from llm import get_chat_model
from retriver import init_retriever

# llm = get_chat_model(AppSettings.from_env())

# input_text = "What is Krishna saying about the nature of the soul in the Bhagavad Gita?"

# response = llm.invoke(input=input_text)
# print(f"AI: {response.content}")

# data ingestion flow test
from pdfProcessor import processPdf

# processPdf(AppSettings.from_env(), 'BhagavadGita.pdf')


# retriever = init_retriever(AppSettings.from_env())
# response = retriever.invoke(input='what is Krishna saying about the nature of the soul in the Bhagavad Gita?')
# print(response)

from create_retrival_chain import build_retrieval_pipeline

retrieval_chain = build_retrieval_pipeline(AppSettings.from_env())
response = retrieval_chain.invoke({"input": 'what is Krishna saying about the nature of the soul in the Bhagavad Gita?'})
print(response['answer'])


