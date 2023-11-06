import os
import sys
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma


# API KEY
load_dotenv(".env")
api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key

loader = DirectoryLoader("docs/")
index = VectorstoreIndexCreator().from_loaders([loader])

chain = ConversationalRetrievalChain.from_llm(
  llm=ChatOpenAI(model="gpt-3.5-turbo"),
  retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
)

chat_history = []

question = None

while True:
  question = input("Prompt (stop to quit): ")

  if "stop" in question:
    print("Thanks for using")
    break

  else:
    result = chain({"question": question, "chat_history": chat_history})
    print(result['answer'])
    chat_history.append((question, result['answer']))
    question = None