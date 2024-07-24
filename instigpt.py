import weaviate
import streamlit as st
from langchain.vectorstores import Weaviate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.memory import ConversationBufferMemory

# Initialize Weaviate client and other components
WEAVIATE_URL = "https://itk9ojwtuat8lomvl78yw.c0.asia-southeast1.gcp.weaviate.cloud"
WEAVIATE_API_KEY = "V4GiwajVKdpNS5VV098jzT12Yal1yUngMIfH"
HUGGINGFACE_API_KEY = "your_huggingface_api_key"

client = weaviate.Client(
    url=WEAVIATE_URL, auth_client_secret=weaviate.AuthApiKey(WEAVIATE_API_KEY)
)

embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

loader = PyPDFLoader("./FINAL_SCRAPED_DATA.pdf", extract_images=True)
pages = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=30)
docs = text_splitter.split_documents(pages)

vector_db = Weaviate.from_documents(docs, embeddings, client=client, by_text=False)
retriever = vector_db.as_retriever()

template = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use ten sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

model = ChatGroq(
    temperature=0.5,
    model="llama3-70b-8192",
    api_key="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"
)

output_parser = StrOutputParser()

# Initialize conversation memory
conversation_memory = ConversationBufferMemory()

# Define RAG chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | output_parser
)
def get_answer(question):
  return rag_chain.invoke(question)


