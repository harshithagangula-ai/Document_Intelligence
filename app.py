import streamlit as st

from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)

from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
st.title("📄 AI PDF RAG Chatbot")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

retriever = db.as_retriever(search_kwargs={"k":3})

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

question = st.text_input("Ask a Question")

if st.button("Submit"):

    if question:

        answer = qa.invoke(question)

        st.success(answer["result"])