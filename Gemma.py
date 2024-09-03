import os
import time
import streamlit as st

from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS                      ##Vectorstore DB
from langchain_community.document_loaders import PyPDFDirectoryLoader

from langchain_google_genai import GoogleGenerativeAIEmbeddings         ##Vector Embeddings
from dotenv import load_dotenv

load_dotenv() ## To load the environment variables

groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")

st.title("Gemma Based Chatbot Doc Q&A")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma-7b-it")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Questions:{input}
    """
)

def vector_embedding():

    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader = PyPDFDirectoryLoader("./datafiles") ##Data Ingestion Phase
        
        st.session_state.docs = st.session_state.loader.load() ##Loading the documents
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)

        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

        #Data Ingestion and Vector DB creation finished


prompt1 = st.text_input("What do you want to ask from the documents?")

if st.button("Creating Vector Store"):
    vector_embedding()
    st.write("Vector Store DB is ready")



if prompt1:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    #timing the overall process
    start = time.process_time()

    response = retrieval_chain.invoke({'input':prompt1})
    st.write(response['answer'])

    #Adding option to check context
    with st.expander("Document Similarity Search Context"):
        #Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("-------------------------------------")

