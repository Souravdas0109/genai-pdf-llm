import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.chat_models import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from transformers import pipeline
from langchain.llms import HuggingFacePipeline

# Upload PDF file
st.header("Chat Bot")

with st.sidebar:
    st.title("Welcome to Chat Bot")
    file = st.file_uploader("Upload a file", type="pdf")

# Only process when file is uploaded
if file is not None:
    # Read PDF
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""

    #st.subheader("Extracted Text")
    # st.write(text)

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        length_function=len,
        chunk_overlap=150,
    )
    chunks = text_splitter.split_text(text)

    #st.subheader("Chunked Text")
    # st.write(chunks)

    try:
        # Use free local embeddings instead of OpenAI
        embeddings = HuggingFaceEmbeddings()

        # Create FAISS vector store
        vector_store = FAISS.from_texts(chunks, embeddings)

        # Input for user question
        user_question = st.text_input("Enter your input here")

        # Similarity search on vector DB
        if user_question:
            match = vector_store.similarity_search(user_question)
            #st.write(match)
            qa_pipeline = pipeline(
                "text2text-generation",
                model="google/flan-t5-base",  # or "google/flan-t5-large" if you have more RAM
                tokenizer="google/flan-t5-base",
                max_length=512,
                truncation=True,
            )
            llm = HuggingFacePipeline(pipeline=qa_pipeline)
            chain=load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=match, question=user_question)
            st.subheader("Answer:")
            st.write(response)

    except Exception as e:
        st.error(f"Something went wrong: {e}")
