import streamlit as st
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from src.utils.logging_setup import setup_logging

logger = setup_logging("retriever")

def get_retriever():
    """Initializes and returns the FAISS retriever."""
    logger.info("Initializing FAISS retriever...")
    
    vector_store_path = Path("data/vector_store")
    
    if not vector_store_path.exists():
        logger.error(f"Vector store path {vector_store_path} does not exist.")
        return None

    # Retrieve the API key dynamically from the Streamlit session state
    api_key = st.session_state.get("api_key", "")
    if not api_key:
        logger.error("OpenAI API key missing from session state.")
        return None

    try:
        # Initialize embeddings with the dynamic user key
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small", 
            openai_api_key=api_key
        )
        
        # Load the FAISS vector store 
        # (allow_dangerous_deserialization is required for local FAISS loading in LangChain)
        vector_store = FAISS.load_local(
            folder_path=str(vector_store_path), 
            embeddings=embeddings, 
            allow_dangerous_deserialization=True
        )
        
        retriever = vector_store.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 4}
        )
        
        logger.info("Retriever successfully initialized.")
        return retriever
        
    except Exception as e:
        logger.error(f"Error loading FAISS vector store: {e}")
        return None