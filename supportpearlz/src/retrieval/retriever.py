import logging
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from src.config import settings
from src.utils.logging_setup import setup_logging

logger = setup_logging()

def get_retriever(k: int = 3):
    """Loads the persistent FAISS vector store and returns a configured retriever."""
    vector_store_path = settings.vector_store_path
    embeddings = OpenAIEmbeddings(openai_api_key=settings.openai_api_key.get_secret_value())
    
    try:
        # Load persisted FAISS index from disk
        vector_store = FAISS.load_local(
            vector_store_path, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        
        # Configure retriever with top-k
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
        logger.info(f"Retriever successfully initialized with k={k}.")
        return retriever
        
    except Exception as e:
        logger.error(f"Failed to load vector store retriever from {vector_store_path}: {e}")
        return None