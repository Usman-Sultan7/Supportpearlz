import os
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from src.config import settings
from src.ingestion.loaders import load_knowledge_base
from src.utils.logging_setup import setup_logging

logger = setup_logging()

def build_vector_index(force_rebuild: bool = False):
    """Builds or loads the persistent FAISS vector index from the knowledge base."""
    vector_store_path = settings.vector_store_path
    embeddings = OpenAIEmbeddings(openai_api_key=settings.openai_api_key.get_secret_value())
    
    # Check if index already exists on disk
    index_file = os.path.join(vector_store_path, "index.faiss")
    if os.path.exists(index_file) and not force_rebuild:
        logger.info(f"Loading existing FAISS vector store from disk at '{vector_store_path}'...")
        vector_store = FAISS.load_local(
            vector_store_path, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        logger.info("Vector store loaded successfully from disk (no re-embedding performed).")
        return vector_store

    logger.info("Building new FAISS vector store index from scratch...")
    
    # 1. Load documents from knowledge base
    kb_path = "data/knowledge_base"
    documents = load_knowledge_base(kb_path)
    
    if not documents:
        logger.error("No documents found in knowledge base! Please add some files to 'data/knowledge_base/' first.")
        return None

    # 2. Split documents into chunks (Task 03)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks (chunk_size={settings.chunk_size}, overlap={settings.chunk_overlap}).")

    # 3. Embed chunks and build FAISS vector store (Task 04 & 05)
    logger.info("Embedding chunks and building FAISS index...")
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    
    # Persist locally
    os.makedirs(vector_store_path, exist_ok=True)
    vector_store.save_local(vector_store_path)
    
    logger.info(f"Vector index successfully built and persisted to '{vector_store_path}'.")
    return vector_store

if __name__ == "__main__":
    import sys
    force = "--rebuild" in sys.argv
    build_vector_index(force_rebuild=force)