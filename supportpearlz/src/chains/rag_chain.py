import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from src.config import settings
from src.retrieval.retriever import get_retriever
from src.utils.logging_setup import setup_logging

logger = setup_logging()

def format_docs(docs):
    """Format retrieved documents into a single string context."""
    return "\n\n".join(doc.page_content for doc in docs)

def get_rag_chain():
    """Constructs and returns the production RAG customer support chain."""
    logger.info("Initializing RAG chain components...")
    
    # 1. Initialize LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
        openai_api_key=settings.openai_api_key.get_secret_value()
    )
    
    # 2. Get Retriever
    retriever = get_retriever(k=3)
    if not retriever:
        logger.error("Failed to retrieve vector store for RAG chain.")
        return None

    # 3. Define Customer Support Prompt Template
    template = """You are SupportPearlz, an expert and polite AI customer support assistant for Pearlz Home Systems. 
Answer the user's question accurately using ONLY the provided context below. If you do not know the answer or if it's not present in the context, politely state that you cannot help with that and direct them to human support.

Context:
{context}

User Question: {question}

Helpful Answer:"""

    prompt = ChatPromptTemplate.from_template(template)

    # 4. Build LCEL Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    logger.info("RAG chain successfully constructed.")
    return rag_chain

if __name__ == "__main__":
    chain = get_rag_chain()
    if chain:
        test_query = "What does the warranty cover for the AquaPearl 500 Pro?"
        logger.info(f"Running test query: '{test_query}'")
        response = chain.invoke(test_query)
        print("\n--- RAG Response ---")
        print(response)
        print("--------------------\n")