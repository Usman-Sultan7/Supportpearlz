Link of Streamlit : https://support-pearlz.streamlit.app/
# SupportPearlz RAG Customer Assistant

An advanced, production-grade Retrieval-Augmented Generation (RAG) customer support assistant built for Pearlz Home Systems using LangChain, OpenAI (`gpt-4o-mini`), FAISS vector store, and an interactive Streamlit web UI.

---

## 🏗️ Project Architecture & Structure
- `data/knowledge_base/`: Contains 10+ comprehensive markdown documents covering product manuals, warranties, shipping policies, and FAQs.
- `data/vector_store/`: Local persistent FAISS vector index storage.
- `src/config.py`: Centralized configuration management using Pydantic Settings.
- `src/ingestion/`: Automated document loaders and vector store builder scripts.
- `src/retrieval/`: Configurable FAISS vector retrieval module.
- `src/chains/`: LCEL RAG generation chains and prompt templates.
- `app.py`: Upgraded, secure Streamlit web application featuring API key authentication and quick-prompt suggestions.

---

## 🚀 Setup & Installation Instructions (Clean Environment)

### 1. Clone the Repository
```cmd
git clone [https://github.com/Usman-Sultan7/Supportpearlz.git](https://github.com/Usman-Sultan7/Supportpearlz.git)
cd Supportpearlz
