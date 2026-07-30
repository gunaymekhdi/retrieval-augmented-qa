import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# Build absolute path to chroma_db in the project root
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_db")

def get_embedding_function():
    # Gets model name from .env, falls back to lightweight open-source model
    model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings

def save_to_chroma(chunks):
    embedding_function = get_embedding_function()
    
    print(f"Generating embeddings and saving {len(chunks)} chunks to '{CHROMA_PATH}'...")
    
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory=CHROMA_PATH
    )
    
    print("Successfully generated embeddings and persisted vector store!")
    return db

def query_chroma(query_text: str, k: int = 3):
    """
    Performs a similarity search in ChromaDB using vector embeddings.
    """
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    
    # Perform similarity search with relevance scores
    results = db.similarity_search_with_score(query_text, k=k)
    return results