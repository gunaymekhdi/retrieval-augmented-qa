import os
import sys

# 1. Add the project root to Python's import search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 2. Imports
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.vectorstore import save_to_chroma

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data")

def load_and_chunk_documents():
    print("Loading documents...")
    loader = PyPDFDirectoryLoader(DATA_PATH)
    raw_documents = loader.load()
    
    if not raw_documents:
        print("No documents found in data/ folder!")
        return []

    print(f"Loaded {len(raw_documents)} page(s).")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    all_chunks = text_splitter.split_documents(raw_documents)
    chunks = [c for c in all_chunks if len(c.page_content.strip()) > 10]
    
    print(f"Generated {len(chunks)} valid text chunks.")
    return chunks

def run_pipeline():
    chunks = load_and_chunk_documents()
    if chunks:
        save_to_chroma(chunks)

if __name__ == "__main__":
    run_pipeline()