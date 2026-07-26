import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_PATH = "data/"

def load_and_chunk_documents():
    print("Loading documents...")
    loader = PyPDFDirectoryLoader(DATA_PATH)
    raw_documents = loader.load()
    
    if not raw_documents:
        print("No documents found in the data/ directory! Add a PDF first.")
        return

    print(f"Loaded {len(raw_documents)} page(s) from documents.\n")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        #is_separator_regex=False,
        separators=["\n\n", "\n", " ", ""],
    )

    #chunks = text_splitter.split_documents(raw_documents)
    chunks = [c for c in text_splitter.split_documents(raw_documents) if len(c.page_content.strip()) > 10]
    
    print(f"Total chunks generated: {len(chunks)}")
    print("--------------------------------------------------")
    
    if chunks:
        print("Preview of First Chunk:")
        print(f"Source: {chunks[0].metadata}")
        print(f"Content Length: {len(chunks[0].page_content)} characters")
        print("\nContent:")
        print(chunks[0].page_content)
        print("--------------------------------------------------")

if __name__ == "__main__":
    load_and_chunk_documents()