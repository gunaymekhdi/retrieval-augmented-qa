import os
import sys

# Add project root to Python search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.vectorstore import query_chroma

def main():
    # Write a quick question related to the document you ingested
    sample_query = "What is the main topic of the document?" 
    
    print(f"🔍 Searching ChromaDB for: '{sample_query}'...\n")
    results = query_chroma(sample_query, k=3)

    if not results:
        print("No matching results found.")
        return

    print("--- TOP RELEVANT CHUNKS FOUND ---")
    for i, (doc, score) in enumerate(results, 1):
        print(f"\n[Result {i}] (Similarity Score: {score:.4f})")
        print(f"Content:\n{doc.page_content.strip()}")
        print("-" * 40)

if __name__ == "__main__":
    main()