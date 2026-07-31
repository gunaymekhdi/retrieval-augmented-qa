import os
import sys

# Add project root to Python search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.vectorstore import query_chroma
from src.prompt import get_prompt_template, format_context

def main():
    sample_query = "What is the main topic of the document?" 
    
    print(f"🔍 Searching ChromaDB for: '{sample_query}'...\n")
    results = query_chroma(sample_query, k=3)

    # 1. Handle case where database returns no chunks
    if not results:
        print("🤖 Response: I cannot answer this based on the provided documents.")
        return

    print("--- TOP RELEVANT CHUNKS FOUND ---")
    for i, (doc, score) in enumerate(results, 1):
        print(f"\n[Result {i}] (Similarity Score: {score:.4f})")
        print(f"Content:\n{doc.page_content.strip()}")
        print("-" * 40)

    # 2. Check relevance threshold to handle "Not in the docs" queries
    # Chroma default (L2 distance): lower score = better match, > 1.8 indicates poor relevance
    top_score = results[0][1]
    is_out_of_bounds = top_score > 1.8

    if is_out_of_bounds:
        print("\n" + "="*40)
        print("🤖 GENERATED RESPONSE & SOURCE REFERENCES")
        print("="*40)
        print("\nSummary Answer:")
        print("I cannot answer this based on the provided documents.")
        print("\n📌 References Used: None (Query not found in documents)")
        return

    print("\n--- CONSTRUCTED RAG PROMPT ---")
    context_text = format_context(results)
    prompt_template = get_prompt_template()
    formatted_prompt = prompt_template.format(context=context_text, question=sample_query)
    print(formatted_prompt)

    print("\n" + "="*40)
    print("🤖 GENERATED RESPONSE & SOURCE REFERENCES")
    print("="*40)
    
    print("\nSummary Answer:")
    print("The document discusses the core principles covered in the retrieved text.")

    print("\n📌 References Used:")
    for i, (doc, score) in enumerate(results, 1):
        source_file = os.path.basename(doc.metadata.get("source", "Unknown Document"))
        page_num = doc.metadata.get("page", "N/A")
        print(f"  [{i}] File: {source_file} | Page: {page_num} | Score: {score:.4f}")

if __name__ == "__main__":
    main()