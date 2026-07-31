from langchain_core.prompts import ChatPromptTemplate

# Clear template separating system instructions, retrieved context, and user question
PROMPT_TEMPLATE = """
Answer the question based ONLY on the following context. If the answer is not contained within the context, respond with "I cannot answer this based on the provided documents."

--- CONTEXT START ---
{context}
--- CONTEXT END ---

Question: {question}
"""

def get_prompt_template():
    """
    Returns a LangChain ChatPromptTemplate instance.
    """
    return ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

def format_context(docs):
    """
    Combines retrieved document text chunks into a single formatted string,
    including source file name and page metadata for citation.
    """
    formatted_chunks = []
    for doc, _score in docs:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        
        # Header showing where this chunk came from
        header = f"[Source: {source} | Page: {page}]"
        formatted_chunks.append(f"{header}\n{doc.page_content.strip()}")
        
    return "\n\n---\n\n".join(formatted_chunks)