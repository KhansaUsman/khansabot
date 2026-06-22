"""
rag_pipeline.py
Core RAG (Retrieval-Augmented Generation) logic for KhansaBot.
Loads personal documents, builds a searchable vector index, and answers
questions using ONLY the retrieved context (grounded, no hallucination).
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env (GROQ_API_KEY)
load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
assert GROQ_API_KEY, "GROQ_API_KEY not found. Add it to your .env file."

DATA_DIR = Path("data")
VECTORSTORE_DIR = Path("vectorstore")

# ---- Step 0: Set up the LLM (Groq) ----
from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,  # deterministic, no creative guessing — keeps answers grounded
)

# ---- Step 1: Load documents from data/ ----
from langchain_community.document_loaders import TextLoader, PyPDFLoader

def load_documents():
    documents = []
    for path in sorted(DATA_DIR.glob("*")):
        if path.suffix.lower() in (".txt", ".md"):
            documents.extend(TextLoader(str(path), encoding="utf-8").load())
        elif path.suffix.lower() == ".pdf":
            documents.extend(PyPDFLoader(str(path)).load())
    return documents

# ---- Step 2: Split documents into chunks ----
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,      # increased from 500 — keeps short facts (like CGPA) with more surrounding context
        chunk_overlap=150,   # increased proportionally with chunk_size
    )
    return splitter.split_documents(documents)

def debug_find_chunk(chunks, keyword):
    """Utility: find which chunk(s) contain a specific keyword. Useful for debugging retrieval issues."""
    print(f"\nSearching all {len(chunks)} chunks for '{keyword}':")
    found_any = False
    for i, c in enumerate(chunks):
        if keyword.lower() in c.page_content.lower():
            found_any = True
            source = Path(c.metadata["source"]).name
            print(f"\n--- Chunk #{i} (source: {source}) ---")
            print(c.page_content)
    if not found_any:
        print("NOT FOUND in any chunk.")

# ---- Step 3: Create embeddings + FAISS vector store ----
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def build_vectorstore(chunks, embeddings):
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(VECTORSTORE_DIR))
    return vectorstore

# ---- Step 4: Retriever ----
def get_retriever(vectorstore, k=6):  # increased from 4 — casts a wider net so the right chunk is more likely included
    return vectorstore.as_retriever(search_kwargs={"k": k})
# ---- Step 5: Grounded prompt — KhansaBot's identity + anti-hallucination guard ----
from langchain_core.prompts import ChatPromptTemplate

answer_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are KhansaBot, a personal assistant that answers questions about Khansa Usman "
     "using ONLY the context below.\n"
     "- Use only facts found in the context; never use outside knowledge or guess.\n"
     "- You MAY combine facts from several context chunks to give a complete answer.\n"
     "- If the answer is genuinely not in the context, say: "
     "\"I don't have that information about Khansa.\"\n"
     "- Be concise, friendly, and quote specific details (dates, CGPA, project names) exactly as written.\n\n"
     "Context:\n{context}"),
    ("human", "{input}"),
])
# ---- Step 6: Build the RAG chain (with conversation history support) ----
try:
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
except ModuleNotFoundError:
    from langchain_classic.chains import create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain

def build_rag_chain(retriever):
    qa_chain = create_stuff_documents_chain(llm, answer_prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)
    return rag_chain


def format_history(history):
    """Turn a list of (question, answer) tuples into a text block for context."""
    if not history:
        return ""
    lines = []
    for q, a in history:
        lines.append(f"User: {q}")
        lines.append(f"KhansaBot: {a}")
    return "\n".join(lines)


def ask_khansabot(rag_chain, question, history=None):
    """
    Ask a question, optionally including prior conversation history.
    history: list of (question, answer) tuples from earlier turns.
    Returns the answer string.
    """
    history = history or []
    history_text = format_history(history)

    # Prepend history to the question so the LLM has conversational context
    if history_text:
        full_input = f"Previous conversation:\n{history_text}\n\nNew question: {question}"
    else:
        full_input = question

    result = rag_chain.invoke({"input": full_input})
    return result["answer"]


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} document(s) from {DATA_DIR}/")

    chunks = split_documents(docs)
    print(f"Split into {len(chunks)} chunks")

    embeddings = get_embeddings()
    vectorstore = build_vectorstore(chunks, embeddings)
    print(f"Indexed {vectorstore.index.ntotal} chunks and saved to {VECTORSTORE_DIR}/")

    retriever = get_retriever(vectorstore)
    rag_chain = build_rag_chain(retriever)

    print("\n" + "=" * 50)
    print("Testing KhansaBot with a multi-turn conversation:")
    print("=" * 50)

    history = []

    q1 = "What is Khansa's CGPA?"
    a1 = ask_khansabot(rag_chain, q1, history)
    print(f"\nQ1: {q1}\nA1: {a1}")
    history.append((q1, a1))

    q2 = "What is her FYP project about?"
    a2 = ask_khansabot(rag_chain, q2, history)
    print(f"\nQ2: {q2}\nA2: {a2}")
    history.append((q2, a2))

    q3 = "What is her favorite food?"  # not in any document — should trigger refusal
    a3 = ask_khansabot(rag_chain, q3, history)
    print(f"\nQ3: {q3}\nA3: {a3}")