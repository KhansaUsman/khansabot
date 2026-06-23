"""
app.py
Streamlit chat interface for KhansaBot — a personal RAG chatbot about Khansa Usman,
built using LangChain, FAISS, Groq, and HuggingFace embeddings.
"""

import streamlit as st
import streamlit as st
st.write(st.secrets.get("GROQ_API_KEY", "NOT FOUND")[:10])  # shows first 10 chars only
from rag_pipeline import (
    load_documents,
    split_documents,
    get_embeddings,
    build_vectorstore,
    get_retriever,
    build_rag_chain,
    ask_khansabot,
)

# ---- Page config ----
st.set_page_config(
    page_title="KhansaBot",
    page_icon="💬",
    layout="centered",
)

st.title("💬 KhansaBot")
st.caption("Ask me anything about Khansa's CV, FYP, or coursework!")


# ---- Build the RAG pipeline once, cache it across reruns ----
@st.cache_resource(show_spinner="Setting up KhansaBot for the first time...")
def setup_rag_chain():
    docs = load_documents()
    chunks = split_documents(docs)
    embeddings = get_embeddings()
    vectorstore = build_vectorstore(chunks, embeddings)
    retriever = get_retriever(vectorstore)
    rag_chain = build_rag_chain(retriever)
    return rag_chain


rag_chain = setup_rag_chain()

# ---- Conversation history (Streamlit session state) ----
if "messages" not in st.session_state:
    st.session_state.messages = []  # for display: list of {"role": ..., "content": ...}
if "history" not in st.session_state:
    st.session_state.history = []  # for the LLM: list of (question, answer) tuples

# ---- Display past messages ----
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---- Chat input ----
user_input = st.chat_input("Ask KhansaBot a question...")

if user_input:
    # Show user's message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Get KhansaBot's answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = ask_khansabot(rag_chain, user_input, st.session_state.history)
            st.markdown(answer)

    # Save to both display history and LLM history
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.history.append((user_input, answer))