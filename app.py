import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
import re
import torch
from sentence_transformers import CrossEncoder
from llm import llm

K = 5
torch.classes.__path__ = []

embedding_model = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
vectorstore = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 15})
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

st.title("Article Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi, I am your article assistant chatbot."}]
if "last_retrieved_text" not in st.session_state:
    st.session_state.last_retrieved_text = ""

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Describe your article...")

def split_document(content: str) -> str:
    lines = content.split("\n")
    quarter_length = len(lines) // 4
    return "\n".join(lines[:quarter_length])

def parse_top_K_articles(docs: list) -> str:
    combined_content = "\n\n".join(split_document(doc.page_content) for doc in docs[:K])
    return combined_content

def is_followup_query(user_input: str, history: list) -> bool:
    if not history or len(history) < 2:
        return False
    prompt = f"Given the conversation history and the latest user input, determine if the latest input is a follow-up question related to the previous context.\n\nHistory:\n"
    for msg in history[-3:]:
        prompt += f"{msg['role'].capitalize()}: {msg['content']}\n"
    prompt += f"\nLatest input: {user_input}\n\nIs the latest input a follow-up question? Respond with 'Yes' or 'No'."
    response = llm([{"role": "user", "parts": [{"text": prompt}]}])
    return response.strip().lower() == "yes"

def build_llm_contents(user_input: str, reranked_text: str, history: list) -> list:
    context = reranked_text
    history_parts = []
    for msg in history[-3:]:
        if msg["role"] == "user":
            history_parts.append({"role": "user", "parts": [{"text": msg["content"]}]})
        elif msg["role"] == "assistant":
            history_parts.append({"role": "assistant", "parts": [{"text": msg["content"]}]})
    history_parts.append({"role": "user", "parts": [{"text": f"Relevant context:\n{context}"}]})
    history_parts.append({"role": "user", "parts": [{"text": user_input}]})
    return history_parts

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    is_followup = is_followup_query(user_input, st.session_state.messages)

    if is_followup and st.session_state.last_retrieved_text:
        reranked_text = st.session_state.last_retrieved_text
        st.subheader("Retrieved Articles")
        st.text_area("Top 5", reranked_text, height=300)
    else:
        with st.spinner("Retrieving and reranking relevant chunks..."):
            docs = retriever.get_relevant_documents(user_input)
            doc_contents = [split_document(doc.page_content) for doc in docs]
            pairs = [[user_input, content] for content in doc_contents]
            scores = reranker.predict(pairs)
            sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            reranked_docs = [docs[i] for i in sorted_indices]
            reranked_text = parse_top_K_articles(reranked_docs)
            st.session_state.last_retrieved_text = reranked_text  # Store retrieved text

        st.subheader("Retrieved Articles")
        st.text_area("Top 5", reranked_text, height=300)

    response = ""
    with st.spinner("Generating response..."):
        contents = build_llm_contents(user_input, reranked_text, st.session_state.messages)
        response = llm(contents)

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": response})