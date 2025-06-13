import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
import re
import torch
from sentence_transformers import CrossEncoder
from llm import *
import pickle
from rank_bm25 import BM25Okapi

K = 5
torch.classes.__path__ = []

embedding_model = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
vectorstore = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 15})
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

with open("title_to_doc.pkl", "rb") as f:
    title_to_doc = pickle.load(f)

# Initialize BM25 index for title search
titles = list(title_to_doc.keys())
tokenized_titles = [title.lower().split() for title in titles]
bm25 = BM25Okapi(tokenized_titles)

st.title("Article Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi, I am your article assistant chatbot."}]
if "last_retrieved_text" not in st.session_state:
    st.session_state.last_retrieved_text = ""
if "top_10_titles" not in st.session_state:
    st.session_state.top_10_titles = []
    
# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Describe your article...")


if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    is_followup = is_followup_query(user_input, st.session_state.messages)

    if is_followup and st.session_state.last_retrieved_text:
        reranked_text = st.session_state.last_retrieved_text
        st.subheader("Retrieved Articles")
        st.text_area("Top 5", reranked_text, height=300, key="retrieved_articles")
    else:
        with st.spinner("Retrieving and reranking relevant chunks..."):
            docs = retriever.get_relevant_documents(user_input)
            doc_contents = [split_document(doc.page_content) for doc in docs]
            pairs = [[user_input, content] for content in doc_contents]
            scores = reranker.predict(pairs)
            sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            reranked_docs = [docs[i] for i in sorted_indices]
            reranked_text = parse_top_K_articles(reranked_docs)
            st.session_state.last_retrieved_text = reranked_text
        st.subheader("Retrieved Articles")
        st.text_area("Top 5", reranked_text, height=300, key="retrieved_articles")

    response = ""
    with st.spinner("Generating response..."):
        contents = build_llm_contents(user_input, st.session_state.last_retrieved_text, st.session_state.messages)
        response = llm(contents)

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": response})

with st.sidebar:
    st.header("Title search")
    title_query = st.text_input("Enter title ...")
    if title_query:
        with st.spinner("Fetching similar titles..."):
            tokenized_query = title_query.lower().split()
            scores = bm25.get_scores(tokenized_query)
            top_10_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:2*K]
            st.session_state.top_10_titles = [titles[i] for i in top_10_indices]
    else:
        st.session_state.top_10_titles = []

    selected_title = st.selectbox("Select an article title", [""] + st.session_state.top_10_titles)
    if selected_title and selected_title != "":
        with st.spinner("Fetching selected article..."):
            selected_article_text = title_to_doc.get(selected_title, "")
            st.session_state.last_retrieved_text = selected_article_text
            st.text_area("Selected article", selected_article_text, height=300, key="retrieved_articles_title")