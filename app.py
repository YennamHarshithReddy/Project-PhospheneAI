import streamlit as st
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from collections import Counter


# Paths
index_dir = "faiss_index"

# 1. Load embeddings + FAISS index
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed(texts):
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

db = FAISS.load_local(index_dir, embeddings=embed, allow_dangerous_deserialization=True)

# 2. Streamlit UI
st.set_page_config(page_title="News RAG with Citations", layout="wide")
st.title("📰 News Q&A (with Citations)")

query = st.text_input("Ask a question about the news:", "")

if query:
    # Search top 3 docs
    results = db.similarity_search(query, k=4)

    # Find the most common source among retrieved docs
    sources = [doc.metadata.get("source", "Unknown") for doc in results]
    majority_source = Counter(sources).most_common(1)[0][0]

    # Keep only docs from the majority source
    filtered_results = [doc for doc in results if doc.metadata.get("source") == majority_source]

    # Combine answer from the filtered docs
    answer_parts = [doc.page_content for doc in filtered_results]
    final_answer = " ".join(answer_parts)

    # Show answer
    st.subheader("Answer:")
    st.write(final_answer)

    # ✅ Show citations only from filtered results
    st.subheader("Citations:")
    st.markdown(f"**1.** {majority_source}")