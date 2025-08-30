import streamlit as st
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
import os

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
    # Search top 1 doc
    results = db.similarity_search(query, k=1)

    # Combine answer from docs
    answer_parts = [doc.page_content for doc in results]
    final_answer = " ".join(answer_parts)

    st.subheader("Answer:")
    st.write(final_answer)

    # Show citations
    st.subheader("Citations:")
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "Unknown")
        st.markdown(f"**{i}.** {source}")
