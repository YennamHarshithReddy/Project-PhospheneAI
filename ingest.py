import os
import zipfile
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# Paths
zip_path = "data/news/news.zip"
extract_dir = "data/news"
index_dir = "faiss_index"

# 1. Extract news.zip if present
if os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"Extraction complete! Files are in {extract_dir}")
else:
    print(f"No zip file found at {zip_path}, using existing files if any...")

# 2. Parse .txt files
docs = []
for fname in os.listdir(extract_dir):
    if fname.endswith(".txt"):
        fpath = os.path.join(extract_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            text = f.read()

        # Chunking
        chunks = [text[i:i+500] for i in range(0, len(text), 500)]
        for chunk in chunks:
            docs.append(Document(page_content=chunk, metadata={"source": fname}))

print(f"Parsed {len(docs)} chunks successfully!")

# 3. Use HuggingFaceEmbeddings (LangChain wrapper)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. Build FAISS index with metadata
texts = [d.page_content for d in docs]
metadatas = [d.metadata for d in docs]

db = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)

# 5. Save index
db.save_local(index_dir)
print(f"FAISS index built and saved to {index_dir}")
