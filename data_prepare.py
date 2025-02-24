import sqlite3
import numpy as np
import faiss
import torch
from datasets import load_dataset
from transformers import DPRContextEncoder, AutoTokenizer
from rank_bm25 import BM25Okapi

device = "cuda"
context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base").to(device)
tokenizer = AutoTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

# Load dataset
squad = load_dataset("squad_v2", split="train").select(range(10000))
docs = [d["context"] for d in squad]

# Kết nối SQLite
db_conn = sqlite3.connect("embeddings.db")
cursor = db_conn.cursor()

# Tạo bảng
cursor.execute("CREATE TABLE IF NOT EXISTS contexts (id INTEGER PRIMARY KEY, text TEXT)")
cursor.execute("CREATE TABLE IF NOT EXISTS embeddings (id INTEGER PRIMARY KEY, embedding BLOB)")

# Lưu dữ liệu vào database
for i, doc in enumerate(docs):
    cursor.execute("INSERT INTO contexts (id, text) VALUES (?, ?)", (i, doc))

db_conn.commit()

# 🔹 BM25 Indexing
tokenized_docs = [doc.split() for doc in docs]  # Tách từ
bm25 = BM25Okapi(tokenized_docs)  # Tạo BM25 index

# 🔹 FAISS Indexing
embedding_size = 768  
faiss_index = faiss.IndexFlatL2(embedding_size)

embeddings = []
for i, doc in enumerate(docs):
    inputs = tokenizer(doc, return_tensors="pt", max_length=512, truncation=True, padding="max_length").to(device)
    with torch.no_grad():
        emb = context_encoder(**inputs).pooler_output.cpu().numpy()
    
    embeddings.append(emb)
    faiss_index.add(emb)  # Thêm vào FAISS CPU

    # Lưu embedding vào SQLite
    cursor.execute("INSERT INTO embeddings (id, embedding) VALUES (?, ?)", (i, emb.tobytes()))

db_conn.commit()
db_conn.close()

# 🔹 Lưu FAISS index ra file
faiss.write_index(faiss_index, "faiss_index.bin")

# 🔹 Lưu BM25 index ra file
import pickle
with open("bm25_index.pkl", "wb") as f:
    pickle.dump(bm25, f)

print("Data preparation completed!")
