import streamlit as st
import sqlite3
import numpy as np
import faiss
import torch
import pickle
from transformers import DPRQuestionEncoder, AutoTokenizer, pipeline
from rank_bm25 import BM25Okapi

# # ✅ Fix lỗi Streamlit Async
# asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# 🔹 Load models
device = "cuda"
torch.set_num_threads(1) 
question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base").to(device)
tokenizer = AutoTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
qa_pipeline = pipeline("question-answering", model="KienLe21/demo_qa_model", device=0)

# 🔹 Load FAISS index
faiss_index = faiss.read_index("faiss_index.bin")

# 🔹 Load BM25 index
with open("bm25_index.pkl", "rb") as f:
    bm25 = pickle.load(f)

# 🔹 Load contexts và embeddings từ SQLite (✅ Fix lỗi "ambiguous column name")
db_conn = sqlite3.connect("embeddings.db")
cursor = db_conn.cursor()
cursor.execute("""
    SELECT contexts.id, contexts.text, embeddings.embedding 
    FROM contexts 
    JOIN embeddings ON contexts.id = embeddings.id
""")
data = cursor.fetchall()
db_conn.close()

# 🔹 Tạo danh sách contexts và embeddings
docs = [row[1] for row in data]
all_embeddings = [np.frombuffer(row[2], dtype=np.float32) for row in data]

def query_celebrity(question):
    # 🔹 BM25 lấy top-25 context gần nhất
    top_n_bm25 = 25
    tokenized_question = question.split()
    bm25_scores = bm25.get_scores(tokenized_question)
    top_k = np.argsort(bm25_scores)[::-1][:top_n_bm25]  # Lấy 25 context tốt nhất
    selected_docs = [docs[i] for i in top_k]
    selected_embeddings = [all_embeddings[i] for i in top_k]

    if not selected_embeddings:
        return {"answer": "Không tìm thấy câu trả lời", "score": 0, "context": ""}

    # 🔹 Encode câu hỏi
    inputs = tokenizer(
        question,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding="max_length"
    ).to(device)

    with torch.no_grad():
        question_emb = question_encoder(**inputs).pooler_output.cpu().numpy()

    # 🔹 FAISS search trên 25 context tốt nhất
    temp_faiss_index = faiss.IndexFlatL2(len(selected_embeddings[0]))
    temp_faiss_index.add(np.array(selected_embeddings))
    _, faiss_results = temp_faiss_index.search(question_emb, k=1)

    if len(faiss_results[0]) == 0:
        return {"answer": "Không tìm thấy câu trả lời", "score": 0, "context": ""}

    best_context = selected_docs[faiss_results[0][0]]

    # 🔹 Dùng QA model để tìm câu trả lời
    answer = qa_pipeline(question=question, context=best_context)

    return {"answer": answer["answer"], "score": answer["score"], "context": best_context}


# 🔹 Streamlit UI
st.title("Celebrity QA System")
question = st.text_input("Ask a question about a celebrity:")
if st.button("Get Answer"):
    if question:
        response = query_celebrity(question)
        st.write("### Answer:")
        st.write(response["answer"])
        st.write("### Context:")
        st.write(response["context"])
