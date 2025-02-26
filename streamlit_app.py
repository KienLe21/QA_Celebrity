import streamlit as st
import sqlite3
import numpy as np
import faiss
import torch
import pickle
from transformers import DPRQuestionEncoder, AutoTokenizer, pipeline
from rank_bm25 import BM25Okapi

# 🔹 Load models
device = "cuda"
torch.set_num_threads(1) 
# question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base").to(device)
# tokenizer = AutoTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
question_encoder = DPRQuestionEncoder.from_pretrained("KienLe21/dpr_squadv2_finetune_question").to(device)
tokenizer = AutoTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
# qa_pipeline = pipeline("question-answering", model="KienLe21/demo_qa_model", device=0)
qa_pipeline = pipeline("question-answering", model="KienLe21/finetune_distilbert", device=0)

# 🔹 Load FAISS index
faiss_index = faiss.read_index("faiss_index.bin")

# 🔹 Load BM25 index
with open("bm25_index.pkl", "rb") as f:
    bm25 = pickle.load(f)

# 🔹 Load contexts và embeddings từ SQLite 
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
    # 🔹 BM25 lấy top context gần nhất
    top_n_bm25 = 100
    tokenized_question = question.split()
    bm25_scores = bm25.get_scores(tokenized_question)
    top_k = np.argsort(bm25_scores)[::-1][:top_n_bm25]
    selected_docs = [docs[i] for i in top_k]
    selected_embeddings = [all_embeddings[i] for i in top_k]
    bm25_selected_scores = [bm25_scores[i] for i in top_k]

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

    # 🔹 FAISS search trên các context đã chọn từ BM25
    selected_embeddings = np.array(selected_embeddings)  # Chuyển thành NumPy array
    temp_faiss_index = faiss.IndexFlatL2(selected_embeddings.shape[1])
    temp_faiss_index.add(selected_embeddings)
    distances, faiss_results = temp_faiss_index.search(question_emb, k=top_n_bm25)

    # 🔹 Tính điểm ranking kết hợp BM25 + FAISS
    combined_scores = []
    for i, doc_idx in enumerate(top_k):
        faiss_score = 1 / (1 + distances[0][i])  # Chuyển khoảng cách L2 thành điểm
        bm25_score = bm25_selected_scores[i]  # Lấy điểm BM25 gốc
        final_score = bm25_score + faiss_score  # Kết hợp điểm BM25 + FAISS
        combined_scores.append((final_score, selected_docs[i]))

    # 🔹 Chọn top-K context tốt nhất
    top_k_final = 5
    sorted_docs = sorted(combined_scores, key=lambda x: x[0], reverse=True)[:top_k_final]

    # 🔹 Dùng QA model trên các context đã chọn
    best_answer = {"answer": "Không tìm thấy câu trả lời", "score": 0, "context": ""}
    score_threshold = 0.2  # Ngưỡng confidence score

    for score, context in sorted_docs:
        answer = qa_pipeline(question=question, context=context)
        if answer["score"] > best_answer["score"]:  # Chọn câu trả lời có độ tin cậy cao nhất
            best_answer = {
                "answer": answer["answer"],
                "score": answer["score"],
                "context": context
            }

    # 🔹 Nếu score quá thấp, trả về "Không tìm thấy câu trả lời"
    if best_answer["score"] < score_threshold:
        return {"answer": "Không tìm thấy câu trả lời", "score": 0, "context": ""}

    return best_answer



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
