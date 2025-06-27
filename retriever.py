import faiss
import numpy as np
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

VECTOR_PATH = "data/faiss.index"
CHUNKS_PATH = "data/chunks.pkl"
VECTORIZER_PATH = "data/vectorizer.pkl"

# 📘 문서 로딩 및 전처리
with open("data/수클운영규칙.txt", encoding="utf-8") as f:
    raw_text = f.read()

chunks = raw_text.strip().split("\n\n")  # 빈 줄 기준 분할

# 📌 벡터화 및 저장 (매번 새로 생성)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(chunks).toarray().astype(np.float32)

print("📦 벡터 DB를 새로 생성합니다...")
index = faiss.IndexFlatL2(X.shape[1])
index.add(X)
faiss.write_index(index, VECTOR_PATH)

with open(CHUNKS_PATH, "wb") as f:
    pickle.dump(chunks, f)
with open(VECTORIZER_PATH, "wb") as f:
    pickle.dump(vectorizer, f)
print("✅ 벡터 DB 저장 완료")

# 🔍 유사 문단 검색 함수
def search_similar_passages(query, top_k=10):
    index = faiss.read_index(VECTOR_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    query_vec = vectorizer.transform([query]).toarray().astype(np.float32)
    D, I = index.search(query_vec, top_k)
    return "\n".join([chunks[i] for i in I[0]])
