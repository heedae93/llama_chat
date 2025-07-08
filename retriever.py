import faiss
import numpy as np
import os
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer

VECTOR_PATH = "data/faiss.index"
CHUNKS_PATH = "data/chunks.pkl"
VECTORIZER_PATH = "data/vectorizer.pkl"

# 📘 문서 로딩 및 전처리
with open("data/수클운영규칙.txt", encoding="utf-8") as f:
    raw_text = f.read()

chunks = raw_text.strip().split("\n\n")  # 빈 줄 기준 분할 ( 문단 단위로 분리 )

# 📌 벡터화 및 저장 (매번 새로 생성)
vectorizer = TfidfVectorizer() # TfidfVectorizer는 텍스트를 숫자(벡터)로 바꾸는 도구
X = vectorizer.fit_transform(chunks).toarray().astype(np.float32)

print("📦 벡터 DB를 새로 생성합니다...")
index = faiss.IndexFlatL2(X.shape[1])
index.add(X)
faiss.write_index(index, VECTOR_PATH) # 벡터 DB에 저장

with open(CHUNKS_PATH, "wb") as f:
    pickle.dump(chunks, f)
with open(VECTORIZER_PATH, "wb") as f:
    pickle.dump(vectorizer, f)
print("✅ 벡터 DB 저장 완료")



# ✅ 질의에서 모든 조항 번호 추출
def extract_article_numbers(text):
    return re.findall(r'(\d+)조', text)  # 예: ['1', '2', '9']

# # 🔍 유사 문단 검색 함수 (다중 조항 대응)
# 챗봇에서 질문이 들어오면 그 질문을 벡터로 바꾸고 미리 저장해 둔 벡터 DB에서 가장 유사한 문단을 검색해서 LLM에 넣어서 답변 생성
def search_similar_passages(query, top_k=10):

    index = faiss.read_index(VECTOR_PATH) # 벡터 DB 로드

    with open(CHUNKS_PATH, "rb") as f: # 텍스트 원본 로드
        chunks = pickle.load(f)

    with open(VECTORIZER_PATH, "rb") as f: # 텍스트를 벡터로 변환하는 변환기 로드 ( 사용자 입력을 벡터화 시키는 역할 )
        vectorizer = pickle.load(f)

    results = []

    # 📌 조항 번호 기반 직접 검색
    article_numbers = extract_article_numbers(query)
    for number in article_numbers:
        for chunk in chunks:
            if f"{number}조" in chunk or f"제{number}조" in chunk:
                results.append(chunk)
                break  # 조항별 1개만 추출

    # 🔁 fallback: 조항 번호가 없거나 검색이 부족한 경우 top-k 유사 문단 추가
    if not results:
        query_vec = vectorizer.transform([query]).toarray().astype(np.float32)
        D, I = index.search(query_vec, top_k)
        results = [chunks[i] for i in I[0]]

    return "\n".join(results)
