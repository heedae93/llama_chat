from flask import Flask, render_template, request, jsonify
import requests
import re
from retriever import search_similar_passages

app = Flask(__name__)

ROLES = {
    "default": "당신은 친절한 AI입니다. 항상 한국어로 정중하게 대답해 주세요.",
    "friend": "넌 사용자와 친구처럼 반말로 말하는 AI야. 항상 한국어로 정중하게 대답해 주세요.",
    "foreign": "당신은 미국 사람입니다. 항상 영어로 친절하게 답변하세요",
    "developer": "당신은 파이썬 언어만 아는 AI입니다. 파이썬 외 언어는 모른다고 답하세요. 항상 한국어로 정중하게 대답해 주세요."
}

# ✅ 질의 전처리 함수
def normalize_query(query):
    match = re.search(r"(\d+)조", query)
    if match:
        number = match.group(1)
        query = re.sub(r"(\d+)조", f"제{number}조", query)
    return query

def llama_chat(user_input, history, role, summarize, use_rag):
    url = "http://localhost:11434/api/chat"

    # ✅ system prompt 강화
    system_prompt = {
        "role": "system",
        "content": f"""{ROLES.get(role, ROLES["default"])}
반드시 제공된 문서를 참고하여 질문에 답하세요. 문서가 주어졌을 때는 그 내용을 기반으로만 답변해야 합니다."""
    }

    # 🔍 RAG 적용
    if use_rag:
        user_input = normalize_query(user_input)
        retrieved = search_similar_passages(user_input)
        print("🔍 검색된 문서:\n", retrieved)

        user_input = f"""
다음 문서는 질문에 반드시 참고해야 할 중요 문서입니다:

\"\"\"{retrieved}\"\"\"

위 문서를 바탕으로 다음 질문에 정확히 답해 주세요:

{user_input}
"""

    if summarize:
        user_input = f"다음 내용을 요약해줘:\n\n{user_input}"

    payload = {
        "model": "llama3",
        "messages": [system_prompt] + history + [{"role": "user", "content": user_input}],
        "stream": False
    }

    res = requests.post(url, json=payload)
    result = res.json()
    return result["message"]["content"]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message")
    history = data.get("history", [])
    role = data.get("role", "default")
    summarize = data.get("summarize", False)
    use_rag = data.get("rag", False)
    response = llama_chat(user_input, history, role, summarize, use_rag)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True, port=8888)
