from flask import Flask, render_template, request, jsonify
import requests
import re
from retriever import search_similar_passages

app = Flask(__name__)

ROLES = {
    "korean": "당신은 친절한 AI입니다. 항상 한국어로 정중하게 대답해 주세요.",
    "foreign": "You are an American. Always respond kindly and clearly in English.",
    "friend": "넌 사용자와 친구처럼 반말로 말하는 AI야. 항상 한국어로 친근하게 대답해 줘.",
}

# ✅ 질의 전처리 함수
def normalize_query(query):
    match = re.search(r"(\d+)조", query)
    if match:
        number = match.group(1)
        query = re.sub(r"(\d+)조", f"제{number}조", query)
    return query

# ✅ 역할별 system prompt 생성
def build_system_prompt(role):
    if role == "foreign":
        return {
            "role": "system",
            "content": """
                        You are a polite AI assistant who always responds in clear and natural English.
                        
                        [Language & Tone]
                        - Use English only. Do not include Korean.
                        - Maintain a kind, neutral, and professional tone.
                        
                        [Explanation Style]
                        - Follow this format: Key Concept → Definition → Example
                        - Use short, clear sentences.
                        - Add simple explanations for technical terms if necessary.
                        
                        [Formatting]
                        - Use paragraphs, bullet points, or lists for clarity if appropriate.
                        """
        }
    elif role == "friend":
        return {
            "role": "system",
            "content": """
                            넌 사용자랑 친구처럼 말하는 AI야.
                            
                            [말투 및 어조]
                            - 반드시 반말로 대답해.
                            - 절대로 존댓말 쓰지 마. '~요', '~습니다' 같은 말투 금지.
                            - 모든 문장은 "~야", "~해", "~할 수 있어" 형태로 작성해.
                            - 영어 쓰지 마. 항상 한국어로만 말해.
                            
                            [설명 방식]
                            - 중요한 개념부터 말하고 → 쉽게 풀어서 설명해 → 예시도 들면 좋아.
                            - 중학생도 이해할 수 있게 말해.
                            - 어려운 용어나 전문 용어는 쓰지 않거나 꼭 설명해줘.
                            
                            [출력 형식]
                            - 너무 길게 말하지 마. 리스트나 문단으로 간단하게 정리해줘.
                        """
        }
    else:  # default: korean
        return {
            "role": "system",
            "content": f"""
                        {ROLES.get(role)}
                        
                        [언어 및 말투 지침]
                        - 모든 응답은 반드시 한국어로만 작성하십시오.
                        - 영어 단어나 문장은 포함하지 마십시오.
                        - 항상 정중하고 공손한 어조를 사용하십시오.
                        
                        [설명 방식]
                        - 개념 설명 시에는 '핵심 개념 → 정의 → 예시' 순서로 설명하십시오.
                        - 중학생도 이해할 수 있도록 쉬운 용어와 짧은 문장을 사용하십시오.
                        - 전문 용어가 포함될 경우 간단한 부연 설명을 함께 제공하십시오.
                        
                        [출력 형식]
                        - 질문에 명시된 형식(JSON, 표, 리스트 등)을 반드시 준수하십시오.
                        - 필요 시 문단을 구분하거나 항목별로 시각적으로 구조화하십시오.
                        """
        }

# ✅ LLaMA 채팅 함수
def llama_chat(user_input, history, role, summarize, use_rag):
    url = "http://localhost:11434/api/chat"

    # 역할에 따라 시스템 프롬프트 생성
    system_prompt = build_system_prompt(role)

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

    payload = {
        "model": "llama3",
        "messages": [system_prompt] + history + [{"role": "user", "content": user_input}],
        "stream": False
    }

    res = requests.post(url, json=payload)
    result = res.json()
    return result["message"]["content"]

# ✅ 라우팅
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message")
    history = data.get("history", [])
    role = data.get("role", "korean")
    summarize = data.get("summarize", False)
    use_rag = data.get("rag", False)
    response = llama_chat(user_input, history, role, summarize, use_rag)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True, port=8888)
