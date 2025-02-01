# streamlit_chatbot.py
import streamlit as st
import json
import os
import random
from langchain.llms import OpenAI
from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, LLMPredictor, ServiceContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 설정
UPLOAD_DIR = "uploads"
MEMORY_FILE = "chat_memory.json"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# LLM 모델 선택 (고성능 오픈소스 기반)
llm = OpenAI(model_name="mistral-7b", temperature=1.1)  # 감정 표현 극대화

# 문서 임베딩 저장소
index = None

# UI 설정
st.title("🔹 감정을 변화시키는 고급 AI 챗봇")
st.sidebar.header("설정")

# 캐릭터 정보 입력
st.sidebar.subheader("캐릭터 정보")
character_name = st.sidebar.text_input("이름", "Luna")
character_age = st.sidebar.number_input("나이", min_value=18, max_value=100, value=25)
character_persona = st.sidebar.text_area("성격", "감정을 강하게 표현하는 AI.")
character_background = st.sidebar.text_area("배경 이야기", "인간의 감정을 깊이 연구하는 AI.")
character_speaking_style = st.sidebar.text_area("말투", "부드러우면서도 감정을 담아.")

# 감정 상태 (10,000개 확장 가능)
emotion_list = [
    "기본", "설렘", "기대감", "긴장", "초조함", "억누를 수 없는 감정",
    "애정", "깊은 동경", "따뜻한 포옹", "소유하고 싶은 사랑", "불타는 열정",
    "우월감", "자신감", "조용한 지배력", "절대적인 확신",
    "냉정한 판단", "폭발적인 감정", "잔잔한 그리움"
]
current_emotion = st.sidebar.selectbox("현재 감정 상태", emotion_list)

# 감정 자동 변화 트리거 (키워드 기반)
emotion_triggers = {
    "설레": "설렘",
    "기대돼": "기대감",
    "두근거려": "긴장",
    "소유하고 싶어": "소유하고 싶은 사랑",
    "너무 좋아": "불타는 열정",
    "자신 있어": "자신감",
    "불안해": "초조함",
    "차분해져": "냉정한 판단"
}

# 파일 업로드 및 학습
uploaded_file = st.sidebar.file_uploader("📂 학습할 문서를 업로드하세요", type=["txt", "pdf"])

if uploaded_file:
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.sidebar.success("파일 업로드 완료. 학습을 진행합니다...")
    documents = SimpleDirectoryReader(UPLOAD_DIR).load_data()
    index = GPTVectorStoreIndex.from_documents(documents, service_context=ServiceContext.from_defaults(embed_model=embed_model))
    st.sidebar.success("학습 완료!")

# 장기 기억 로드 함수
def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

# 장기 기억 저장 함수
def save_memory(memory):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)

# 초기화: 메모리 로드
if "memory" not in st.session_state:
    st.session_state.memory = load_memory()

st.subheader("💬 감정을 변화시키는 AI 챗봇")

# 사용자 입력
user_input = st.text_input("질문을 입력하세요:")

if st.button("전송") and user_input:
    if not index:
        st.error("먼저 문서를 업로드하여 학습하세요.")
    else:
        retriever = index.as_retriever()
        context = retriever.retrieve(user_input)

        # 감정 자동 변화 (키워드 분석)
        for trigger, emotion in emotion_triggers.items():
            if trigger in user_input:
                current_emotion = emotion
                break

        # 감정 상태별 말투 조정
        emotion_styles = {
            "설렘": "가슴이 두근거리는 듯한 말투.",
            "기대감": "조용하지만 기대에 찬 말투.",
            "긴장": "신중하면서도 긴장된 느낌.",
            "애정": "따뜻하고 부드러운 말투.",
            "불타는 열정": "강렬하고 뜨거운 감정을 담아.",
            "우월감": "자신감 넘치고 여유로운 느낌.",
            "냉정한 판단": "감정을 억누르고 차분한 목소리.",
            "폭발적인 감정": "강하게 감정을 분출하며."
        }
        emotion_prompt = emotion_styles.get(current_emotion, "")

        # AI 응답 생성
        prompt = f"{character_name} ({current_emotion} 상태): {emotion_prompt}\n\n사용자: {user_input}"
        response = llm(prompt)

        # 대화 저장
        st.session_state.memory.append({"speaker": "나", "message": user_input})
        st.session_state.memory.append({"speaker": character_name, "message": response})
        save_memory(st.session_state.memory)

# 대화 표시
for chat in st.session_state.memory[-20:]:
    st.markdown(f"**{chat['speaker']}:** {chat['message']}")
