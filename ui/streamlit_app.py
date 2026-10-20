"""GovRule AI 채팅 UI.

표현 계층. RAG 로직을 직접 호출하지 않고 백엔드 API만 사용한다.
"""
import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="GovRule AI", page_icon="⚖️")
st.title("⚖️ GovRule AI")
st.caption("공공기관 계약·구매 법령 검토 — 온프레미스 로컬 추론")

with st.sidebar:
    st.subheader("시스템 상태")
    try:
        h = requests.get(f"{API_URL}/health", timeout=5).json()
        if h.get("ollama"):
            st.success("Ollama 연결됨")
        else:
            st.error("Ollama 응답 없음")
    except requests.RequestException:
        st.error("백엔드 미응답")
        st.caption("uvicorn 실행 여부를 확인하세요")

    st.divider()
    st.caption("외부 LLM API를 사용하지 않습니다.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("계약·구매 관련 질문을 입력하세요"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("로컬 모델 추론 중..."):
            try:
                r = requests.post(
                    f"{API_URL}/chat",
                    json={"question": prompt},
                    timeout=120,
                )
                r.raise_for_status()
                answer = r.json()["answer"]
            except requests.RequestException as e:
                answer = f"요청 실패: {e}"
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
