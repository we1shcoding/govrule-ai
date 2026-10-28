"""GovRule AI 채팅 UI.

표현 계층. RAG 로직을 직접 호출하지 않고 백엔드 API만 사용한다.
판정(ANSWER/CLARIFY/REFUSE)과 근거 조문을 함께 보여 준다.
근거 없이 생성된 답은 담당자가 검증할 수 없으므로, 출처 표시는
부가 기능이 아니라 이 도구의 전제다.
"""
import requests
import streamlit as st

API_URL = "http://localhost:8000"

VERDICT = {
    "ANSWER":  ("근거 있음", "근거 조문을 찾아 답했습니다. 아래 원문을 확인하세요."),
    "CLARIFY": ("조건 부족", "조건에 따라 적용 조문이 달라집니다. 되물었습니다."),
    "REFUSE":  ("확인 불가", "수집된 법령에서 근거를 찾지 못했습니다."),
}

st.set_page_config(page_title="GovRule AI", page_icon="⚖️", layout="centered")
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
        if h.get("rag_ready"):
            st.success("검색 색인 적재됨")
        else:
            st.warning("색인 적재 중")
    except requests.RequestException:
        st.error("백엔드 미응답")
        st.caption("uvicorn 실행 여부를 확인하세요")

    st.divider()
    st.subheader("수집 범위")
    st.caption(
        "국가계약법 · 같은 법 시행령 · 같은 법 시행규칙\n\n"
        "별표 · 서식 · 계약예규 · 법령해석례는 포함되지 않습니다. "
        "해당 내용은 '확인 불가'로 처리됩니다."
    )
    st.divider()
    st.caption("외부 LLM API를 사용하지 않습니다.")


def render_sources(sources: list[dict]) -> None:
    """근거 조문. 답변이 실제 인용한 것을 먼저, 나머지는 접어서 보여 준다."""
    if not sources:
        return
    used = [s for s in sources if s["used"]]
    rest = [s for s in sources if not s["used"]]

    if used:
        st.markdown("**인용된 근거**")
        for s in used:
            with st.expander(f"✓ {s['cite']}"):
                st.text(s["text"])
    if rest:
        with st.expander(f"검색된 다른 조문 {len(rest)}건"):
            for s in rest:
                st.markdown(f"**{s['rank']}. {s['cite']}**")
                st.text(s["text"])
                st.divider()


def render_answer(msg: dict) -> None:
    label, note = VERDICT.get(msg["verdict"], ("", ""))
    if msg["verdict"] == "ANSWER":
        st.success(label, icon="✅")
    elif msg["verdict"] == "CLARIFY":
        st.info(label, icon="❓")
    else:
        st.warning(label, icon="🚫")
    st.caption(note)

    st.markdown(msg["content"])

    if msg.get("ungrounded"):
        st.caption("※ 모델이 단정했으나 근거 조문을 인용하지 못해 답변을 보류했습니다.")
    render_sources(msg.get("sources", []))


if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        if m["role"] == "user":
            st.markdown(m["content"])
        else:
            render_answer(m)

if prompt := st.chat_input("계약·구매 관련 질문을 입력하세요"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("조문 검색 후 로컬 모델 추론 중..."):
            try:
                r = requests.post(
                    f"{API_URL}/chat", json={"question": prompt}, timeout=180,
                )
                r.raise_for_status()
                data = r.json()
                msg = {
                    "role": "assistant",
                    "content": data["answer"],
                    "verdict": data["verdict"],
                    "sources": data["sources"],
                    "ungrounded": data["ungrounded"],
                }
            except requests.RequestException as e:
                msg = {"role": "assistant", "content": f"요청 실패: {e}",
                       "verdict": "REFUSE", "sources": [], "ungrounded": False}
        render_answer(msg)

    st.session_state.messages.append(msg)