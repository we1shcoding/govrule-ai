"""GovRule AI 백엔드 API."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.config import LLM_MODEL
from app.llm.client import chat, health, model_available

app = FastAPI(
    title="GovRule AI",
    description="공공기관 계약·구매 법령 검토 API (온프레미스)",
    version="0.1.0",
)


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str


@app.get("/health")
def get_health():
    """Ollama 추론 서버와 모델 상태를 확인한다."""
    server_ok = health()
    model_ok = model_available() if server_ok else False
    return {
        "status": "ok" if (server_ok and model_ok) else "degraded",
        "ollama": server_ok,
        "model": LLM_MODEL,
        "model_available": model_ok,
    }


@app.post("/chat", response_model=ChatResponse)
def post_chat(req: ChatRequest):
    """질문에 대한 답변을 생성한다."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="질문이 비어 있습니다")
    try:
        return ChatResponse(answer=chat(req.question))
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"추론 실패: {e}")
