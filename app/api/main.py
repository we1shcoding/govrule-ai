"""GovRule AI 백엔드 API."""
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.llm.client import chat, health

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
    """Ollama 추론 서버 상태를 확인한다."""
    ok = health()
    return {"status": "ok" if ok else "degraded", "ollama": ok}


@app.post("/chat", response_model=ChatResponse)
def post_chat(req: ChatRequest):
    """질문에 대한 답변을 생성한다."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="질문이 비어 있습니다")
    try:
        return ChatResponse(answer=chat(req.question))
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"추론 실패: {e}")
