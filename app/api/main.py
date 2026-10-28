"""GovRule AI 백엔드 API."""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.config import LLM_MODEL
from app.llm.client import health, model_available
from app.rag import answer as answer_mod

# 리랭커·임베딩 모델을 들고 있어 생성에 20초쯤 걸린다. 요청마다 만들면
# 매번 그만큼 멈추므로 기동 때 한 번만 만들어 재사용한다.
_answerer: answer_mod.Answerer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _answerer
    _answerer = answer_mod.load()
    yield
    _answerer = None


app = FastAPI(
    title="GovRule AI",
    description="공공기관 계약·구매 법령 검토 API (온프레미스)",
    version="0.2.0",
    lifespan=lifespan,
)


class ChatRequest(BaseModel):
    question: str


class Source(BaseModel):
    cite: str
    text: str
    rank: int
    score: float
    used: bool          # 답변이 실제로 이 조문을 인용했는가


class ChatResponse(BaseModel):
    answer: str
    verdict: str        # ANSWER / CLARIFY / REFUSE
    sources: list[Source]
    ungrounded: bool    # 단정했으나 출처를 못 대 REFUSE 로 내려간 경우


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
        "rag_ready": _answerer is not None,
    }


@app.post("/chat", response_model=ChatResponse)
def post_chat(req: ChatRequest):
    """검색된 조문을 근거로 답변을 생성한다."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="질문이 비어 있습니다")
    if _answerer is None:
        raise HTTPException(status_code=503, detail="검색 색인이 아직 준비되지 않았습니다")

    try:
        r = _answerer.ask(req.question)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"추론 실패: {e}")

    used = set(r["sources"])
    return ChatResponse(
        answer=r["body"],
        verdict=r["verdict"],
        ungrounded=r["ungrounded"],
        sources=[
            Source(cite=h["cite"], text=h["text"], rank=h["rank"],
                   score=h["score"], used=h["cite"] in used)
            for h in r["context"]
        ],
    )