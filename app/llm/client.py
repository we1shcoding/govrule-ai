"""Ollama 로컬 추론 클라이언트.

외부 API를 사용하지 않고 .env 의 OLLAMA_BASE_URL(기본 localhost:11434)에만
요청한다.
"""
import requests

from app.config import LLM_MODEL, OLLAMA_BASE_URL

TIMEOUT = 120


def chat(question: str, system: str | None = None) -> str:
    """질문을 보내고 답변 전문을 받는다."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": question})

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": LLM_MODEL,
            "messages": messages,
            # 4B thinking은 수렴하지 않아 토큰만 소모한다 (ADR 0003)
            "think": False,
            "stream": False,
        },
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def health() -> bool:
    """Ollama 서버가 살아있는지 확인한다."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return r.status_code == 200
    except requests.RequestException:
        return False


def model_available() -> bool:
    """LLM_MODEL 태그가 로컬에 존재하는지 확인한다.

    clone 직후 `ollama create` 를 하지 않으면 추론 시점에야
    model not found 가 나므로, 기동 시점에 먼저 확인한다.
    """
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        r.raise_for_status()
        tags = {m["name"] for m in r.json().get("models", [])}
    except (requests.RequestException, KeyError, ValueError):
        return False
    # Ollama 는 태그를 "govrule-4b:latest" 형태로 반환한다
    return LLM_MODEL in tags or f"{LLM_MODEL}:latest" in tags


if __name__ == "__main__":
    print("Ollama 연결:", "OK" if health() else "실패")
    print(f"모델({LLM_MODEL}):", "OK" if model_available() else "없음")
    if not model_available():
        print()
        print("  ollama create govrule-4b -f app/llm/Modelfile.govrule")
        raise SystemExit(1)
    print()
    print(chat("공공기관 계약 담당자에게 수의계약을 2문장으로 설명해줘"))