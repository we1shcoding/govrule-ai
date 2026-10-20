"""Ollama 로컬 추론 클라이언트.

외부 API를 사용하지 않고 localhost:11434 의 Ollama 서버에만 요청한다.
"""
import requests

OLLAMA_URL = "http://localhost:11434"
MODEL = "govrule-4b"
TIMEOUT = 120


def chat(question: str, system: str | None = None) -> str:
    """질문을 보내고 답변 전문을 받는다."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": question})

    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": MODEL,
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
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        return r.status_code == 200
    except requests.RequestException:
        return False


if __name__ == "__main__":
    print("Ollama 연결:", "OK" if health() else "실패")
    print()
    print(chat("공공기관 계약 담당자에게 수의계약을 2문장으로 설명해줘"))
