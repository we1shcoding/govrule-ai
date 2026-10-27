"""환경변수 로드.

.env 를 이 파일 기준 절대경로로 찾는다.
실행 위치(cwd)에 의존하면 uvicorn / streamlit / 스크립트 직접 실행에서
각각 다른 결과가 나오므로 고정한다.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def _require(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise RuntimeError(
            f".env 에 {key} 가 없습니다. .env.example 을 복사해 값을 채우십시오."
        )
    return value


# --- 법제처 OPEN API ---
LAW_API_OC = _require("LAW_API_OC")
LAW_API_BASE = os.getenv("LAW_API_BASE", "https://www.law.go.kr/DRF")

# --- Ollama ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Modelfile 로 생성한 태그여야 한다.
# 베이스 태그(qwen3.5:4b)를 넣으면 num_ctx 가 262144 로 돌아가
# 8GB VRAM 에서 CPU 오프로드가 발생한다 (ADR 0003).
LLM_MODEL = os.getenv("LLM_MODEL", "govrule-4b")

# --- 경로 ---
DATA_RAW = ROOT / "data" / "raw"


if __name__ == "__main__":
    print(f"ROOT          : {ROOT}")
    print(f"LAW_API_OC    : {LAW_API_OC[:2]}***")
    print(f"LAW_API_BASE  : {LAW_API_BASE}")
    print(f"OLLAMA_BASE_URL: {OLLAMA_BASE_URL}")
    print(f"LLM_MODEL     : {LLM_MODEL}")