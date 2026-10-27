# GovRule AI

> 공공기관 계약·구매 업무를 위한 **온프레미스 법령 검토 AI**

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-local-000000)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/status-개발중-orange)

---

## 프로젝트 개요

공공기관 계약·구매 담당자가 자연어로 질문하면, **실제 법령 원문을 검색해 근거와 함께** 검토 의견을 제시하는 시스템입니다.

외부 LLM API를 사용하지 않고 **개인 PC(RTX 3070 8GB) 한 대에서 전체 서비스가 동작**합니다. 질문과 데이터가 외부로 전송되지 않으므로, 내부망만 허용되는 공공기관 환경을 그대로 재현합니다.

### 일반 RAG 챗봇과 다른 점

| | 일반 RAG 챗봇 | GovRule AI |
|---|---|---|
| 근거 없는 질문 | 그럴듯하게 답변 | **"확인 불가" 반환** |
| 정보 부족 시 | 추측해서 답변 | **역질문으로 조건 확보** |
| 출처 | 문서명 정도 | **법령명 + 조문번호 + 시행일 + 원문 URL** |
| 계산 | LLM이 암산 | **Python 함수로 분리** |
| 법령 개정 | 구분 없음 | **현행/연혁 구분** |

---

## 시스템 구성

```
브라우저
   │
   ▼
Streamlit (:8501)          표현 계층 — 로직 없음
   │  HTTP
   ▼
FastAPI (:8000)            API 계층
   │
   ▼
RAG 파이프라인
   ├─ 하이브리드 검색 (BM25 + 벡터)
   ├─ 리랭커 (cross-encoder)
   ├─ 근거성 검증 (decision model)
   └─ Ollama (:11434) — 로컬 LLM
   │
   ▼
SQLite — 법령 원문 · 메타데이터 · 벡터
```

### 데이터 수집 파이프라인

```
법제처 OPEN API → JSON 수집 → 조문 단위 파싱
   → 메타데이터 저장 → 임베딩 → 벡터 색인
```

---

## 기술 스택

> 버전은 `pip freeze` 기준 실측값으로 유지합니다.

### 실행 환경

| 구분 | 내용 |
|---|---|
| OS | Windows 11 |
| GPU | NVIDIA RTX 3070 (VRAM 8GB) |
| RAM | DDR4-3200 16GB |
| Python | 3.11 |

### AI 모델 — 전부 로컬 실행, 비용 0

| 역할 | 모델 | 크기 | 라이선스 |
|---|---|---|---|
| LLM | `govrule-4b` (`qwen3.5:4b` 기반) | 약 3.4GB | Apache-2.0 |
| 임베딩 | `nlpai-lab/KURE-v1` | 1024차원 / 8192토큰 | **MIT** |
| 리랭커 | `BAAI/bge-reranker-v2-m3` | 약 0.57B | Apache-2.0 |
| 판정 | (검토 예정) decision model | — | 확인 필요 |

`govrule-4b`는 `qwen3.5:4b`에 컨텍스트 길이·temperature 등을 재설정한 로컬 모델입니다. 파라미터와 근거는 [ADR 0003](./docs/adr/0003-llm-selection.md) 참조.

- 외부 LLM API를 **사용하지 않습니다**
- Ollama 클라우드 모델(유료)을 **사용하지 않습니다** — 로컬 태그만 사용

### 라이브러리

| 패키지 | 버전 | 역할 | 라이선스 |
|---|---|---|---|
| fastapi | 0.150.0 | REST API | MIT |
| uvicorn | 0.54.0 | ASGI 서버 | BSD |
| streamlit | 1.66.0 | 웹 UI | Apache-2.0 |
| requests | 2.34.2 | HTTP 클라이언트 | Apache-2.0 |
| pydantic | 2.14.1 | 스키마 검증 | MIT |
| python-dotenv | 1.2.4 | 환경변수 로드 | BSD |

직접 의존성은 MIT / Apache-2.0 / BSD / PSF 계열만 사용합니다. 전이 의존성 중 `certifi`(MPL-2.0)는 정책 범위 밖이지만 비수정 사용이라 의무가 전파되지 않으므로, 예외로 명시하고 유지합니다. 판단 근거는 [ADR 0001](./docs/adr/0001-license-policy.md), 전체 목록은 [`LICENSES.md`](./LICENSES.md) 참조.

### 사용하지 않기로 한 것

| 기술 | 사유 |
|---|---|
| Kiwi / kiwipiepy | LGPL v3 — BM25 자체 구현으로 대체 |
| PyMuPDF | AGPL-3.0 — API가 텍스트를 제공하므로 불필요 |
| ChromaDB | 수천 건 규모에 과잉, SQLite 단일화 |
| lxml | 법제처 API가 JSON을 지원 |
| OpenAI / Claude API | 온프레미스 요구사항에 위배 |

---

## 데이터 출처

**법제처 국가법령정보 공동활용** — https://open.law.go.kr

| 대상 | target | 용도 |
|---|---|---|
| 현행법령 | `law` | 국가계약법 등 |
| 행정규칙 | `admrul` | 기획재정부 계약예규 |
| 법령해석례 | `expc` | 유권해석 |
| 법령 체계도 | `lsStmd` | 상하위 관계 |

- 인증키(OC)는 가입 이메일의 ID 부분입니다
- **수집한 법령 원문은 저장소에 포함하지 않습니다**(`data/` 제외). 재배포 조건을 고려한 결정이며, 수집 스크립트만 공개합니다

---

## 설치 및 실행

### 1. 사전 준비

```bash
# Ollama 설치 후 — 반드시 크기 태그를 명시
ollama pull qwen3.5:4b
ollama list          # SIZE에 GB가 표시되는지 확인
```

### 2. 환경 구축

```bash
git clone https://github.com/we1shcoding/govrule-ai.git
cd govrule-ai

python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### 3. 모델 생성

`qwen3.5:4b` 그대로는 쓰지 않습니다. Modelfile로 파라미터를 재설정한 `govrule-4b`를 만들어야 합니다.

```bash
ollama create govrule-4b -f app/llm/Modelfile.govrule
ollama list          # govrule-4b 가 보이면 성공
```

이 단계를 건너뛰면 실행 시 `model not found` 또는 VRAM 초과가 발생합니다.

### 4. 환경변수

`.env.example`을 복사해 `.env` 생성 후 법제처 OC 값 입력.

```
LAW_API_OC=your_email_id
LAW_API_BASE=https://www.law.go.kr/DRF
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=govrule-4b
```

설정이 제대로 읽히는지 확인:

```bash
python -m app.config         # LLM_MODEL 이 govrule-4b 로 출력되는지
python -m app.llm.client     # 서버·모델 확인 후 테스트 응답
```

### 5. 법령 수집

```bash
python -m app.collect.fetch_law      # data/raw/ 에 JSON 저장
python -m app.collect.inspect_raw    # 조문 구조 확인 (선택)
```

### 6. 실행

```bash
uvicorn app.api.main:app --reload --port 8000
streamlit run ui/streamlit_app.py
```

두 명령 모두 저장소 루트에서 실행해야 합니다.

---

## 평가 지표

> 2026-11-17 측정 예정. 값은 실측 후 갱신합니다.

| 지표 | 목표 | 측정값 | 측정 방법 |
|---|---|---|---|
| Recall@5 | ≥ 0.85 | TODO | 50문항 정답 조문 포함률 |
| 출처 정확성 | ≥ 0.95 | TODO | 반환 조문과 원문 자동 대조 |
| 근거성 통과율 | ≥ 0.85 | TODO | 판정 모델 검증 |
| **미존재 법령 거절률** | **1.00** | TODO | 함정 10문항 |
| 응답시간 p50 | ≤ 5초 | TODO | 질문→최종응답 |

### 평가셋 구성

| 유형 | 문항 | 내용 |
|---|---|---|
| L1 단순 조회 | 15 | 조문 직접 질의 |
| L2 조건 판단 | 10 | 금액·조건 매칭 |
| L3 정보 부족 | 8 | 역질문 발생 확인 |
| L4 다중 법령 | 7 | 상하위 법령 종합 |
| L5 거절 | 10 | 미존재 법령·범위 밖 |

---

## 개발 로드맵

- [x] **~10/27** 환경 구축 · 로컬 LLM · 전체 통신 확인 · 법령 수집
- [ ] **10/28~11/1** 조문 파싱 · SQLite 저장 · 평가셋 작성
- [ ] **11/2~11/10** 하이브리드 검색 · 리랭커 · 재측정
- [ ] **11/11~11/16** 근거성 검증 · 출처 표시 UI · Docker 패키징
- [ ] **11/17~11/20** 성능 측정 · 문서화 · 데모

---

## 설계 의사결정

주요 기술 선택의 근거는 [`docs/adr/`](./docs/adr/)에 기록합니다.

| 번호 | 제목 |
|---|---|
| [0001](./docs/adr/0001-license-policy.md) | 라이선스 정책 — 카피레프트 배제 |
| [0002](./docs/adr/0002-no-docker-in-dev.md) | 개발 중에는 Docker 를 쓰지 않는다 |
| [0003](./docs/adr/0003-llm-selection.md) | LLM 선정 및 추론 파라미터 |

---

## 면책

본 시스템은 담당자의 **업무 검토를 지원**하기 위한 것이며, 법률 자문이나 계약 가능 여부에 대한 최종 판단을 제공하지 않습니다. 최종 결정은 기관 내부 절차와 관계 법령에 따라야 합니다.

---

## 라이선스

본 저장소의 코드는 MIT 라이선스를 따릅니다. 사용된 모델과 라이브러리의 라이선스는 [`LICENSES.md`](./LICENSES.md)를 확인하십시오. 법령 데이터의 저작권은 법제처에 있습니다.
