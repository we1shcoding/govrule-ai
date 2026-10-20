# 개발 진행 로그

## 2026-10-20

### 완료
- Git 2.56.0 설치, GitHub 저장소 생성 (Public, MIT)
- 프로젝트 구조 및 .gitignore 설정
- 법제처 OPEN API 신청 → 자동승인, OC 발급
- API 호출 테스트: law / admrul / expc 3종 모두 성공
- Python 3.11.9 + venv 구성, 기본 패키지 설치
- LICENSES.md 생성, 카피레프트 패키지 0건 확인
- GPU 확인: RTX 3070 8GB / 드라이버 591.59 / CUDA 13.1
- Ollama 0.40.6 설치 (install.ps1) — 계정 로그인 없음, 클라우드 호출 불가 상태
- qwen3.5:4b 수신 (3.3GB, Q4_K_M, Apache-2.0), ollama list SIZE GB 표시 확인
- Modelfile로 govrule-4b 생성 — 컨텍스트 262144 → 8192
- ollama ps: PROCESSOR 100% GPU / CONTEXT 8192 검증
- 한국어 법령 품질 테스트 6문항 수행, 86~88 tokens/s 실측

### 이슈
- winget 미설치로 Git/Python 수동 설치
- PowerShell 실행 정책으로 venv 활성화 실패
  → Set-ExecutionPolicy RemoteSigned -Scope CurrentUser 로 해결
- qwen3.5 thinking 기본 활성 → "안녕" 한 마디에 800토큰 이상 소모
  → Modelfile의 think 파라미터 미지원(0.40.6). /set nothink 및 API 옵션으로 대응
- 품질 테스트에서 존재하지 않는 법률명 생성, 금액 기준 창작 확인
  → 모델 내부 지식 배제, 검색 원문 기반 생성 구조 확정 (ADR 0003)

### 다음
- 양자화 수준(Q4 vs Q8) 비교 채점, 9b 적재 가능 여부 실측
