# 개발 진행 로그

## 2026-10-20

### 완료
- Git 2.56.0 설치, GitHub 저장소 생성 (Public, MIT)
- 프로젝트 구조 및 .gitignore 설정
- 법제처 OPEN API 신청 → 자동승인, OC 발급
- API 호출 테스트: law / admrul / expc 3종 모두 성공
- Python 3.11.9 + venv 구성, 기본 패키지 설치
- LICENSES.md 생성, 카피레프트 패키지 0건 확인
  ※ 2026-10-27 정정: certifi(MPL-2.0) 누락. 아래 10-27 항목 참조
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

---

## 2026-10-27

### 완료
- app/config.py 신설 — .env 를 파일 기준 절대경로로 로드
  (cwd 의존 제거. 컨테이너 WORKDIR 에서 깨지지 않도록)
- client.py 의 모델명 하드코딩 제거, model_available() 추가
  → clone 직후 ollama create 누락을 추론 시점이 아니라 기동 시점에 감지
- .env 의 LLM_MODEL 을 qwen3.5:4b → govrule-4b 로 정정
  (베이스 태그를 쓰면 num_ctx 가 262144 로 복귀해 CPU 오프로드 발생)
- ADR 0001 라이선스 정책 작성 (빈 파일이었음)
- ADR 0002 개발 중 Docker 미사용 작성 (빈 파일이었음)
- app/collect/fetch_law.py — 국가계약법 3종 수집
  법 44조 / 시행령 163조 / 시행규칙 105조, 총 312개 조문을 data/raw/ 에 저장
- app/collect/inspect_raw.py — 조문 구조 조사
  결과: eval/2026-10-27_json_structure.txt
- app/api/main.py 의 sys.path.insert 제거
- README 설치 안내에 `ollama create` 단계 추가, ADR 표를 실제 파일과 일치시킴

### 이슈
- LICENSES.md 의 "카피레프트 0건"이 사실과 다름을 확인.
  certifi 2026.10.19 가 MPL-2.0 으로 정책(MIT/Apache/BSD/PSF) 밖.
  → 비수정 사용이라 의무가 전파되지 않으므로 유지하되 ADR 0001 에 예외로 명시.
    표현을 "전염성 카피레프트 0건, 파일 단위 1건(비수정 사용)"으로 정정
- 법제처 API 응답에서 파싱 함정 6건 확인 (inspect_raw 결과)
  1. 항이 1개면 dict, 2개 이상이면 list — 법 2건 / 시행규칙 22건 / 시행령 30건이 dict
  2. dict 인 항에는 항번호·항내용 키가 아예 없음 (호만 존재)
  3. 조문가지번호 (제4조의2 형태) 법 9 / 시행규칙 10 / 시행령 36건
     → 조문번호만 쓰면 제4조와 제4조의2 가 충돌. 조문키(0004002)를 기본키로 검토
  4. 조문여부='전문' 행은 장 제목 (시행령 10 / 시행규칙 8건, 법은 0건)
     → 색인 제외, 단 상위 맥락으로는 보존
  5. 조문내용이 항 유무에 따라 전문 또는 제목만 담음 → 본문은 이어붙여야 함
  6. "삭제<2023.7.18>" 형태의 폐지 조문 존재 → 플래그 처리 필요
- 시행규칙에만 별표단위 20건 존재. 금액 기준표 가능성 있어 별도 확인 필요

### 다음
- 조문 파서 작성 (app/collect/parse_law.py) — 위 함정 6건 반영
- SQLite 스키마 설계 (app/db/) — 조문키 기본키, 장 맥락·삭제여부 컬럼 포함
- 312개 조문이 누락·중복 없이 적재되는지 검증
