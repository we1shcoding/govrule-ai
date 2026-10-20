# ADR 0003: LLM 선정 및 추론 파라미터

## 결정
qwen3.5:4b (Q4_K_M)를 Modelfile로 재구성한 govrule-4b를 사용한다.

| 파라미터 | 기본값 | 적용값 | 근거 |
|---|---|---|---|
| num_ctx | 262144 | 8192 | 8GB VRAM. 256K는 KV 캐시가 넘쳐 CPU 오프로드 발생 |
| temperature | 1.0 | 0.2 | 법령 인용의 재현성 확보 |
| presence_penalty | 1.5 | 0.0 | 법령 문체의 정상적 반복("제O조에 따라")을 보존 |
| repeat_penalty | - | 1.05 | 상동 |

## thinking 비활성화
qwen3.5는 thinking 기본 활성(default true)이나, 4B 규모에서는 결론이 수렴하지
않고 동일 문장을 반복 수정한다. "안녕" 한 마디에 800토큰 이상 소모를 확인했다.
Ollama 0.40.6은 Modelfile의 think 파라미터를 지원하지 않으므로
(Error: unknown parameter 'think'), CLI는 /set nothink, API 호출 시에는
요청 JSON의 "think": false로 고정한다.

## 검증
ollama ps 결과: PROCESSOR 100% GPU / CONTEXT 8192 / SIZE 3.4GB
추론 속도 86~88 tokens/s (RTX 3070 8GB)

## RAG 필요성의 실증
품질 테스트(eval/2026-10-20_model_test.md) 결과 6문항 중 3문항에서 환각이
발생했고, 존재하지 않는 법률명을 생성했다. 반면 "자료가 없으면 확인 불가라고만
답하라"는 제약 지시는 정확히 따랐다.
이에 따라 모델 내부 지식을 답변 근거로 사용하지 않고, 검색된 법령 원문만을
컨텍스트로 제공하는 구조를 채택한다.

## 대안
- 9b(Q4_K_M, 6.6GB): 가용 VRAM 7.3GB 대비 KV 캐시 포함 시 초과. 추후 실측 예정
- 상용 API: 온프레미스 요구사항에 위배
