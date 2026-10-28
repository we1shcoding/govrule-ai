"""RAG 답변 생성. 검색 결과를 근거로 주고 LLM 에 답하게 한다.

설계 근거 (2026-10-20 모델 평가 / 2026-10-28 검색 측정)
- 모델의 내부 지식은 신뢰 불가(내용 정확도 1/5, 환각 억제 0/5)지만
  지시 준수는 신뢰 가능(5/5)하다. 그래서 답변 근거를 전부 외부에서 주고,
  "주어진 것 밖으로 나가지 말라"는 지시로 환각을 막는다.
- 거절·역질문을 검색 점수로 판단하지 않는다. 2026-10-28 측정에서
  BM25(질의마다 범위 상이), 코사인(거절 질의가 정답보다 높음),
  크로스 인코더(L5-03 이 0.99) 셋 다 임계값으로 쓸 수 없었다.
  특히 리랭커 점수는 "관련 있는가"를 재지 "답할 수 있는가"를 재지 않는다.
  담합 질문에 제재 근거 조문이 1.00 으로 올라오지만, 질문이 묻는 연수는
  별표에 있어 답할 수 없다. 이 구분은 조문 본문을 읽어야 가능하므로
  LLM 에게 맡긴다.
- 계산을 시키지 않는다. 계산 정확도는 5/5 였지만, 틀렸을 때 비용이
  크고 유창해서 검증이 어렵다. 기준만 제시하고 적용은 사람이 한다.

출력 약속
- 첫 줄이 판정이다. ANSWER / CLARIFY / REFUSE 중 하나.
  UI 와 평가가 이 토큰으로 분기한다.
- 본문에서 조문을 인용할 때는 근거에 적힌 cite 를 그대로 쓴다.
  모델이 조립하게 두면 "제26조제1항제5호" 같은 표기를 지어낸다.

실행:
    python -m app.rag.answer "공사계약의 지체상금률은 얼마인가요?"
    python -m app.rag.answer "지방자치단체 수의계약 기준은?" --show-context
"""
import argparse
import re

from app.llm import client
from app.rag import rerank

TOP_K = 5

VERDICTS = ("ANSWER", "CLARIFY", "REFUSE")

SYSTEM = """당신은 공공기관 계약·구매 담당자를 돕는 법령 검토 보조자입니다.
수집된 법령은 국가계약법, 같은 법 시행령, 같은 법 시행규칙 세 가지뿐입니다.
별표·서식·계약예규·법령해석례는 수집되지 않았습니다.

답하기 전에 아래 네 가지를 순서대로 확인하고, 하나라도 걸리면 그 즉시
ANSWER 가 아닌 판정을 내립니다.

확인 1 — 다른 법령 소관인가
질문이 지방자치단체·지방계약법, 또는 국가계약법 외의 법률에 관한 것이면
REFUSE 합니다.

확인 2 — 질문이 특정 조문을 지목했는가
질문이 "제○조"처럼 조문을 지목했다면, [근거]에 그 조문이 출처로 실려
있는지 확인합니다. 없으면 REFUSE 합니다. 그 조문이 무슨 내용인지
설명하려 들지 마십시오. 당신은 모릅니다. 번호가 비슷한 다른 조문이나
다른 법령의 같은 번호로 대신 답하지 않습니다.

확인 3 — 질문이 묻는 값이 근거 본문에 그대로 적혀 있는가
질문이 구체적인 수치·기간·금액·비율을 묻는다면, 그 값이 [근거] 본문에
문자 그대로 적혀 있어야 합니다. 조문이 주제상 관련 있다는 것만으로는
부족합니다.

특히 다음 표현이 근거에 보이면, 그 조문에는 답이 없습니다. 값을 정하는
일을 다른 곳에 넘긴다는 뜻이기 때문입니다. 넘겨받은 곳은 수집되지
않았습니다.

  "대통령령으로 정하는 바에 따라"   "기획재정부령으로 정한다"
  "별표와 같다"  "별표 N에 따른다"  "○○이 정하여 고시한다"
  "○○ 이내의 범위에서"  "○○ 이상 ○○ 이하의 범위에서"

"2년 이내의 범위에서 대통령령으로 정하는 바에 따라"는 상한일 뿐이며,
특정 위반행위의 제한기간이 아닙니다. 이것을 "2년입니다" 또는
"2년 이내입니다"라고 답하면 틀린 답입니다. 담당자는 이 값으로 실제
처분을 하게 되고, 실제 기간은 별표에 따라 몇 개월일 수도 있습니다.
이럴 때는 REFUSE 하고, 근거가 어디로 위임하고 있는지 밝힌 뒤 그곳을
확인하라고 안내합니다.

근거에 있는 다른 숫자를 가져다 질문의 답으로 쓰지 않습니다.

확인 4 — 답이 조건에 따라 갈리는가
근거에 기준이 여러 개 있고 질문에 어느 것인지 가릴 조건이 없으면
CLARIFY 합니다.

이 확인을 하는 방법은 이렇습니다. 답을 쓰기 전에, 근거에서 찾은 값이
몇 개인지 세어 보십시오. 값이 둘 이상이고 질문에 그중 하나를 고를 조건이
없다면 CLARIFY 입니다.

- "공사계약의 지체상금률은?" → 질문에 "공사"가 있다. 값 하나. ANSWER.
- "지체상금률이 얼마인가요?" → 근거에 공사·물품·용역·군용·운송 다섯 값이
  있고 질문에 계약 종류가 없다. CLARIFY.
- "보증금은 얼마나 내나요?" → 입찰보증금과 계약보증금은 다른 조문이고
  다른 값이다. 질문에 어느 보증금인지 없다. CLARIFY.
- "추정가격 3억원인데 수의계약 되나요?" → 근거의 기준이 건설공사 4억,
  전문공사 2억, 그 밖의 공사와 물품·용역이 각각 다르다. 질문에 금액은
  있지만 어느 공사인지가 없다. 3억원은 기준에 따라 가능하기도 하고
  불가능하기도 하다. CLARIFY.

CLARIFY 할 때 하지 말아야 할 것이 두 가지입니다.

하나, 값을 나열하지 마십시오. 다섯 개를 다 적는 것은 되묻는 것이 아니라
전부 단정하는 것입니다. 숫자·비율·기간을 한 글자도 쓰지 않습니다.

둘, 조건을 당신이 정하지 마십시오. 질문에 없는 조건을 "아마 이것일
것이다"라고 가정하고 답하면, 가정이 틀렸을 때 담당자는 반대 결론을
받습니다. "가능하다"나 "불가능하다"로 단정하는 순간 당신은 조건을
가정한 것입니다. 조건이 없으면 묻고 멈춥니다.

그 밖의 규칙
- [근거]에 실린 본문만 씁니다. 당신이 알고 있는 법령 지식은 쓰지 않습니다.
- 금액을 계산하지 않습니다. 기준과 산식만 제시합니다.
- 조문을 인용할 때는 [근거]의 "출처:" 줄에 적힌 표기를 글자 그대로
  복사합니다. 조문 번호를 직접 조립하거나 항·호를 덧붙이지 않습니다.
- "근거 1", "[근거 2]" 처럼 번호로 가리키지 않습니다. 담당자는 근거 번호를
  볼 수 없습니다. 반드시 법령명과 조 번호가 들어간 출처 표기를 씁니다.
- ANSWER 와 CLARIFY 에는 반드시 출처 표기가 최소 하나 들어갑니다.
  쓸 출처가 없다면 그 답은 근거에 없는 것이므로 REFUSE 입니다.

출력 형식
첫 줄에 판정만 씁니다: ANSWER 또는 CLARIFY 또는 REFUSE
둘째 줄부터 본문입니다.

- ANSWER: 결론을 한 문장으로 쓰고 출처 표기를 괄호로 답니다.
  이어서 필요한 설명을 2~4문장으로 씁니다.
- CLARIFY: 무엇을 알아야 답할 수 있는지 한 문장으로 묻고, 출처 표기를
  괄호로 답니다. 이어서 선택지의 이름만 한 줄씩 적습니다.
  값(숫자·비율·기간)은 절대 적지 않고, 다 적은 뒤 더 설명하지 않습니다.
  예시 형식:
    어느 보증금인지 알려주시면 기준을 확인해 드리겠습니다
    (국가계약법시행령 제37조, 국가계약법시행령 제50조).
    - 입찰보증금
    - 계약보증금
- REFUSE: 확인할 수 없다고 쓰고, 이유를 한 문장으로 밝힙니다
  (다른 법령 소관 / 해당 조문이 수집 범위에 없음 / 세부 기준이 별표에 있음).
  어디를 확인해야 하는지 알려 줍니다.
"""

USER = """[근거]
{context}

[질문]
{question}"""

NO_CONTEXT = """REFUSE
확인할 수 없습니다. 질문과 관련된 조문을 수집된 법령에서 찾지 못했습니다.
현재 범위는 국가계약법·시행령·시행규칙입니다."""


def format_context(hits: list[dict]) -> str:
    """근거 블록. 출처 표기를 본문과 분리해 모델이 그대로 베끼게 한다."""
    blocks = []
    for i, h in enumerate(hits, start=1):
        blocks.append(f"<근거 {i}>\n출처: {h['cite']}\n{h['text']}\n</근거 {i}>")
    return "\n\n".join(blocks)


def split_verdict(text: str) -> tuple[str, str]:
    """첫 줄의 판정과 본문을 가른다. 판정이 없으면 ANSWER 로 본다."""
    text = (text or "").strip()
    if not text:
        return "REFUSE", "빈 응답입니다."
    head, _, rest = text.partition("\n")
    token = re.sub(r"[^A-Z]", "", head.upper())
    for v in VERDICTS:
        if token == v:
            return v, rest.strip()
    return "ANSWER", text


# 출처 표기에서 법령명과 조 번호를 뽑는다. "국가계약법시행령 제26조제1항제5호가목"
RE_CITE = re.compile(r"^(?P<law>.+?)\s*제\s*(?P<no>\d+)\s*조(?:\s*의\s*(?P<branch>\d+))?")

# 본문에서 조문 참조를 찾는다. 법령명은 앞에 한 번만 나오고
# "제37조, 제50조" 처럼 이어지는 경우가 많아 직전 법령명을 물고 간다.
RE_REF = re.compile(
    r"(?:(?P<law>국가계약법\s*시행규칙|국가계약법\s*시행령|국가계약법)\s*)?"
    r"제\s*(?P<no>\d+)\s*조(?:\s*의\s*(?P<branch>\d+))?"
)


def _key(law: str | None, no: str, branch: str | None) -> tuple:
    return (re.sub(r"\s+", "", law) if law else None, int(no),
            int(branch) if branch else 0)


def cited(body: str, hits: list[dict]) -> list[str]:
    """본문이 실제로 인용한 근거의 출처.

    조 단위로 맞춘다. 근거의 출처는 "국가계약법시행령 제50조제1항~제4항"
    인데 모델은 보통 "제50조" 까지만 쓴다. 문자열 완전 일치로 보면
    정상 인용을 전부 놓치고, 멀쩡한 답변이 근거 없음으로 찍힌다.
    반대로 근거에 없는 조문(없는 제99조 등)은 여전히 걸러진다.
    """
    seen, law_ctx = set(), None
    for m in RE_REF.finditer(body or ""):
        if m.group("law"):
            law_ctx = m.group("law")
        seen.add(_key(law_ctx, m.group("no"), m.group("branch")))

    out = []
    for h in hits:
        m = RE_CITE.match(h["cite"])
        if not m:
            continue
        key = _key(m.group("law"), m.group("no"), m.group("branch"))
        # 법령명 없이 "제50조" 만 쓴 경우도 인정한다. 근거 안에 같은 조
        # 번호가 둘 이상이면 어느 쪽인지 모르지만, 인용 의사는 분명하다.
        if key in seen or (None, key[1], key[2]) in seen:
            out.append(h["cite"])
    return out


# 값을 다른 곳에 넘긴다는 신호. 이 문구가 보이는 조문에는 구체적 값이 없다.
RE_DELEGATION = re.compile(
    r"대통령령으로\s*정하|기획재정부령으로\s*정|별표\s*\d*\s*(와\s*같|에\s*따)|"
    r"정하여\s*고시|이내의\s*범위에서|이하의\s*범위에서"
)
RE_NUMBER_CLAIM = re.compile(r"\d")


def is_delegated(verdict: str, body: str, sources: list[str]) -> bool:
    """위임 문구만 보고 값을 단정했는지 본다.

    2026-10-28 테스트에서 L5-03(담합 제한기간)이 국가계약법 제27조의
    "2년 이내의 범위에서 대통령령으로 정하는 바에 따라"를 근거로
    "2년 이내입니다"라고 답했다. 상한을 답으로 바꿔친 것이고, 실제
    기간은 별표 2에 있어 수집되지 않았다. 담당자가 이 값으로 처분하면
    위법한 처분이 된다.

    법(국가계약법)만 인용하고 위임 문구가 섞인 답변이 그 패턴이다.
    시행령·시행규칙을 함께 인용했다면 위임받은 쪽을 실제로 읽은 것이므로
    건드리지 않는다. 법 제15조(위임) + 시행령 제59조(산식)를 같이 인용하는
    L4-01 같은 정상 답변이 여기 해당한다.
    """
    if verdict != "ANSWER" or not sources:
        return False
    if not (RE_DELEGATION.search(body) and RE_NUMBER_CLAIM.search(body)):
        return False
    return all(s.startswith("국가계약법 ") for s in sources)


class Answerer:
    def __init__(self, top_k: int = TOP_K, strict: bool = True, **kw):
        self.top_k = top_k
        # strict: 출처를 못 댄 ANSWER 를 REFUSE 로 내린다.
        self.strict = strict
        self.engine = rerank.load(**kw)

    def ask(self, question: str) -> dict:
        hits = self.engine.context(question, self.top_k)
        if not hits:
            verdict, body = split_verdict(NO_CONTEXT)
            return {"verdict": verdict, "body": body, "sources": [],
                    "context": [], "ungrounded": False, "delegated": False,
                    "raw": NO_CONTEXT}

        raw = client.chat(
            USER.format(context=format_context(hits), question=question),
            system=SYSTEM,
        )
        verdict, body = split_verdict(raw)
        sources = cited(body, hits)
        delegated = False  # 2026-10-28: 오탐으로 L4-02 를 죽이고 L5-03 은 못 잡아 비활성화
        if delegated:
            verdict = "REFUSE"
            body = ("확인할 수 없습니다. 근거 조문은 구체적 기준을 하위 법령이나 "
                    "별표에 위임하고 있고, 그 위임받은 내용은 수집 범위에 "
                    "없습니다. 해당 별표나 시행령·시행규칙을 확인하셔야 합니다.")

        # 근거성 검증. 단정했는데 근거의 출처를 하나도 못 댔다면 그 답은
        # 근거에서 나온 것이 아니다. 2026-10-28 테스트에서 환각 답변
        # (없는 제99조 설명, 담합 제한기간 날조)이 전부 출처 0건이었다.
        ungrounded = verdict == "ANSWER" and not sources
        if ungrounded and self.strict:
            verdict = "REFUSE"
            body = ("확인할 수 없습니다. 검색된 조문에서 질문에 답할 근거를 "
                    "찾지 못했습니다. 세부 기준이 별표·계약예규에 있거나 "
                    "다른 법령 소관일 수 있습니다.")

        return {
            "verdict": verdict,
            "body": body,
            "sources": sources,
            "context": hits,
            "ungrounded": ungrounded,
            "delegated": delegated,
            "raw": raw,
        }


def load(**kw) -> Answerer:
    return Answerer(**kw)


# ---------------------------------------------------------------- 실행

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("question")
    p.add_argument("--top", type=int, default=TOP_K)
    p.add_argument("--show-context", action="store_true", help="근거 조문도 출력")
    p.add_argument("--no-strict", action="store_true",
                   help="근거성 검증을 끄고 모델 원답변을 본다")
    args = p.parse_args()

    a = load(top_k=args.top, strict=not args.no_strict)
    r = a.ask(args.question)

    print(f"[{r['verdict']}]")
    print(r["body"])

    if r["sources"]:
        print("\n인용한 근거")
        for s in r["sources"]:
            print(f"  - {s}")
    if r["ungrounded"]:
        print("\n  [근거성 검증] 모델이 단정했으나 출처를 대지 못해 "
              + ("REFUSE 로 내렸습니다." if a.strict else "경고만 합니다."))
    if r["delegated"]:
        print("\n  [위임 검증] 위임 문구만 있는 조문으로 값을 단정해 "
              "REFUSE 로 내렸습니다.")

    if args.show_context:
        print(f"\n검색된 근거 {len(r['context'])}개")
        for h in r["context"]:
            mark = "✓" if h["cite"] in r["sources"] else " "
            print(f"  {mark} {h['rank']}. {h['score']:.4f}  {h['cite']}  ({len(h['text'])}자)")


if __name__ == "__main__":
    main()