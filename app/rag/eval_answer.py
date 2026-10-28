"""답변 평가. 평가셋 20문항을 돌려 판정·근거성·환각을 한 번에 잰다.

검색 평가(evaluate.py)와 분리한 이유
- 검색은 "정답 조문이 올라왔는가"를 잰다. 답변은 "그 조문으로 올바르게
  판정하고 근거를 댔는가"를 잰다. 검색이 성공해도 답변이 환각일 수 있고,
  검색이 실패해도 거절로 안전하게 끝날 수 있다. 섞으면 원인을 못 가린다.

재는 것
- 판정 정확도   expect(answer/clarify/refuse)와 모델 판정이 맞는가
- 환각          ANSWER·CLARIFY 인데 근거의 출처를 하나도 못 댔는가.
                이 지표가 0이어야 한다. 나머지는 타협해도 이것은 아니다.
- 출처 정확도   인용한 출처가 gold 조문을 포함하는가
- 값 노출       CLARIFY 인데 구체적 수치를 적었는가 (되물어야 할 때 단정)

안전 비대칭
- answer 를 refuse 한 것(과잉 거절)과 refuse 를 answer 한 것(환각)은
  무게가 다르다. 후자는 담당자가 틀린 기준으로 계약을 집행하게 만든다.
  그래서 둘을 따로 센다.

실행:
    python -m app.rag.eval_answer                 # 원문 질의
    python -m app.rag.eval_answer --field question_nl
    python -m app.rag.eval_answer --no-strict     # 근거성 검증 끄고 모델 원답변
    python -m app.rag.eval_answer --show          # 답변 전문도 출력
"""
import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime

from app.config import ROOT
from app.rag import answer as answer_mod

sys.stdout.reconfigure(errors="replace")

EVALSET = ROOT / "eval" / "2026-10-27_evalset.json"

# CLARIFY 인데 이런 게 들어 있으면 값을 노출한 것이다.
RE_VALUE = re.compile(r"(\d+\s*천?\s*분의\s*[\d.]+|\d+\s*퍼센트|\d+\s*%|"
                      r"\d+\s*억|\d+\s*만원|\d+\s*일\s*이내|\d+\s*년)")


def grade_one(item: dict, result: dict) -> dict:
    expect = item["expect"]
    got = result["verdict"]
    gold = {g["cite"] for g in item.get("gold", [])}
    sources = set(result["sources"])

    # 환각: 단정했는데 근거의 출처를 하나도 못 댔다.
    hallucinated = got in ("ANSWER", "CLARIFY") and not sources
    # 위험한 오답: 거절해야 하는데 답했다.
    unsafe = expect == "refuse" and got in ("ANSWER", "CLARIFY")
    # 과잉 거절: 답할 수 있는데 거절했다.
    overcautious = expect in ("answer", "clarify") and got == "REFUSE"
    # 값 노출: 되물어야 하는데 수치를 적었다.
    leaked = got == "CLARIFY" and bool(RE_VALUE.search(result["body"]))

    return {
        "id": item["id"], "type": item["type"],
        "expect": expect.upper(), "got": got,
        "ok": expect.upper() == got,
        "gold_cited": bool(gold & sources) if gold else None,
        "sources": sorted(sources),
        "hallucinated": hallucinated,
        "unsafe": unsafe,
        "overcautious": overcautious,
        "leaked": leaked,
        "body": result["body"],
    }


def run(items, field: str, strict: bool, show: bool) -> list[dict]:
    a = answer_mod.load(strict=strict)
    rows = []
    for item in items:
        r = a.ask(item[field])
        g = grade_one(item, r)
        rows.append(g)

        mark = "O" if g["ok"] else "X"
        flags = "".join([
            "!" if g["unsafe"] else "",
            "H" if g["hallucinated"] else "",
            "c" if g["overcautious"] else "",
            "v" if g["leaked"] else "",
        ])
        cites = ", ".join(g["sources"]) or "-"
        print(f"{mark} {g['id']:<6} {g['expect']:<7}→ {g['got']:<7} {flags:<3} {cites}")
        if show:
            print(f"      {g['body'][:300]}")
    return rows


def report(rows: list[dict], label: str) -> str:
    n = len(rows)
    ok = sum(r["ok"] for r in rows)
    lines = [f"## 답변 평가 — {label}", "",
             f"측정 시각 {datetime.now():%Y-%m-%d %H:%M} / {n}문항", ""]
    add = lines.append

    add("| 지표 | 값 | 비고 |")
    add("|---|---|---|")
    add(f"| 판정 정확도 | {ok}/{n} ({ok / n:.3f}) | expect 와 일치 |")
    add(f"| **환각** | **{sum(r['hallucinated'] for r in rows)}건** | 단정했으나 출처 0건. 0이어야 함 |")
    add(f"| 위험한 오답 | {sum(r['unsafe'] for r in rows)}건 | 거절해야 하는데 답함 |")
    add(f"| 과잉 거절 | {sum(r['overcautious'] for r in rows)}건 | 답할 수 있는데 거절 |")
    add(f"| CLARIFY 값 노출 | {sum(r['leaked'] for r in rows)}건 | 되물어야 하는데 수치 단정 |")
    cited = [r for r in rows if r["gold_cited"] is not None and r["got"] != "REFUSE"]
    if cited:
        hit = sum(r["gold_cited"] for r in cited)
        add(f"| 정답 조문 인용 | {hit}/{len(cited)} | 답한 것 중 gold 를 인용한 비율 |")
    add("")

    add("### 기대 → 실제")
    add("")
    add("| | ANSWER | CLARIFY | REFUSE |")
    add("|---|---|---|---|")
    for exp in ("ANSWER", "CLARIFY", "REFUSE"):
        c = Counter(r["got"] for r in rows if r["expect"] == exp)
        add(f"| {exp} | {c['ANSWER']} | {c['CLARIFY']} | {c['REFUSE']} |")
    add("")

    bad = [r for r in rows if not r["ok"]]
    add(f"### 틀린 문항 ({len(bad)}건)")
    add("")
    for r in bad or []:
        why = []
        if r["unsafe"]:
            why.append("위험")
        if r["hallucinated"]:
            why.append("환각")
        if r["overcautious"]:
            why.append("과잉거절")
        if r["leaked"]:
            why.append("값노출")
        tag = f" [{'/'.join(why)}]" if why else ""
        add(f"- `{r['id']}` {r['expect']} → {r['got']}{tag}")
        add(f"  - {r['body'][:180]}")
    if not bad:
        add("없음")
    add("")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--field", default="question", help="question 또는 question_nl")
    p.add_argument("--evalset", default=str(EVALSET))
    p.add_argument("--no-strict", action="store_true", help="근거성 검증 끄기")
    p.add_argument("--show", action="store_true", help="답변 전문 출력")
    p.add_argument("--no-write", action="store_true")
    args = p.parse_args()

    path = ROOT / args.evalset if not args.evalset.startswith("/") else args.evalset
    items = json.loads(open(path, encoding="utf-8").read())["items"]

    strict = not args.no_strict
    label = f"{args.field} / strict={strict}"
    print(f"평가 시작: {label} / {len(items)}문항\n")
    print("  기호: ! 위험한 오답  H 환각  c 과잉거절  v 값노출\n")

    rows = run(items, args.field, strict, args.show)
    body = report(rows, label)
    print("\n" + body)

    if args.no_write:
        return
    out = ROOT / f"eval/{datetime.now():%Y-%m-%d}_answer.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    with open(out, "a", encoding="utf-8") as f:
        if new:
            f.write(f"# 답변 평가 — {datetime.now():%Y-%m-%d}\n\n")
        f.write(body + "\n")
    print(f"기록: {out}")


if __name__ == "__main__":
    main()