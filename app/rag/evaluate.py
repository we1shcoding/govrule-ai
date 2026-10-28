"""검색 평가. 평가셋으로 Recall@k 를 재고 거절 임계값 분포를 뽑는다.

오늘 이후 하이브리드·리랭커 재측정에 그대로 다시 쓴다. 검색기만 갈아끼운다.

평가셋을 읽고 정한 것
- 검색 평가 대상은 gold 가 있는 18문항(정답 조문 22개)이다. L3(clarify)도
  포함한다. 역질문하려면 먼저 그 조문을 찾아야 하므로 검색은 성공해야 한다.
  L5-03 도 포함한다. 시행규칙 제76조는 찾되 별표 2가 없다는 건
  생성 단계에서 거절할 일이다.
- 임계값으로 잡을 수 있는 거절은 gold 가 빈 2문항뿐이다.
  L5-01(없는 제99조)은 점수가 바닥이라 잡힌다.
  L5-02(지방자치단체)는 "수의계약" 때문에 시행령 제26조가 높은 점수로 올라온다.
  소관 법령이 다르다는 사실은 검색 점수에 나타나지 않으므로, 임계값이 아니라
  프롬프트에서 막아야 한다. 이 스크립트는 그 사실을 숫자로 보여주기만 한다.

측정
- Recall@k  정답 조문 중 상위 k개 청크 안에서 찾은 비율 (문항별 평균)
- Hit@k     정답 조문을 하나라도 찾은 문항 비율
- MRR       첫 정답 조문의 순위 역수
- 청크 여러 개가 같은 조문을 가리키면 조문 하나로 센다. 다만 k 는 청크로
  센다. 실제로 LLM 에 넘어가는 단위가 청크이기 때문이다.

질의 두 벌
- question     조문 어휘를 그대로 쓴 질의. BM25 에 유리하다.
- question_nl  같은 gold 에 대한 자연어 질의. 어휘 겹침이 적다.
  BM25 단독으로 Recall@5 0.972 가 나와 천장에 닿았기 때문에, 하이브리드와
  리랭커의 효과는 이쪽에서만 측정할 수 있다. --both 로 둘 다 재서 비교한다.

실행:
    python -m app.rag.evaluate                      # BM25, 원문 질의
    python -m app.rag.evaluate --both               # 원문·자연어 나란히
    python -m app.rag.evaluate --query-field question_nl
    python -m app.rag.evaluate --b 0.3              # 파라미터 바꿔서
    python -m app.rag.evaluate --scores             # 임계값용 점수 분포까지
    python -m app.rag.evaluate --no-write           # 파일에 안 남기고 화면만
"""
import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime

from app.config import ROOT
from app.rag import bm25

sys.stdout.reconfigure(errors="replace")

EVALSET = ROOT / "eval" / "2026-10-27_evalset.json"
KS = (1, 3, 5, 10)


# ---------------------------------------------------------------- 검색기

def make_engine(name: str, **kw):
    """이름으로 검색 함수를 만든다. 하이브리드·리랭커가 붙을 자리다."""
    if name == "bm25":
        index = bm25.load(**kw)
        return index.search, f"bm25(k1={index.k1}, b={index.b})"
    if name == "vector":
        from app.rag import embed
        index = embed.load()
        return index.search, f"vector({embed.MODEL_NAME})"
    if name == "hybrid":
        from app.rag import search
        index = search.load(weight=kw.pop("weight", search.WEIGHT_BM25),
                            fusion=kw.pop("fusion", "weighted"), **kw)
        return index.search, index.label()
    if name == "rerank":
        from app.rag import rerank, search
        index = rerank.load(candidates=kw.pop("candidates", rerank.CANDIDATES),
                            weight=kw.pop("weight", search.WEIGHT_BM25),
                            fusion=kw.pop("fusion", "weighted"), **kw)
        return index.search, index.label()
    raise SystemExit(f"모르는 검색기: {name}")


# ---------------------------------------------------------------- 채점

def grade(items, search, max_k: int, field: str = "question"):
    """문항별 결과를 만든다. field 로 어느 질의를 쓸지 고른다."""
    out = []
    for item in items:
        gold = {(g["law_id"], g["article_key"]) for g in item.get("gold", [])}
        hits = search(item[field], max_k)

        # 청크 순서대로 보면서 정답 조문이 처음 나온 순위를 기록한다.
        rank_of = {}
        for i, h in enumerate(hits, start=1):
            key = (h["law_id"], h["article_key"])
            if key in gold and key not in rank_of:
                rank_of[key] = i

        out.append({
            "item": item,
            "gold": gold,
            "hits": hits,
            "rank_of": rank_of,
            "first_rank": min(rank_of.values()) if rank_of else None,
            "top_score": hits[0]["score"] if hits else 0.0,
        })
    return out


def metrics(results, ks):
    """gold 가 있는 문항만 집계한다."""
    scored = [r for r in results if r["gold"]]
    agg = {}
    for k in ks:
        recalls = [
            len([v for v in r["rank_of"].values() if v <= k]) / len(r["gold"])
            for r in scored
        ]
        hits = [any(v <= k for v in r["rank_of"].values()) for r in scored]
        agg[k] = {
            "recall": sum(recalls) / len(recalls),
            "hit": sum(hits) / len(hits),
        }
    mrr = sum(1 / r["first_rank"] if r["first_rank"] else 0 for r in scored) / len(scored)
    return agg, mrr, len(scored)


def by_type(results, k: int):
    groups = defaultdict(list)
    for r in results:
        if r["gold"]:
            groups[r["item"]["type"]].append(r)
    out = {}
    for t, rs in sorted(groups.items()):
        recalls = [
            len([v for v in r["rank_of"].values() if v <= k]) / len(r["gold"])
            for r in rs
        ]
        out[t] = (sum(recalls) / len(recalls), len(rs))
    return out


# ---------------------------------------------------------------- 출력

def render(results, label: str, ks, k_main: int, show_scores: bool) -> str:
    agg, mrr, n = metrics(results, ks)
    lines = []
    add = lines.append

    add(f"## {label}")
    add("")
    add(f"측정 시각 {datetime.now():%Y-%m-%d %H:%M} / 검색 평가 대상 {n}문항 "
        f"/ 정답 조문 {sum(len(r['gold']) for r in results)}개")
    add("")
    add("| k | Recall@k | Hit@k |")
    add("|---|---|---|")
    for k in ks:
        add(f"| {k} | {agg[k]['recall']:.3f} | {agg[k]['hit']:.3f} |")
    add("")
    add(f"MRR {mrr:.3f}")
    add("")

    add(f"### 유형별 Recall@{k_main}")
    add("")
    add("| 유형 | Recall | 문항 |")
    add("|---|---|---|")
    for t, (v, c) in by_type(results, k_main).items():
        add(f"| {t} | {v:.3f} | {c} |")
    add("")

    # 못 찾은 정답 조문. 다음에 뭘 고칠지는 여기서 나온다.
    misses = []
    for r in results:
        if not r["gold"]:
            continue
        for g in r["item"]["gold"]:
            key = (g["law_id"], g["article_key"])
            rank = r["rank_of"].get(key)
            if rank is None or rank > k_main:
                found = f"{rank}위" if rank else f"{k_main}위 밖"
                misses.append(f"- `{r['item']['id']}` {g['cite']} — {found} "
                              f"/ 1위는 {r['hits'][0]['cite'] if r['hits'] else '없음'}")
    add(f"### 놓친 정답 조문 ({len(misses)}건)")
    add("")
    lines.extend(misses or ["없음"])
    add("")

    # 거절 임계값. gold 가 빈 문항과 있는 문항의 1위 점수를 나란히 둔다.
    add("### 거절 임계값 참고")
    add("")
    add("| 문항 | 기대 | 1위 점수 | 1위 조문 |")
    add("|---|---|---|---|")
    for r in results:
        if r["gold"] and not show_scores:
            continue
        top = r["hits"][0]["cite"] if r["hits"] else "없음"
        add(f"| {r['item']['id']} | {r['item']['expect']} | {r['top_score']:.2f} | {top} |")
    add("")
    add("> BM25 점수는 질의마다 범위가 달라 절대값으로 임계값을 걸 수 없다. "
        "하이브리드에서 질의별 정규화를 한 뒤 다시 본다.")
    add("")
    return "\n".join(lines)


def render_compare(runs: dict, label: str, ks, k_main: int) -> str:
    """원문 질의와 자연어 질의를 나란히 놓는다. 이 표가 개선의 근거가 된다."""
    lines = [f"## {label} — 질의 두 벌 비교", "",
             f"측정 시각 {datetime.now():%Y-%m-%d %H:%M}", ""]
    add = lines.append

    names = {"question": "원문(조문 어휘)", "question_nl": "자연어"}
    cols = list(runs)

    add("| 지표 | " + " | ".join(names.get(c, c) for c in cols) + " | 차이 |")
    add("|---|" + "---|" * (len(cols) + 1))
    for k in ks:
        vals = [metrics(runs[c], [k])[0][k]["recall"] for c in cols]
        gap = vals[-1] - vals[0] if len(vals) > 1 else 0
        add(f"| Recall@{k} | " + " | ".join(f"{v:.3f}" for v in vals) + f" | {gap:+.3f} |")
    mrrs = [metrics(runs[c], ks)[1] for c in cols]
    add("| MRR | " + " | ".join(f"{v:.3f}" for v in mrrs) +
        f" | {mrrs[-1] - mrrs[0]:+.3f} |")
    add("")

    add(f"### 유형별 Recall@{k_main}")
    add("")
    add("| 유형 | " + " | ".join(names.get(c, c) for c in cols) + " | 차이 |")
    add("|---|" + "---|" * (len(cols) + 1))
    types = sorted({t for c in cols for t in by_type(runs[c], k_main)})
    for t in types:
        vals = [by_type(runs[c], k_main).get(t, (0, 0))[0] for c in cols]
        add(f"| {t} | " + " | ".join(f"{v:.3f}" for v in vals) +
            f" | {vals[-1] - vals[0]:+.3f} |")
    add("")
    add("> 자연어 질의 쪽이 낮을수록 의미 검색이 메울 여지가 크다. "
        "하이브리드·리랭커의 효과는 이 열에서 확인한다.")
    add("")
    return "\n".join(lines)


# ---------------------------------------------------------------- 실행

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--engine", default="bm25")
    p.add_argument("--evalset", default=str(EVALSET))
    p.add_argument("--k", type=int, default=5, help="유형별·놓친 조문 기준 k")
    p.add_argument("--k1", type=float, default=bm25.K1)
    p.add_argument("--b", type=float, default=bm25.B)
    # 기본값을 박아두면 search.py 의 WEIGHT_BM25 를 조용히 덮어쓴다. None 으로 두고 위임한다.
    p.add_argument("--weight", type=float, default=None,
                   help="hybrid/rerank: BM25 가중치 (기본: search.WEIGHT_BM25)")
    p.add_argument("--fusion", default="weighted", choices=["weighted", "rrf"])
    p.add_argument("--candidates", type=int, default=20, help="rerank: 후보 개수")
    p.add_argument("--scores", action="store_true", help="전 문항 1위 점수 출력")
    p.add_argument("--query-field", default="question",
                   help="question 또는 question_nl")
    p.add_argument("--both", action="store_true", help="두 질의를 나란히 비교")
    p.add_argument("--no-write", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    path = ROOT / args.evalset if not args.evalset.startswith("/") else args.evalset
    data = json.loads(open(path, encoding="utf-8").read())
    items = data["items"]

    engine_kw = {"k1": args.k1, "b": args.b}
    if args.engine in ("hybrid", "rerank"):
        from app.rag import search as _search
        weight = args.weight if args.weight is not None else _search.WEIGHT_BM25
        engine_kw |= {"weight": weight, "fusion": args.fusion}
    if args.engine == "rerank":
        engine_kw |= {"candidates": args.candidates}
    search, label = make_engine(args.engine, **engine_kw)

    fields = ["question", "question_nl"] if args.both else [args.query_field]
    for f in fields:
        if any(f not in i for i in items):
            raise SystemExit(
                f"평가셋에 '{f}' 가 없습니다. python -m app.db.add_paraphrase --write 먼저."
            )

    runs = {f: grade(items, search, max(KS), f) for f in fields}

    if args.both:
        body = render_compare(runs, label, KS, args.k)
        for f in fields:
            body += "\n" + render(runs[f], f"{label} · {f}", KS, args.k, args.scores)
    else:
        f = fields[0]
        body = render(runs[f], f"{label} · {f}", KS, args.k, args.scores)

    print(body)

    if args.no_write:
        return
    out = ROOT / (args.out or f"eval/{datetime.now():%Y-%m-%d}_retrieval.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    with open(out, "a", encoding="utf-8") as f:
        if new:
            f.write(f"# 검색 평가 — {datetime.now():%Y-%m-%d}\n\n"
                    f"평가셋 {path.name} / 청크 테이블 기준.\n\n")
        f.write(body + "\n")
    print(f"\n기록: {out}")


if __name__ == "__main__":
    main()