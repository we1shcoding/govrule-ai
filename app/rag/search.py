"""하이브리드 검색. BM25 + KURE-v1 결합.

왜 결합하는가 (2026-10-28 측정 기준)
- 원문 질의: BM25 Recall@5 0.972 / 벡터 0.917
- 자연어 질의: BM25 0.333 / 벡터 0.722
  둘이 서로 다른 걸 놓친다. BM25 만 L5-03·L4-02 를 10위 안에서 찾고,
  벡터만 L1 자연어를 전부 찾는다.

결합 방식 두 가지
- weighted  각 점수를 질의 안에서 min-max 정규화한 뒤 가중합.
            BM25 는 질의마다 범위가 달라(4~44) 정규화 없이는 못 더한다.
            코사인은 범위가 좁아(0.45~0.65) 정규화 없이 더하면 묻힌다.
- rrf       순위만 쓴다. 1/(k+rank) 를 더한다. 점수 분포에 둔감해서
            이질적인 검색기를 합칠 때 흔히 더 안정적이다.
  두 방식을 평가셋으로 비교해서 고른다.

점수를 전 청크에 대해 계산한다. 339개뿐이라 후보 풀을 따로 뽑을 이유가 없고,
풀 경계에 따라 정규화 결과가 달라지는 문제도 피한다.

문자 예산
- 상위 k개를 그대로 넘기지 않는다. 청크 길이가 77~1,482자로 들쭉날쭉해서
  k 고정은 컨텍스트(8,192토큰)를 넘길 수 있다. 누적 글자 수로 자른다.

거절에 대하여
- 하이브리드 점수로는 거절을 판단하지 않는다. 2026-10-28 측정에서
  L5-01(없는 조문) 1위 코사인 0.63, L5-02(소관 밖) 0.67 인데
  정상 질의의 정답 조문이 0.573 이었다. 거절해야 할 질의가 더 높다.
  거절은 리랭커 점수와 프롬프트에서 판단한다. 여기서는 점수를 실어 보내기만 한다.

실행:
    python -m app.rag.search "공사가 늦어지면 하루에 얼마씩 물어내나요?"
    python -m app.rag.search "지체상금률" --weight 0.7
    python -m app.rag.search "지체상금률" --fusion rrf
"""
import argparse

import numpy as np

from app.db.schema import connect
from app.rag import bm25, embed

# BM25 가중치. 나머지가 벡터로 간다. 0.3 에서 시작해 평가셋으로 조정한다.
WEIGHT_BM25 = 0.3
# RRF 상수. 60 이 관례다.
RRF_K = 60
# LLM 에 넘길 누적 글자 수 상한.
CHAR_BUDGET = 4500
TOP_K = 5


def minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    return np.zeros_like(x) if hi - lo < 1e-9 else (x - lo) / (hi - lo)


def to_ranks(x: np.ndarray) -> np.ndarray:
    """점수 배열 → 1부터 시작하는 순위 배열."""
    order = np.argsort(-x)
    ranks = np.empty(len(x), dtype=np.int32)
    ranks[order] = np.arange(1, len(x) + 1)
    return ranks


class HybridSearch:
    def __init__(self, weight: float = WEIGHT_BM25, fusion: str = "weighted",
                 char_budget: int = CHAR_BUDGET, **bm_kw):
        self.weight = weight
        self.fusion = fusion
        self.char_budget = char_budget
        self.bm = bm25.load(**bm_kw)
        self.vec = embed.VectorIndex()

        # 두 색인이 같은 순서여야 배열을 그냥 더할 수 있다.
        if self.bm.ids != self.vec.ids:
            raise SystemExit(
                "BM25 와 벡터 색인의 청크 순서가 다릅니다. "
                "chunk.py 를 다시 돌린 뒤 embed.py 도 다시 돌리세요."
            )
        self.ids = self.bm.ids
        self.meta = self.bm.meta

    def label(self) -> str:
        if self.fusion == "rrf":
            return f"hybrid(rrf, k={RRF_K})"
        return f"hybrid(weighted, bm25={self.weight:.2f}, vec={1 - self.weight:.2f})"

    def combine(self, query: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        bm_raw = self.bm.score_all(query)
        vec_raw = self.vec.score_all(query)

        if self.fusion == "rrf":
            combined = (self.weight / (RRF_K + to_ranks(bm_raw))
                        + (1 - self.weight) / (RRF_K + to_ranks(vec_raw)))
        else:
            combined = (self.weight * minmax(bm_raw)
                        + (1 - self.weight) * minmax(vec_raw))
        return combined, bm_raw, vec_raw

    def search(self, query: str, top_k: int = TOP_K) -> list[dict]:
        combined, bm_raw, vec_raw = self.combine(query)
        order = np.argsort(-combined)[:top_k]
        return [
            dict(self.meta[i],
                 chunk_id=self.ids[i],
                 score=float(combined[i]),
                 bm25=float(bm_raw[i]),
                 vector=float(vec_raw[i]),
                 rank=r + 1)
            for r, i in enumerate(order)
        ]

    def context(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """LLM 에 넘길 근거. 본문을 붙이고 문자 예산으로 자른다."""
        hits = self.search(query, top_k)
        if not hits:
            return []

        conn = connect()
        texts = {
            r["chunk_id"]: r["text"]
            for r in conn.execute(
                "SELECT chunk_id, text FROM chunks WHERE chunk_id IN "
                f"({','.join('?' * len(hits))})",
                [h["chunk_id"] for h in hits],
            )
        }
        conn.close()

        out, used = [], 0
        for h in hits:
            text = texts.get(h["chunk_id"], "")
            # 1위는 예산을 넘겨도 넣는다. 빈손으로 보내는 것보다 낫다.
            if out and used + len(text) > self.char_budget:
                break
            out.append(dict(h, text=text))
            used += len(text)
        return out


def load(weight: float = WEIGHT_BM25, fusion: str = "weighted", **kw) -> HybridSearch:
    return HybridSearch(weight=weight, fusion=fusion, **kw)


# ---------------------------------------------------------------- 실행

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("query")
    p.add_argument("--weight", type=float, default=WEIGHT_BM25, help="BM25 가중치")
    p.add_argument("--fusion", default="weighted", choices=["weighted", "rrf"])
    p.add_argument("--top", type=int, default=TOP_K)
    p.add_argument("--context", action="store_true", help="문자 예산 적용 결과")
    args = p.parse_args()

    hs = load(weight=args.weight, fusion=args.fusion)
    print(f"{hs.label()} / 청크 {len(hs.ids)}개\n[{args.query}]")

    if args.context:
        ctx = hs.context(args.query, args.top)
        used = sum(len(c["text"]) for c in ctx)
        print(f"  근거 {len(ctx)}개 / 누적 {used}자 (예산 {hs.char_budget})")
        for c in ctx:
            print(f"  {c['rank']}. {c['score']:.4f}  {c['cite']}  ({len(c['text'])}자)")
        return

    for h in hs.search(args.query, args.top):
        print(f"  {h['rank']}. {h['score']:.4f}  "
              f"(bm25 {h['bm25']:6.2f} / vec {h['vector']:.4f})  {h['cite']}")


if __name__ == "__main__":
    main()