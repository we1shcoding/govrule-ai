"""리랭커. bge-reranker-v2-m3 로 하이브리드 후보를 다시 줄 세운다.

구조
- 하이브리드(BM25 0.3 / 벡터 0.7)로 상위 CANDIDATES 개를 뽑고,
  크로스 인코더가 질의와 청크를 같이 읽어 관련성 점수를 매긴다.
- 바이 인코더(KURE-v1)는 질의와 문서를 따로 벡터로 만들어 비교하지만,
  크로스 인코더는 둘을 한 번에 읽는다. 느린 대신 정확하다.
  339개 전체에 돌릴 수는 없으니 후보 20개로 좁힌 뒤 쓴다.

VRAM (RTX 3070 8GB)
- bge-reranker-v2-m3 는 bge-m3 계열 568M 이다. fp32 2.3GB / fp16 1.2GB.
  sentence-transformers 기본이 fp32 라 그냥 두면 govrule-4b(3.4GB) 와
  합쳐 위태롭다. fp16 으로 올려 3.4 + 1.2 = 4.6GB 로 맞춘다.
- 임베더는 질의 인코딩만 하므로 CPU 에 둔다(embed.py 참조).
  결과적으로 GPU 에는 LLM 과 리랭커만 올라간다.

거절 판단
- 2026-10-28 측정에서 BM25 점수는 질의마다 범위가 다르고(4~44),
  코사인은 거절해야 할 질의가 정답보다 높게 나왔다(0.63~0.67 vs 0.573).
  둘 다 임계값으로 못 쓴다.
- 크로스 인코더 점수는 질의와 문서를 같이 읽고 내는 관련성 로짓이라
  질의 간 비교가 가능하다. 거절 임계값의 유일한 후보다.
  이 파일은 점수를 싣기만 하고, 임계값은 측정 결과를 보고 정한다.

실행:
    python -m app.rag.rerank "짜고 친 업체는 몇 년 동안 입찰 못 들어오나요?"
    python -m app.rag.rerank "지체상금률" --candidates 30
"""
import argparse

import numpy as np

from app.db.schema import connect
from app.rag import search

MODEL_NAME = "BAAI/bge-reranker-v2-m3"
CANDIDATES = 20      # 하이브리드에서 리랭커로 넘길 개수
TOP_K = 5
MAX_LENGTH = 1024    # 최대 청크가 1,482자다. 8192 를 열 이유가 없다.


def load_model(device: str = "cuda", fp16: bool = True):
    from sentence_transformers import CrossEncoder

    kw = {"max_length": MAX_LENGTH, "device": device}
    if fp16 and device != "cpu":
        import torch
        kw["model_kwargs"] = {"torch_dtype": torch.float16}
    return CrossEncoder(MODEL_NAME, **kw)


class RerankSearch:
    """하이브리드 → 리랭커. evaluate.py 가 쓰는 search(query, top_k) 를 그대로 맞춘다."""

    def __init__(self, candidates: int = CANDIDATES, device: str = "cuda",
                 weight: float = 0.3, fusion: str = "weighted", **kw):
        self.candidates = candidates
        self.device = device
        self.base = search.load(weight=weight, fusion=fusion, **kw)
        self._model = None
        self._texts: dict[str, str] | None = None

    @property
    def model(self):
        if self._model is None:
            self._model = load_model(self.device)
        return self._model

    @property
    def texts(self) -> dict[str, str]:
        """청크 본문. 리랭커는 원문을 읽어야 하므로 전부 메모리에 올린다."""
        if self._texts is None:
            conn = connect()
            self._texts = {r["chunk_id"]: r["text"]
                           for r in conn.execute("SELECT chunk_id, text FROM chunks")}
            conn.close()
        return self._texts

    def label(self) -> str:
        return f"rerank({MODEL_NAME}, cand={self.candidates}, base={self.base.label()})"

    def search(self, query: str, top_k: int = TOP_K) -> list[dict]:
        cands = self.base.search(query, self.candidates)
        if not cands:
            return []

        pairs = [(query, self.texts.get(c["chunk_id"], "")) for c in cands]
        scores = np.asarray(self.model.predict(pairs), dtype=np.float32)

        order = np.argsort(-scores)[:top_k]
        return [
            dict(cands[i],
                 score=float(scores[i]),
                 hybrid_score=cands[i]["score"],
                 hybrid_rank=cands[i]["rank"],
                 rank=r + 1)
            for r, i in enumerate(order)
        ]

    def context(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """LLM 에 넘길 근거. 문자 예산으로 자른다."""
        hits = self.search(query, top_k)
        out, used = [], 0
        for h in hits:
            text = self.texts.get(h["chunk_id"], "")
            if out and used + len(text) > self.base.char_budget:
                break
            out.append(dict(h, text=text))
            used += len(text)
        return out


def load(**kw) -> RerankSearch:
    return RerankSearch(**kw)


# ---------------------------------------------------------------- 실행

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("query")
    p.add_argument("--candidates", type=int, default=CANDIDATES)
    p.add_argument("--top", type=int, default=TOP_K)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    rs = load(candidates=args.candidates, device=args.device)
    print(f"{rs.label()}\n[{args.query}]")
    for h in rs.search(args.query, args.top):
        print(f"  {h['rank']}. {h['score']:8.4f}  "
              f"(하이브리드 {h['hybrid_rank']:>2}위)  {h['cite']}")


if __name__ == "__main__":
    main()