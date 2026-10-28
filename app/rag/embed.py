"""KURE-v1 임베딩 색인.

청크를 벡터로 바꿔 SQLite 에 BLOB 으로 넣고, 질의 시 코사인 유사도로 찾는다.

VRAM 메모 (RTX 3070 8GB)
- KURE-v1 은 bge-m3 기반 568M 파라미터다. fp32 로 올리면 2.3GB,
  fp16 이면 1.2GB. sentence-transformers 기본은 fp32 이므로 그냥 두면
  govrule-4b(3.4GB) + 리랭커(2.3GB) 와 합쳐 8GB 를 넘는다.
- 색인은 GPU fp16 으로 돌린다. 그 전에 `ollama stop govrule-4b` 로 LLM 을 내린다.
- 질의용 인코더는 CPU 로 따로 연다. 질의는 한 문장이라 CPU 로 100ms 가
  안 걸리고, 이러면 서빙 때 임베더가 VRAM 에서 완전히 빠진다.
  3.4GB(LLM) + 1.2GB(리랭커) = 4.6GB 로 여유가 생긴다.

저장
- 벡터는 L2 정규화해서 넣는다. 그러면 코사인 유사도가 내적과 같아져서
  질의 때 나눗셈이 사라진다.
- 339개 × 1024차원 × 4바이트 = 1.4MB. 전부 메모리에 올려 쓴다.
  이 규모에 FAISS 같은 걸 붙일 이유가 없다.

실행:
    ollama stop govrule-4b
    python -m app.rag.embed                 # 색인 (GPU fp16)
    python -m app.rag.embed --device cpu    # VRAM 이 모자랄 때
    python -m app.rag.embed --check         # 색인 없이 검증 질의만
"""
import argparse
import sqlite3
import time

import numpy as np

from app.db.schema import connect

MODEL_NAME = "nlpai-lab/KURE-v1"
DIM = 1024
# 청크 최대가 1,482자다. 8192 를 다 열어둘 이유가 없고, 줄이면 색인이 빨라진다.
MAX_SEQ = 1024

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunk_vectors (
    chunk_id TEXT PRIMARY KEY,
    model    TEXT NOT NULL,
    dim      INTEGER NOT NULL,
    vec      BLOB NOT NULL,      -- float32, L2 정규화됨
    FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id)
);
"""


def load_model(device: str, fp16: bool = True):
    """임베딩 모델을 연다. import 가 무거우니 필요할 때만 한다."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME, device=device)
    model.max_seq_length = MAX_SEQ
    if fp16 and device != "cpu":
        model = model.half()
    return model


def build(device: str, batch_size: int) -> None:
    conn = connect()
    conn.executescript(SCHEMA)

    rows = conn.execute(
        "SELECT chunk_id, text FROM chunks ORDER BY law_id, seq"
    ).fetchall()
    print(f"청크 {len(rows)}개 / 모델 {MODEL_NAME} / device={device}")

    t0 = time.time()
    model = load_model(device)
    print(f"  모델 적재 {time.time() - t0:.1f}초")

    t0 = time.time()
    vecs = model.encode(
        [r["text"] for r in rows],
        batch_size=batch_size,
        normalize_embeddings=True,   # 코사인 = 내적
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    print(f"  인코딩 {time.time() - t0:.1f}초 / shape {vecs.shape}")

    if vecs.shape[1] != DIM:
        raise SystemExit(f"차원이 다릅니다: {vecs.shape[1]} (예상 {DIM})")

    with conn:
        conn.execute("DELETE FROM chunk_vectors")
        conn.executemany(
            "INSERT INTO chunk_vectors (chunk_id, model, dim, vec) VALUES (?,?,?,?)",
            [(r["chunk_id"], MODEL_NAME, DIM, v.tobytes())
             for r, v in zip(rows, vecs)],
        )

    n = conn.execute("SELECT COUNT(*) FROM chunk_vectors").fetchone()[0]
    print(f"적재 완료: 벡터 {n}개 (청크 {len(rows)}개)")
    if n != len(rows):
        print("  [문제] 벡터 수와 청크 수가 다릅니다")
    conn.close()


# ---------------------------------------------------------------- 검색

class VectorIndex:
    """메모리에 올린 벡터 행렬. 질의 인코더는 CPU 로 따로 연다."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._model = None
        conn = connect()
        rows = conn.execute(
            "SELECT v.chunk_id, v.vec, c.law_id, c.article_key, c.cite, c.n_chars "
            "FROM chunk_vectors v JOIN chunks c ON c.chunk_id = v.chunk_id "
            "ORDER BY c.law_id, c.seq"
        ).fetchall()
        conn.close()
        if not rows:
            raise SystemExit("벡터가 없습니다. python -m app.rag.embed 를 먼저 돌리세요.")

        self.ids = [r["chunk_id"] for r in rows]
        self.meta = [{"cite": r["cite"], "law_id": r["law_id"],
                      "article_key": r["article_key"], "n_chars": r["n_chars"]}
                     for r in rows]
        self.matrix = np.vstack(
            [np.frombuffer(r["vec"], dtype=np.float32) for r in rows]
        )

    @property
    def model(self):
        if self._model is None:
            self._model = load_model(self.device, fp16=False)
        return self._model

    def encode_query(self, query: str) -> np.ndarray:
        return self.model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)[0]

    def score_all(self, query: str) -> np.ndarray:
        """전 청크 코사인 유사도. 정규화돼 있어 내적이 곧 코사인이다."""
        return self.matrix @ self.encode_query(query)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        sims = self.score_all(query)
        order = np.argsort(-sims)[:top_k]
        return [
            dict(self.meta[i], chunk_id=self.ids[i], score=float(sims[i]), rank=r + 1)
            for r, i in enumerate(order)
        ]


def load(**_) -> VectorIndex:
    return VectorIndex()


# ---------------------------------------------------------------- 실행

CHECKS = [
    ("공사가 늦어지면 하루에 얼마씩 물어내야 하나요?", "L1-03 자연어. BM25 는 5위 밖이었다"),
    ("뽑혀놓고 계약을 안 하면 미리 걸어둔 돈은 못 돌려받나요?", "L4-02 자연어. BM25 는 전멸"),
    ("지체상금률", "용어 질의. BM25 가 이미 잘하던 것"),
    ("제26조", "조문 번호. 임베딩이 약할 수밖에 없는 질의"),
]


def check(top_k: int) -> None:
    idx = VectorIndex()
    print(f"벡터 {len(idx.ids)}개 / {idx.matrix.shape}")
    for q, why in CHECKS:
        print(f"\n[{q}]")
        for h in idx.search(q, top_k):
            print(f"  {h['rank']}. {h['score']:.4f}  {h['cite']}")
        print(f"     ↳ {why}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true", help="색인 없이 검증 질의만")
    p.add_argument("--device", default="cuda", help="cuda 또는 cpu")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--top", type=int, default=5)
    args = p.parse_args()

    if args.check:
        check(args.top)
    else:
        build(args.device, args.batch_size)


if __name__ == "__main__":
    main()