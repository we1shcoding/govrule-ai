"""BM25 검색. 직접 구현.

Kiwi 를 LGPL 때문에 배제했으므로 토크나이저도 직접 만든다. 외부 형태소
분석기 없이 법령 텍스트를 다루기 위한 설계는 아래와 같다.

토크나이저
- 법령 참조(제26조, 제4조의2, 별표 2)는 통째로 한 토큰이다.
  음절 n-gram 으로 쪼개면 "제26조" 와 "제2조" 가 "제2" 를 공유해서 섞인다.
  법령 질의는 조문 번호를 직접 대는 경우가 많아 여기서 바로 틀린다.
- 나머지 한글·한자는 음절 2-gram 으로 자른다. 법령 용어는 한자어
  복합명사 위주라("입찰참가자격", "지체상금률") 어절만 자르면
  "입찰참가자격" 과 "입찰" 이 안 걸린다.
- 어절과 n-gram 을 한 인덱스에 섞지 않는다. 한 청크에서 어절은 수십 개,
  n-gram 은 수백 개라 그냥 합치면 점수가 n-gram 으로만 결정된다.
  그래서 n-gram 을 본류로 두고, 변별력이 필요한 것만 특수 토큰으로 뺐다.
- <개정 2019. 9. 17.>, [전문개정 …], ①②③ 같은 표기는 버린다.
  「」 는 괄호만 버리고 안의 법령명은 남긴다.

머리글 가중 (HEADER_PREFIX)
- 청크 머리글(법령명·조·조문제목·장)의 토큰을 "H:" 를 붙여 한 번 더 넣는다.
  질의할 때도 두 형태를 다 던진다.
- 가중치를 손으로 정하지 않는 게 요점이다. "제26조" 는 그 조문을 참조만 하는
  수십 개 청크에도 들어 있어 idf 가 낮지만, "H:제26조" 는 자기가 제26조인
  청크에만 있어 df 가 작고 idf 가 크다. 자기 조문과 단순 참조가 자동으로 갈린다.
- 조문제목도 머리글이라 같이 가중된다. "수의계약을 할 수 있는 경우" 질의가
  조문제목 "수의계약에 의할 수 있는 경우" 에 걸리는 경로가 이것이다.

색인
- 339개 청크면 메모리에 올려도 수 MB다. 파일로 떨어뜨리지 않고
  기동할 때 DB 에서 읽어 만든다. 1초 안 걸린다.

실행:
    python -m app.rag.bm25 --check          # 검증 질의
    python -m app.rag.bm25 "지체상금률"      # 임의 질의
    python -m app.rag.bm25 "지체상금률" --b 0.3
"""
import argparse
import math
import re
import sqlite3
import unicodedata
from collections import Counter

import numpy as np

from app.db.schema import connect

K1 = 1.2
B = 0.75

# ---------------------------------------------------------------- 토크나이저

# 통째로 보존할 법령 참조. 긴 것부터 봐야 "제4조의2" 가 "제4조" 로 먹히지 않는다.
REF_PATTERNS = [
    r"제\s*\d+\s*조\s*의\s*\d+",      # 제4조의2
    r"제\s*\d+\s*조",
    r"제\s*\d+\s*항",
    r"제\s*\d+\s*호",
    r"별표\s*\d+",
    r"별지\s*제?\s*\d+\s*호?\s*서식",
]
RE_REF = re.compile("|".join(f"(?:{p})" for p in REF_PATTERNS))

# 버릴 표기
RE_REVISION = re.compile(r"<[^>]*>")            # <개정 2019. 9. 17.>
RE_BRACKET_NOTE = re.compile(r"\[[^\]]*\]")     # [전문개정 2010. 7. 21.]
RE_CIRCLED = re.compile(r"[\u2460-\u2473]")     # ① ~ ⑳
RE_QUOTE_MARK = re.compile(r"[「」『』]")        # 괄호만 버리고 안은 남긴다

RE_LATIN = re.compile(r"[A-Za-z][A-Za-z0-9_\-]*")
RE_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
RE_HANGUL = re.compile(r"[가-힣\u4e00-\u9fff]+")

NGRAM = 2

# 머리글에서 나온 토큰에 붙이는 이름공간. 본문의 같은 단어와 구별된다.
HEADER_PREFIX = "H:"


def normalize_ref(s: str) -> str:
    """"제 26 조" → "제26조". 공백만 지운다."""
    return re.sub(r"\s+", "", s)


def tokenize(text: str) -> list[str]:
    """텍스트 → 토큰 리스트. 색인과 질의가 같은 함수를 쓴다."""
    text = unicodedata.normalize("NFC", text)
    text = RE_REVISION.sub(" ", text)
    text = RE_BRACKET_NOTE.sub(" ", text)
    text = RE_CIRCLED.sub(" ", text)
    text = RE_QUOTE_MARK.sub(" ", text)

    tokens: list[str] = []

    # 1. 법령 참조를 먼저 떼어낸다. 뗀 자리는 공백으로 비워 n-gram 에 안 섞이게 한다.
    def take_ref(m: re.Match) -> str:
        tokens.append(normalize_ref(m.group(0)))
        return " "

    text = RE_REF.sub(take_ref, text)

    # 2. 영문·숫자
    for m in RE_LATIN.finditer(text):
        tokens.append(m.group(0).lower())
    text = RE_LATIN.sub(" ", text)
    for m in RE_NUMBER.finditer(text):
        tokens.append(m.group(0).replace(",", ""))

    # 3. 남은 한글·한자 덩어리를 음절 2-gram 으로
    for m in RE_HANGUL.finditer(text):
        run = m.group(0)
        if len(run) < NGRAM:
            tokens.append(run)
            continue
        tokens += [run[i:i + NGRAM] for i in range(len(run) - NGRAM + 1)]

    return tokens


# ---------------------------------------------------------------- 색인

class BM25:
    def __init__(self, k1: float = K1, b: float = B):
        self.k1, self.b = k1, b
        self.ids: list[str] = []          # chunk_id
        self.meta: list[dict] = []        # cite, article_key, law_id, n_chars
        self.lens: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = {}  # term → [(doc, tf)]
        self.avgdl = 0.0

    def build(self, rows) -> "BM25":
        df: dict[str, list[tuple[int, int]]] = {}
        for i, r in enumerate(rows):
            self.ids.append(r["chunk_id"])
            self.meta.append({
                "cite": r["cite"], "law_id": r["law_id"],
                "article_key": r["article_key"], "n_chars": r["n_chars"],
            })
            tf = Counter(tokenize(r["text"]))
            # 머리글은 text 안에 이미 들어 있다. H: 형태로 한 번 더 넣어 가중한다.
            tf.update(HEADER_PREFIX + t for t in tokenize(r["header"]))
            self.lens.append(sum(tf.values()))
            for term, n in tf.items():
                df.setdefault(term, []).append((i, n))
        self.postings = df
        self.avgdl = sum(self.lens) / max(len(self.lens), 1)
        return self

    def idf(self, term: str) -> float:
        n = len(self.postings.get(term, ()))
        if n == 0:
            return 0.0
        N = len(self.ids)
        return math.log(1 + (N - n + 0.5) / (n + 0.5))

    def query_terms(self, query: str) -> set[str]:
        plain = tokenize(query)
        return set(plain) | {HEADER_PREFIX + t for t in plain}

    def score_all(self, query: str) -> np.ndarray:
        """전 청크 점수. 하이브리드 결합은 이 배열을 정규화해서 쓴다.

        339개뿐이라 후보 풀을 따로 뽑지 않고 전체를 민다. 후보 풀에서
        정규화하면 풀 경계에 따라 점수가 달라져 비교가 흔들린다.
        """
        scores = np.zeros(len(self.ids), dtype=np.float32)
        for term in self.query_terms(query):
            posting = self.postings.get(term)
            if not posting:
                continue
            idf = self.idf(term)
            for doc, tf in posting:
                norm = 1 - self.b + self.b * self.lens[doc] / self.avgdl
                scores[doc] += idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)
        return scores

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        scores = {i: float(s) for i, s in enumerate(self.score_all(query)) if s > 0}
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        return [
            dict(self.meta[d], chunk_id=self.ids[d], score=s, rank=i + 1)
            for i, (d, s) in enumerate(ranked)
        ]


def load(conn: sqlite3.Connection | None = None, **kw) -> BM25:
    """chunks 전체로 색인을 만든다."""
    own = conn is None
    conn = conn or connect()
    rows = conn.execute(
        "SELECT chunk_id, law_id, article_key, cite, header, text, n_chars "
        "FROM chunks ORDER BY law_id, seq"
    ).fetchall()
    if own:
        conn.close()
    return BM25(**kw).build(rows)


# ---------------------------------------------------------------- 실행

# 토크나이저가 제대로 동작하는지 보는 최소 질의.
# 둘 다 1위가 안 나오면 토크나이저가 틀린 것이지 BM25 가 틀린 게 아니다.
CHECKS = [
    ("지체상금률", "한자어 복합명사가 n-gram 으로 걸리는지"),
    ("제26조", "조문 번호가 통째로 걸리는지. 제2조·제6조가 올라오면 실패"),
    ("수의계약을 할 수 있는 경우", "자연어 질의"),
    ("입찰참가자격 제한", "복합명사 + 일반어"),
]


def show(bm: BM25, query: str, top_k: int) -> None:
    print(f"\n[{query}]")
    toks = tokenize(query)
    print(f"  토큰 {len(toks)}개: {' '.join(toks[:18])}{' …' if len(toks) > 18 else ''}")
    hits = bm.search(query, top_k)
    if not hits:
        print("  결과 없음")
        return
    for h in hits:
        print(f"  {h['rank']}. {h['score']:7.3f}  {h['cite']}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("query", nargs="?", help="검색어")
    p.add_argument("--check", action="store_true", help="검증 질의 실행")
    p.add_argument("--top", type=int, default=5)
    p.add_argument("--k1", type=float, default=K1)
    p.add_argument("--b", type=float, default=B)
    args = p.parse_args()

    bm = load(k1=args.k1, b=args.b)
    print(f"색인 완료: 청크 {len(bm.ids)}개 / 고유 토큰 {len(bm.postings)}개 "
          f"/ 평균 길이 {bm.avgdl:.0f}토큰 (k1={args.k1}, b={args.b})")

    if args.check or not args.query:
        for q, why in CHECKS:
            show(bm, q, args.top)
            print(f"     ↳ {why}")
    else:
        show(bm, args.query, args.top)


if __name__ == "__main__":
    main()