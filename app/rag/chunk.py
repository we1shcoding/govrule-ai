"""조문 → 검색 청크.

articles 에서 is_searchable=1 인 조문을 읽어 검색 단위로 자른다.

자르는 규칙
- 조문 전체가 SOFT_MAX 이하면 통째로 한 청크다.
- 넘으면 항 단위로 내려가서, 연속한 항을 SOFT_MAX 까지 묶는다.
- 항 하나가 HARD_MAX 를 넘으면 그 항의 호로, 호가 넘으면 목으로 내려간다.
- 내려갈 때마다 상위 머리글(항내용·호내용)을 각 묶음 앞에 다시 붙인다.
  묶음만 검색에 걸려도 무엇에 대한 규정인지 보여야 근거로 쓸 수 있다.

설계 메모
- 항·호·목 분할은 articles.raw 의 원본 JSON 을 다시 읽어서 한다.
  body 를 정규식으로 되자르지 않는다. 원본이 그대로 있는데 되자르면
  parse_law 에서 처리한 파싱 함정(항이 dict/list, 목내용이 list[str] 등)을
  다시 밟게 된다.
- 자르지 않는 조문은 body 를 그대로 쓴다. parse_law 가 조립한 결과를
  신뢰하고 같은 일을 두 번 하지 않는다.
- 연속한 짧은 항을 묶는 이유는 BM25 때문이다. 길이 정규화(b=0.75)가
  짧은 문서에 점수를 몰아줘서, 65자짜리 위임 조항이 실질 조문보다
  위로 올라온다. 항 하나 = 청크 하나로 두면 이게 수십 건 생긴다.
- 모든 청크는 (law_id, article_key) 로 조문을 역추적할 수 있어야 한다.
  출처 표시 때문만이 아니라, 평가셋 정답이 조문 단위라 채점이 불가능해진다.

실행:
    python -m app.rag.chunk            # 청킹 (chunks 를 비우고 다시 만든다)
    python -m app.rag.chunk --check    # 적재하지 않고 검증만
"""
import argparse
import json
import sqlite3

from app.db.schema import connect

# 조문·묶음이 이 길이를 넘으면 더 잘게 간다.
SOFT_MAX = 900
# 단위 하나가 이 길이를 넘으면 한 단계 아래로 내려간다.
HARD_MAX = 1200

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    TEXT PRIMARY KEY,   -- "{law_id}:{article_key}:{nn}"
    law_id      TEXT NOT NULL,
    article_key TEXT NOT NULL,
    para_no     TEXT,               -- 항번호. 조문 전체 청크면 NULL
    cite        TEXT NOT NULL,      -- "국가계약법시행령 제26조제1항제5호가목~다목"
    header      TEXT NOT NULL,      -- 청크 맨 앞에 붙인 맥락 한 줄
    text        TEXT NOT NULL,      -- header + 본문. 색인 대상은 이 컬럼이다
    n_chars     INTEGER NOT NULL,   -- len(text). search.py 의 문자 예산이 쓴다
    seq         INTEGER NOT NULL,   -- 법령 안에서의 순서
    FOREIGN KEY (law_id, article_key) REFERENCES articles(law_id, article_key)
);

CREATE INDEX IF NOT EXISTS idx_chunks_article ON chunks(law_id, article_key);
CREATE INDEX IF NOT EXISTS idx_chunks_law     ON chunks(law_id, seq);
"""

# ① ~ ⑳ → "1" ~ "20"
CIRCLED = {chr(0x2460 + i): str(i + 1) for i in range(20)}

# 분할 깊이별 단위 이름. 0=항, 1=호, 2=목
UNIT_NAME = ("항", "호", "목")


# ---------------------------------------------------------------- 원본 JSON

def as_list(v):
    """항·호·목은 1개면 dict, 2개 이상이면 list 로 온다. 리스트로 맞춘다."""
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


def clean(v) -> str:
    """목내용처럼 str 이 아니라 list[str] 로 오는 값이 있다. 재귀로 펴서 잇는다."""
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, list):
        return "\n".join(x for x in (clean(i) for i in v) if x)
    return str(v).strip()


def norm_no(s: str) -> str:
    """번호를 표기용 문자열로. "①" → "1", "2." → "2", "가." → "가"."""
    s = (s or "").strip()
    if not s:
        return ""
    if s in CIRCLED:
        return CIRCLED[s]
    return s.rstrip(".)").strip()


def mok_units(raw) -> list[dict]:
    out = []
    for m in as_list(raw):
        if isinstance(m, dict):
            no, head = norm_no(clean(m.get("목번호"))), clean(m.get("목내용"))
        else:
            no, head = "", clean(m)
        if head:
            out.append({"no": no, "head": head, "items": []})
    return out


def ho_units(raw) -> list[dict]:
    out = []
    for h in as_list(raw):
        if isinstance(h, dict):
            no = norm_no(clean(h.get("호번호")))
            head = clean(h.get("호내용"))
            items = mok_units(h.get("목"))
        else:
            no, head, items = "", clean(h), []
        if head or items:
            out.append({"no": no, "head": head, "items": items})
    return out


def hang_units(raw: dict) -> list[dict]:
    """조문을 항 단위로 편다. 항이 없는 조문이면 빈 리스트."""
    out = []
    for p in as_list(raw.get("항")):
        if isinstance(p, dict):
            # 항이 dict 일 때 항번호·항내용 키가 아예 없는 경우가 있다. get 으로 간다.
            no = norm_no(clean(p.get("항번호")))
            head = clean(p.get("항내용"))
            items = ho_units(p.get("호"))
        else:
            no, head, items = "", clean(p), []
        if head or items:
            out.append({"no": no, "head": head, "items": items})
    return out


def full_text(u: dict) -> str:
    """단위 하나의 전문. 머리글 + 하위 단위 전부."""
    parts = [u["head"]] if u["head"] else []
    parts += [full_text(c) for c in u["items"]]
    return "\n".join(p for p in parts if p).strip()


# ---------------------------------------------------------------- 표기

def label(no: str, depth: int) -> str:
    """제1항 / 제5호 / 가목. 목은 "제가목" 이 아니다."""
    if not no:
        return ""
    return f"{no}목" if depth == 2 else f"제{no}{UNIT_NAME[depth]}"


def span_label(nos: list[str], depth: int) -> str:
    if not nos:
        return ""
    first, last = label(nos[0], depth), label(nos[-1], depth)
    return first if first == last else f"{first}~{last}"


def build_header(art: sqlite3.Row, suffix: str) -> str:
    """상위 맥락 한 줄. "국가계약법시행령 제26조(수의계약…) 제1항제5호 / 제3장 …" """
    line = art["cite"]
    if art["title"]:
        line += f"({art['title']})"
    if suffix:
        line += f" {suffix}"
    if art["chapter"]:
        line += f" / {art['chapter']}"
    return line


def make_chunk(art: sqlite3.Row, cite: str, body: str, para_no: str | None) -> dict:
    suffix = cite[len(art["cite"]):]
    header = build_header(art, suffix)
    text = f"{header}\n\n{body}".strip()
    return {"cite": cite, "header": header, "text": text,
            "n_chars": len(text), "para_no": para_no or None}


# ---------------------------------------------------------------- 청크 조립

def emit(art, units, depth, head, cite_prefix, para_no, out) -> None:
    """units 를 SOFT_MAX 까지 묶어 청크로 낸다.

    head        각 묶음 앞에 다시 붙일 상위 머리글 (항내용·호내용)
    cite_prefix 지금까지 조립된 인용. 여기에 묶음의 번호 표기를 덧붙인다
    para_no     이 묶음이 속한 항번호. 깊이와 무관하게 끝까지 들고 간다
    """
    group: list[dict] = []
    glen = 0

    def flush():
        nonlocal glen
        if not group:
            return
        nos = [u["no"] for u in group if u["no"]]
        body = "\n".join(([head] if head else []) + [full_text(u) for u in group])
        out.append(make_chunk(art, cite_prefix + span_label(nos, depth), body,
                              para_no if depth else (nos[0] if nos else None)))
        group.clear()
        glen = 0

    for u in units:
        text = full_text(u)
        # 혼자서 HARD_MAX 를 넘고 더 쪼갤 하위 단위가 있으면 한 단계 내려간다.
        if len(text) > HARD_MAX and u["items"] and depth + 1 < len(UNIT_NAME):
            flush()
            emit(art, u["items"], depth + 1,
                 "\n".join(x for x in (head, u["head"]) if x),
                 cite_prefix + label(u["no"], depth),
                 para_no if depth else (u["no"] or None),
                 out)
            continue
        if group and glen + len(text) > SOFT_MAX:
            flush()
        group.append(u)
        glen += len(text) + (len(head) if not glen else 0)
    flush()


def chunk_article(art: sqlite3.Row) -> tuple[list[dict], str | None]:
    """조문 하나를 청크 리스트로. 두 번째 값은 경고 메시지(없으면 None)."""
    body = (art["body"] or "").strip()
    if not body:
        return [], f"{art['cite']}: body 가 비었다"

    header = build_header(art, "")
    whole = [make_chunk(art, art["cite"], body, None)]

    # 짧으면 통째로. parse_law 가 조립한 body 를 그대로 쓴다.
    if len(body) + len(header) <= SOFT_MAX:
        return whole, None

    try:
        raw = json.loads(art["raw"])
    except (TypeError, ValueError):
        return whole, f"{art['cite']}: raw 가 JSON 이 아니다. 자르지 않고 둔다 ({len(body)}자)"

    units = hang_units(raw)
    if not units:
        return whole, f"{art['cite']}: 항을 못 찾았다. 자르지 않고 둔다 ({len(body)}자)"

    out: list[dict] = []
    emit(art, units, 0, "", art["cite"], None, out)
    return (out, None) if out else (whole, f"{art['cite']}: 분할 결과가 비었다")


# ---------------------------------------------------------------- 실행

def run(write: bool) -> None:
    conn = connect()
    conn.executescript(SCHEMA)

    arts = conn.execute(
        "SELECT * FROM articles WHERE is_searchable = 1 ORDER BY law_id, seq"
    ).fetchall()

    rows, warnings = [], []
    seq_by_law: dict[str, int] = {}

    for art in arts:
        chunks, warn = chunk_article(art)
        if warn:
            warnings.append(warn)
        for i, c in enumerate(chunks):
            law = art["law_id"]
            seq_by_law[law] = seq_by_law.get(law, 0) + 1
            rows.append((
                f"{law}:{art['article_key']}:{i:02d}",
                law, art["article_key"], c["para_no"],
                c["cite"], c["header"], c["text"], c["n_chars"], seq_by_law[law],
            ))

    if write:
        with conn:
            conn.execute("DELETE FROM chunks")
            conn.executemany(
                "INSERT INTO chunks (chunk_id, law_id, article_key, para_no, "
                "cite, header, text, n_chars, seq) VALUES (?,?,?,?,?,?,?,?,?)",
                rows,
            )

    report(arts, rows, warnings, wrote=write)
    conn.close()


def report(arts, rows, warnings, wrote: bool) -> None:
    lens = sorted(r[7] for r in rows)
    covered = {(r[1], r[2]) for r in rows}
    missing = [a["cite"] for a in arts if (a["law_id"], a["article_key"]) not in covered]
    per_art: dict[tuple, int] = {}
    for r in rows:
        per_art[(r[1], r[2])] = per_art.get((r[1], r[2]), 0) + 1

    print(f"{'적재' if wrote else '검증'} 완료")
    print(f"  조문 {len(arts)}개 → 청크 {len(rows)}개")
    print(f"  길이: 최소 {lens[0]} / 25% {lens[len(lens)//4]} / 중앙 {lens[len(lens)//2]} "
          f"/ 최대 {lens[-1]}자")
    print(f"  300자 미만 {sum(1 for n in lens if n < 300)}개 / "
          f"SOFT_MAX({SOFT_MAX}) 초과 {sum(1 for n in lens if n > SOFT_MAX)}개 / "
          f"HARD_MAX({HARD_MAX}) 초과 {sum(1 for n in lens if n > HARD_MAX)}개")
    print(f"  분할된 조문 {sum(1 for v in per_art.values() if v > 1)}개")

    if missing:
        print(f"\n  [문제] 청크가 하나도 안 생긴 조문 {len(missing)}개")
        for c in missing[:10]:
            print(f"    - {c}")
    else:
        print("  역추적: 모든 조문이 청크를 가진다")

    if warnings:
        print(f"\n  [확인 필요] {len(warnings)}건")
        for w in warnings:
            print(f"    - {w}")

    print("\n  샘플 — 시행령 제26조 / 시행규칙 제74조")
    for a in arts:
        if a["no"] in (26, 74) and len(a["body"] or "") > SOFT_MAX:
            mine = [r for r in rows if r[1] == a["law_id"] and r[2] == a["article_key"]]
            print(f"    {a['cite']} ({len(a['body'])}자) → {len(mine)}청크")
            for r in mine:
                print(f"      {r[7]:>5}자  {r[4]}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true", help="적재하지 않고 검증만")
    args = p.parse_args()
    run(write=not args.check)