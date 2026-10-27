"""수집한 원본 JSON 을 조문 단위로 파싱해 SQLite 에 넣는다.

inspect_raw 로 확인한 함정 6건을 전부 처리한다.
  1. 항이 1개면 dict, 2개 이상이면 list  → as_list 로 정규화
  2. dict 인 항에는 항번호·항내용이 없다   → get 으로 접근, 없으면 건너뜀
  3. 조문가지번호 (제4조의2)              → 인용 표기에 반영
  4. 조문여부='전문' 은 장 제목           → 조문으로 저장하지 않고 맥락으로 보존
  5. 조문내용의 의미가 두 가지            → 항이 있으면 제목, 없으면 본문
  6. 삭제 조문                            → is_searchable=0

실행:
    python -m app.db.schema
    python -m app.collect.parse_law
"""
import json
import re
import sys
from collections import Counter

from app.config import DATA_RAW
from app.db.schema import connect

sys.stdout.reconfigure(errors="replace")

DELETED = re.compile(r"^제\d+조(의\d+)?\s*삭제")
WS = re.compile(r"[ \t]+")


def as_list(v) -> list:
    """dict 하나든 list 든 목이 없든, 항상 list 로 돌려준다."""
    if v is None:
        return []
    if isinstance(v, dict):
        return [v]
    if isinstance(v, list):
        return v
    return []


def clean(v) -> str:
    """내용 필드를 문자열로 만든다.

    목내용·호내용은 여러 줄일 때 문자열이 아니라 문자열 리스트로 온다.
    (함정 7) 숫자로 오는 경우도 있어 str 이 아니면 변환한다.
    """
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        inner = [clean(x) for x in v]
        return "\n".join(p for p in inner if p)
    if not isinstance(v, str):
        v = str(v)
    return WS.sub(" ", v).strip()


def build_body(unit: dict) -> str:
    """조문내용 + 항 + 호 + 목 을 읽는 순서대로 이어붙인다.

    항내용은 이미 '①' 를, 호내용은 '1. ' 를 포함하므로 번호를 덧붙이지 않는다.
    """
    parts = [clean(unit.get("조문내용"))]

    def walk_ho(container: dict) -> None:
        for ho in as_list(container.get("호")):
            parts.append(clean(ho.get("호내용")))
            for mok in as_list(ho.get("목")):
                parts.append(clean(mok.get("목내용")))

    walk_ho(unit)  # 항 없이 호가 바로 붙는 경우 대비
    for hang in as_list(unit.get("항")):
        parts.append(clean(hang.get("항내용")))
        walk_ho(hang)

    return "\n".join(p for p in parts if p)


def make_cite(short_name: str, no: int | None, branch: int | None) -> str:
    if no is None:
        return short_name
    if branch:
        return f"{short_name} 제{no}조의{branch}"
    return f"{short_name} 제{no}조"


def to_int(v) -> int | None:
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


def parse_file(path) -> tuple[dict, list[dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    law = data["법령"]
    info = law["기본정보"]

    name = info["법령명_한글"]
    short = info.get("법령명약칭") or name
    short = short.replace(" ", "")  # "국가계약법 시행령" → "국가계약법시행령"

    if "시행규칙" in name:
        kind = "시행규칙"
    elif "시행령" in name:
        kind = "시행령"
    else:
        kind = "법"

    meta = {
        "law_id": info["법령ID"],
        "mst": str(law.get("법령키", "")),
        "name": name,
        "short_name": short,
        "kind": kind,
        "enforced_on": str(info.get("시행일자", "")),
        "promulgated_on": str(info.get("공포일자", "")),
        "promulgation_no": str(info.get("공포번호", "")),
        "collected_on": path.name[:10],
    }

    rows = []
    chapter = None
    for seq, unit in enumerate(law["조문"]["조문단위"]):
        content = clean(unit.get("조문내용"))

        # 4. 장 제목 행은 조문이 아니다. 맥락으로만 기억한다.
        if unit.get("조문여부") != "조문":
            if content:
                chapter = content
            continue

        no = to_int(unit.get("조문번호"))
        branch = to_int(unit.get("조문가지번호"))
        body = build_body(unit)
        deleted = bool(DELETED.match(content))

        rows.append({
            "law_id": meta["law_id"],
            "article_key": unit["조문키"],
            "no": no,
            "branch_no": branch,
            "title": clean(unit.get("조문제목")) or None,
            "chapter": chapter,
            "body": body,
            "cite": make_cite(short, no, branch),
            "enforced_on": str(unit.get("조문시행일자", "")),
            "is_deleted": int(deleted),
            "is_searchable": int(not deleted and len(body) > 0),
            "seq": seq,
            "raw": json.dumps(unit, ensure_ascii=False),
        })

    return meta, rows


def save(meta: dict, rows: list[dict]) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO laws "
            "(law_id, mst, name, short_name, kind, enforced_on, "
            " promulgated_on, promulgation_no, collected_on) "
            "VALUES (:law_id, :mst, :name, :short_name, :kind, :enforced_on, "
            "        :promulgated_on, :promulgation_no, :collected_on)",
            meta,
        )
        conn.execute("DELETE FROM articles WHERE law_id = ?", (meta["law_id"],))
        conn.executemany(
            "INSERT INTO articles "
            "(law_id, article_key, no, branch_no, title, chapter, body, cite, "
            " enforced_on, is_deleted, is_searchable, seq, raw) "
            "VALUES (:law_id, :article_key, :no, :branch_no, :title, :chapter, "
            "        :body, :cite, :enforced_on, :is_deleted, :is_searchable, "
            "        :seq, :raw)",
            rows,
        )


def report(meta: dict, rows: list[dict]) -> None:
    print(f"\n=== {meta['name']} ===")
    print(f"  조문 {len(rows)}개 저장 (시행 {meta['enforced_on']})")

    deleted = [r for r in rows if r["is_deleted"]]
    empty = [r for r in rows if not r["body"]]
    no_chap = [r for r in rows if not r["chapter"]]
    branches = [r for r in rows if r["branch_no"]]

    print(f"  삭제 조문   : {len(deleted)}")
    print(f"  본문 없음   : {len(empty)}")
    print(f"  장 미상     : {len(no_chap)}")
    print(f"  가지 조문   : {len(branches)}")

    if branches:
        print("  --- 가지조문 샘플 (조문키와 번호 대응 확인용) ---")
        for r in branches[:5]:
            print(f"    {r['article_key']}  번호={r['no']} 가지={r['branch_no']}  {r['cite']}")

    lens = [len(r["body"]) for r in rows if r["body"]]
    if lens:
        lens.sort()
        print(f"  본문 길이   : 중앙 {lens[len(lens)//2]}자 / 최대 {lens[-1]}자")

    longest = max(rows, key=lambda r: len(r["body"]))
    print(f"  최장 조문   : {longest['cite']} ({len(longest['body'])}자)")

    sample = next((r for r in rows if r["chapter"] and r["branch_no"]), None) or rows[1]
    print("  --- 적재 샘플 ---")
    print(f"    cite    : {sample['cite']}")
    print(f"    title   : {sample['title']}")
    print(f"    chapter : {sample['chapter']}")
    body = sample["body"]
    print(f"    body    : {body[:120]}{'...' if len(body) > 120 else ''}")


def main() -> int:
    files = sorted(DATA_RAW.glob("*.json"))
    if not files:
        print(f"{DATA_RAW} 에 JSON 이 없습니다. fetch_law 를 먼저 실행하십시오.")
        return 1

    total = 0
    keys = Counter()
    for path in files:
        meta, rows = parse_file(path)
        save(meta, rows)
        report(meta, rows)
        total += len(rows)
        for r in rows:
            keys[(r["law_id"], r["article_key"])] += 1

    dup = [k for k, n in keys.items() if n > 1]
    print(f"\n{'=' * 50}")
    print(f"총 {total}개 조문 적재")
    print(f"키 중복: {len(dup)}건" + ("  ← 확인 필요" if dup else ""))

    with connect() as conn:
        n = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        s = conn.execute(
            "SELECT COUNT(*) FROM articles WHERE is_searchable = 1"
        ).fetchone()[0]
    print(f"DB 확인: articles {n}행, 검색 대상 {s}행")
    return 0 if n == total and not dup else 1


if __name__ == "__main__":
    raise SystemExit(main())