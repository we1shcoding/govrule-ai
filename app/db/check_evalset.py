"""평가셋의 정답 조문이 실제 DB에 있는지 검증한다.

평가셋은 손으로 쓰기 때문에 조문키 오타가 나기 쉽다. 틀린 정답으로
Recall 을 재면 점수가 통째로 의미를 잃으므로, 쓰기 직전에 확인한다.

실행:
    python -m app.db.check_evalset
    python -m app.db.check_evalset eval/2026-10-27_evalset.json
"""
import json
import sys
from collections import Counter

from app.config import ROOT
from app.db.schema import connect

sys.stdout.reconfigure(errors="replace")

DEFAULT = ROOT / "eval" / "2026-10-27_evalset.json"


def main() -> int:
    path = ROOT / sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    if not path.exists():
        print(f"파일이 없습니다: {path}")
        return 1

    data = json.loads(path.read_text(encoding="utf-8"))
    items = data["items"]
    problems = []

    with connect() as conn:
        for item in items:
            for g in item.get("gold", []):
                row = conn.execute(
                    "SELECT cite, title, is_searchable FROM articles "
                    "WHERE law_id = ? AND article_key = ?",
                    (g["law_id"], g["article_key"]),
                ).fetchone()

                if row is None:
                    problems.append(
                        f"{item['id']}: DB에 없음 — {g['law_id']}/{g['article_key']} ({g['cite']})"
                    )
                elif row["cite"] != g["cite"]:
                    problems.append(
                        f"{item['id']}: 인용 불일치 — 평가셋 '{g['cite']}' / DB '{row['cite']}'"
                    )
                elif not row["is_searchable"]:
                    problems.append(
                        f"{item['id']}: 검색 대상이 아님 — {g['cite']} (삭제 조문)"
                    )

    kinds = Counter(i["type"] for i in items)
    expects = Counter(i["expect"] for i in items)
    golds = sum(len(i.get("gold", [])) for i in items)

    print(f"평가셋: {path.name}")
    print(f"  문항 {len(items)}개 / 정답 조문 {golds}개")
    print(f"  유형: {dict(sorted(kinds.items()))}")
    print(f"  기대: {dict(sorted(expects.items()))}")

    ids = [i["id"] for i in items]
    dup = [k for k, n in Counter(ids).items() if n > 1]
    if dup:
        problems.append(f"id 중복: {dup}")

    print()
    if problems:
        print(f"문제 {len(problems)}건")
        for p in problems:
            print(f"  - {p}")
        return 1

    print("정답 조문이 전부 DB에 있고 인용 표기도 일치합니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
