"""DB에 적재된 조문을 텍스트 파일로 덤프한다.

평가셋을 만들 때 실제 조문을 보고 질문·정답을 짝짓기 위한 용도다.
검색 대상(is_searchable=1) 조문만 내보낸다.

실행:
    python -m app.db.dump_articles
    → eval/<오늘날짜>_articles.txt
"""
import sys
from datetime import date

from app.config import ROOT
from app.db.schema import connect

sys.stdout.reconfigure(errors="replace")

OUT = ROOT / "eval" / f"{date.today().isoformat()}_articles.txt"


def main() -> int:
    with connect() as conn:
        laws = conn.execute(
            "SELECT law_id, name, short_name, kind, enforced_on FROM laws "
            "ORDER BY CASE kind WHEN '법' THEN 1 WHEN '시행령' THEN 2 ELSE 3 END"
        ).fetchall()

        OUT.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        total = 0

        for law in laws:
            rows = conn.execute(
                "SELECT article_key, cite, title, chapter, body "
                "FROM articles WHERE law_id = ? AND is_searchable = 1 "
                "ORDER BY seq",
                (law["law_id"],),
            ).fetchall()

            lines.append("=" * 70)
            lines.append(f"{law['name']}  (시행 {law['enforced_on']})")
            lines.append(f"조문 {len(rows)}개")
            lines.append("=" * 70)
            lines.append("")

            for r in rows:
                head = f"[{r['article_key']}] {r['cite']}"
                if r["title"]:
                    head += f" ({r['title']})"
                if r["chapter"]:
                    head += f"  |  {r['chapter']}"
                lines.append(head)
                lines.append(r["body"])
                lines.append("")

            total += len(rows)

        OUT.write_text("\n".join(lines), encoding="utf-8")

    size_kb = OUT.stat().st_size / 1024
    print(f"{OUT}")
    print(f"조문 {total}개 / {size_kb:,.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
