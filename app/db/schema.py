"""SQLite 스키마.

조문 단위로 저장한다. 청킹·검색은 이 테이블을 읽어서 한다.

설계 메모
- 조문키는 법령 안에서만 유일하다. 세 법령 모두 "0001001" 을 가진다.
  따라서 기본키는 (law_id, 조문키) 복합키다.
- 인용 표기는 조문번호·가지번호로 조립한다. 조문키의 뒷자리와
  가지번호의 대응 관계는 추정이므로 근거로 쓰지 않는다.
- raw 컬럼에 원본 조문 JSON 을 그대로 둔다. 나중에 항 단위로 다시
  자를 때 원본을 다시 수집하지 않아도 된다.

실행:
    python -m app.db.schema          # DB 생성 (이미 있으면 그대로 둠)
    python -m app.db.schema --reset  # 지우고 다시 생성
"""
import sqlite3
import sys

from app.config import ROOT

DB_PATH = ROOT / "data" / "govrule.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS laws (
    law_id        TEXT PRIMARY KEY,   -- 법령ID (예: 000695)
    mst           TEXT NOT NULL,      -- 법령일련번호. 원문 URL 조립에 쓴다
    name          TEXT NOT NULL,      -- 법령명 한글
    short_name    TEXT,               -- 법령명약칭
    kind          TEXT,               -- 법/시행령/시행규칙
    enforced_on   TEXT,               -- 시행일자 YYYYMMDD
    promulgated_on TEXT,              -- 공포일자
    promulgation_no TEXT,             -- 공포번호
    collected_on  TEXT NOT NULL       -- 수집일 YYYY-MM-DD
);

CREATE TABLE IF NOT EXISTS articles (
    law_id        TEXT NOT NULL,
    article_key   TEXT NOT NULL,      -- 조문키 (법령 안에서 유일)
    no            INTEGER,            -- 조문번호
    branch_no     INTEGER,            -- 조문가지번호. 없으면 NULL
    title         TEXT,               -- 조문제목
    chapter       TEXT,               -- 직전 '전문' 행에서 끌어온 장 제목
    body          TEXT NOT NULL,      -- 조문내용 + 항 + 호 + 목 을 이어붙인 전문
    cite          TEXT NOT NULL,      -- "국가계약법 제4조의2" 형태
    enforced_on   TEXT,
    is_deleted    INTEGER NOT NULL DEFAULT 0,
    is_searchable INTEGER NOT NULL DEFAULT 1,  -- 삭제 조문·빈 조문은 0
    seq           INTEGER NOT NULL,   -- 원본 등장 순서
    raw           TEXT NOT NULL,      -- 원본 조문 JSON
    PRIMARY KEY (law_id, article_key),
    FOREIGN KEY (law_id) REFERENCES laws(law_id)
);

CREATE INDEX IF NOT EXISTS idx_articles_law      ON articles(law_id, seq);
CREATE INDEX IF NOT EXISTS idx_articles_no       ON articles(law_id, no, branch_no);
CREATE INDEX IF NOT EXISTS idx_articles_search   ON articles(is_searchable);
"""


def connect() -> sqlite3.Connection:
    """DB 연결을 연다. 폴더가 없으면 만든다."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def create() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)


def reset() -> None:
    with connect() as conn:
        conn.executescript("DROP TABLE IF EXISTS articles; DROP TABLE IF EXISTS laws;")
        conn.executescript(SCHEMA)


if __name__ == "__main__":
    if "--reset" in sys.argv:
        reset()
        print(f"초기화 완료: {DB_PATH}")
    else:
        create()
        print(f"생성 완료: {DB_PATH}")
