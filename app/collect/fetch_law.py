"""법제처 OPEN API 수집 — 국가계약법 3종.

목록조회(lawSearch.do)로 법령일련번호(MST)를 얻고,
본문조회(lawService.do)로 조문 전문을 받아 data/raw/ 에 저장한다.

응답 구조는 파싱하지 않는다. 원본 JSON 을 그대로 저장하고
최소한의 구조 요약만 출력한다. 구조를 눈으로 확인한 뒤에
파싱 코드를 쓴다 (2026-10-28 작업).

실행:
    python -m app.collect.fetch_law
"""
import json
import sys
from datetime import date
from pathlib import Path

import requests

from app.config import DATA_RAW, LAW_API_BASE, LAW_API_OC

TIMEOUT = 30

# 법령명은 약칭이 아닌 정식 명칭으로 검색한다.
TARGETS = [
    "국가를 당사자로 하는 계약에 관한 법률",
    "국가를 당사자로 하는 계약에 관한 법률 시행령",
    "국가를 당사자로 하는 계약에 관한 법률 시행규칙",
]

SHORT = {
    "국가를 당사자로 하는 계약에 관한 법률": "국가계약법",
    "국가를 당사자로 하는 계약에 관한 법률 시행령": "국가계약법시행령",
    "국가를 당사자로 하는 계약에 관한 법률 시행규칙": "국가계약법시행규칙",
}


def _get(path: str, params: dict) -> dict:
    """법제처 API 호출. JSON 이 아니면 본문을 보여주고 중단한다."""
    url = f"{LAW_API_BASE}/{path}"
    params = {"OC": LAW_API_OC, "type": "JSON", **params}
    r = requests.get(url, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    r.encoding = "utf-8"
    try:
        return r.json()
    except ValueError:
        # OC 오류·권한 문제는 200 + HTML 로 돌아오는 경우가 있다
        raise RuntimeError(
            f"JSON 이 아닌 응답입니다.\n"
            f"  URL: {r.url}\n"
            f"  앞부분: {r.text[:300]}"
        )


def search_law(name: str) -> dict:
    """법령명으로 현행법령을 검색해 첫 번째 결과를 반환한다."""
    data = _get("lawSearch.do", {"target": "law", "query": name, "display": "20", "search": "1"})

    # 응답 껍데기 이름이 문서와 다를 수 있으므로 방어적으로 찾는다
    root = data.get("LawSearch") or data.get("lawSearch") or data
    items = root.get("law") or []
    if isinstance(items, dict):
        items = [items]
    if not items:
        raise RuntimeError(
            f"검색 결과 없음: {name}\n  응답 최상위 키: {list(data.keys())}"
        )

    # 법령명이 정확히 일치하는 것을 우선한다
    for it in items:
        if (it.get("법령명한글") or "").strip() == name:
            return it
    print(f"  ! 정확히 일치하는 법령명이 없어 첫 결과를 사용합니다", file=sys.stderr)
    return items[0]


def fetch_body(mst: str) -> dict:
    """법령일련번호(MST)로 본문 전문을 받는다."""
    return _get("lawService.do", {"target": "law", "MST": mst})


def summarize(obj, prefix: str = "", depth: int = 0, max_depth: int = 3) -> None:
    """저장한 JSON 의 윤곽만 출력한다 (구조 확인용)."""
    if depth > max_depth:
        return
    pad = "  " * depth
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, list):
                print(f"{pad}{k}: list[{len(v)}]")
                if v:
                    summarize(v[0], prefix, depth + 1, max_depth)
            elif isinstance(v, dict):
                print(f"{pad}{k}: dict")
                summarize(v, prefix, depth + 1, max_depth)
            else:
                text = str(v).replace("\n", " ")
                if len(text) > 50:
                    text = text[:50] + "…"
                print(f"{pad}{k}: {text}")
    elif isinstance(obj, list):
        print(f"{pad}list[{len(obj)}]")
        if obj:
            summarize(obj[0], prefix, depth + 1, max_depth)


def main() -> int:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    failed = []

    for name in TARGETS:
        short = SHORT[name]
        print(f"\n=== {name} ===")
        try:
            hit = search_law(name)
        except Exception as e:
            print(f"  검색 실패: {e}", file=sys.stderr)
            failed.append(short)
            continue

        mst = hit.get("법령일련번호") or hit.get("법령ID")
        print(f"  MST          : {mst}")
        print(f"  법령명        : {hit.get('법령명한글')}")
        print(f"  시행일자      : {hit.get('시행일자')}")
        print(f"  공포일자      : {hit.get('공포일자')}")
        print(f"  현행연혁      : {hit.get('현행연혁코드')}")

        if not mst:
            print(f"  ! 일련번호를 못 찾음. 검색 결과 키: {list(hit.keys())}", file=sys.stderr)
            failed.append(short)
            continue

        try:
            body = fetch_body(str(mst))
        except Exception as e:
            print(f"  본문 조회 실패: {e}", file=sys.stderr)
            failed.append(short)
            continue

        out = DATA_RAW / f"{today}_{short}.json"
        out.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
        size_kb = out.stat().st_size / 1024
        print(f"  저장          : {out.name} ({size_kb:,.0f} KB)")

        print(f"  --- 구조 ---")
        summarize(body, max_depth=2)

    if failed:
        print(f"\n실패: {', '.join(failed)}", file=sys.stderr)
        return 1
    print(f"\n완료. {DATA_RAW} 확인")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())