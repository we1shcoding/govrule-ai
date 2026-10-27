"""수집한 원본 JSON 의 조문 구조를 조사한다.

파싱 코드를 쓰기 전에 실행한다. 목적은 두 가지다.
  1. 조문단위 안에 어떤 키가 있고, 어떤 키가 빠질 수 있는가
  2. 항·호·목이 dict 인가 list 인가, 아니면 없는가
     (법령마다, 조문마다 다르면 파싱 코드가 거기서 터진다)

실행:
    python -m app.collect.inspect_raw
"""
import json
import sys
from collections import Counter

from app.config import DATA_RAW

sys.stdout.reconfigure(errors="replace")

SEP = "=" * 60


def shape(v) -> str:
    if isinstance(v, list):
        return f"list[{len(v)}]"
    if isinstance(v, dict):
        return "dict"
    if v is None:
        return "None"
    if isinstance(v, str):
        return "str(빈값)" if v == "" else "str"
    return type(v).__name__


def trim(obj, limit: int = 60):
    """출력용으로 긴 문자열을 자른다."""
    if isinstance(obj, dict):
        return {k: trim(v, limit) for k, v in obj.items()}
    if isinstance(obj, list):
        return [trim(v, limit) for v in obj]
    if isinstance(obj, str) and len(obj) > limit:
        return obj[:limit] + f"...({len(obj)}자)"
    return obj


def dump(obj) -> str:
    return json.dumps(trim(obj), ensure_ascii=False, indent=2)


def inspect(path) -> None:
    print(f"\n{SEP}\n{path.name}\n{SEP}")
    data = json.loads(path.read_text(encoding="utf-8"))
    units = data["법령"]["조문"]["조문단위"]
    print(f"조문단위: {len(units)}개\n")

    # --- 1. 키 출현 빈도 ---
    keys = Counter()
    for u in units:
        keys.update(u.keys())
    print("[키 출현 빈도]  (전체보다 작으면 없는 조문이 있다는 뜻)")
    for k, n in keys.most_common():
        mark = "" if n == len(units) else "   ← 일부에만 있음"
        print(f"  {n:>4}/{len(units)}  {k}{mark}")

    # --- 2. 중첩 필드의 자료형 분포 ---
    print("\n[중첩 필드 자료형]  (dict 와 list 가 섞이면 분기 필요)")
    for field in ("항", "호", "목"):
        found = Counter()
        for u in units:
            if field in u:
                found[shape(u[field])] += 1
            # 항 안의 호, 호 안의 목까지 한 단계 더
            for sub in _walk(u):
                if isinstance(sub, dict) and field in sub:
                    found[shape(sub[field])] += 1
        if found:
            print(f"  {field}: {dict(found)}")
        else:
            print(f"  {field}: (조문단위 최상위에는 없음)")

    # --- 3. 조문여부 분포 ---
    kinds = Counter(u.get("조문여부", "(키 없음)") for u in units)
    print(f"\n[조문여부] {dict(kinds)}")
    print("  '전문' 이 아닌 항목은 장·절 제목 행일 수 있다. 본문이 없으므로")
    print("  색인 대상에서 빼야 하는지 샘플로 확인할 것.")

    # --- 4. 샘플 ---
    simple = next((u for u in units if "항" not in u), None)
    nested = next((u for u in units if isinstance(u.get("항"), list)), None)
    single = next((u for u in units if isinstance(u.get("항"), dict)), None)

    for label, sample in (
        ("항이 없는 조문", simple),
        ("항이 dict 인 조문 (항 1개)", single),
        ("항이 list 인 조문 (항 여러 개)", nested),
    ):
        print(f"\n--- 샘플: {label} ---")
        if sample is None:
            print("  해당 없음")
        else:
            print(dump(sample))


def _walk(obj):
    """dict/list 를 재귀적으로 훑으며 dict 를 내놓는다."""
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v)


def main() -> int:
    files = sorted(DATA_RAW.glob("*.json"))
    if not files:
        print(f"{DATA_RAW} 에 JSON 이 없습니다. fetch_law 를 먼저 실행하십시오.")
        return 1
    for f in files:
        inspect(f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
