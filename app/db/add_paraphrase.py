"""평가셋에 자연어 질의(question_nl)를 추가한다.

왜 필요한가
- 2026-10-28 BM25 단독 측정에서 Recall@5 0.972 가 나왔다. 천장이다.
  평가셋 질문이 조문을 덤프해서 쓴 것이라 조문의 어휘를 그대로 쓰고 있고,
  이건 BM25 가 가장 잘하는 조건이다. 이 상태로 임베딩을 붙이면
  개선분이 0.028 밖에 없어서 하이브리드 효과를 측정할 수 없다.
- 그래서 같은 gold 에 대해 "법령 용어를 쓰지 않은 질의"를 하나씩 단다.
  어휘 겹침이 적어 BM25 가 무너지고, 의미 검색이 메우는 폭이 드러난다.

작성 규칙
- gold 는 건드리지 않는다. 정답 조문은 그대로다.
- 조문에 나오는 법령 용어를 쓰지 않는다.
  "지체상금률" → "늦어지면 하루에 얼마씩 물어내나"
- 실무자가 실제로 물을 법한 문장으로 쓴다. 억지로 어렵게 쓰지 않는다.
- L3(clarify)는 조건이 빠진 상태를 유지한다. L5 는 거절 시나리오를 유지한다.

실행:
    python -m app.db.add_paraphrase            # 미리보기
    python -m app.db.add_paraphrase --write    # 평가셋에 반영
"""
import argparse
import json

from app.config import ROOT

EVALSET = ROOT / "eval" / "2026-10-27_evalset.json"

# id → 자연어 질의
NL = {
    # ---- L1 단순조회 -------------------------------------------------
    "L1-01": "입찰할 때 미리 걸어두는 돈은 얼마나 되나요?",
    "L1-02": "계약하면서 맡겨두는 돈은 계약한 금액의 얼마쯤인가요?",
    "L1-03": "공사가 늦어지면 하루에 얼마씩 물어내야 하나요?",
    "L1-04": "물건 다 받고 확인까지 끝났는데 돈은 언제 들어오나요?",
    "L1-05": "미리 자격 심사받는 건으로 서류를 보려면 언제까지 가능한가요?",
    "L1-06": "떨어진 결과가 납득이 안 가면 언제까지 문제 제기할 수 있나요?",

    # ---- L2 조건판단 -------------------------------------------------
    "L2-01": "3억원짜리 건물 짓는 일인데 그냥 업체 하나 정해서 맡겨도 되나요?",
    "L2-02": "천오백만원어치 물건을 업체 정해서 사려는데 가격 받아볼 곳이 몇 군데 필요한가요?",
    "L2-03": "80억 공사인데 원가에 세금 더한 금액의 97%로 써낸 업체를 뽑아도 되나요?",
    "L2-04": "짐 나르는 일이 늦어졌는데 하루에 얼마씩 물어내나요?",
    "L2-05": "7천만원짜리 일을 동네 작은 업체에 그냥 맡겨도 되나요?",

    # ---- L3 정보부족 (조건이 빠진 상태를 유지한다) ----------------------
    "L3-01": "늦어지면 하루에 얼마씩 물어내나요?",
    "L3-02": "3억원인데 업체 하나 정해서 맡겨도 되나요?",
    "L3-03": "미리 맡겨두는 돈은 얼마나 내나요?",

    # ---- L4 다중법령 -------------------------------------------------
    "L4-01": "돈 주기로 한 날을 넘기면 저희가 뭘 더 물어줘야 하나요?",
    "L4-02": "뽑혀놓고 계약을 안 하면 미리 걸어둔 돈은 못 돌려받나요?",
    "L4-03": "물건 만들어 주기로 한 게 늦었는데 무슨 근거로 얼마를 떼나요?",

    # ---- L5 거절 (근거가 없는 상태를 유지한다) --------------------------
    "L5-01": "국가계약법 아흔아홉 번째 조항에 전자입찰 꼭 해야 한다고 돼 있다는데 무슨 내용인가요?",
    "L5-02": "시청이나 구청에서 업체 하나 정해서 계약할 때 금액 한도가 어떻게 되나요?",
    "L5-03": "짜고 친 업체는 몇 년 동안 입찰 못 들어오나요?",
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--write", action="store_true", help="평가셋 파일에 반영")
    p.add_argument("--evalset", default=str(EVALSET))
    args = p.parse_args()

    path = ROOT / args.evalset if not args.evalset.startswith("/") else args.evalset
    data = json.loads(open(path, encoding="utf-8").read())

    missing = [i["id"] for i in data["items"] if i["id"] not in NL]
    extra = [k for k in NL if k not in {i["id"] for i in data["items"]}]
    if missing or extra:
        print(f"[문제] 평가셋에만 있음 {missing} / 여기에만 있음 {extra}")
        return

    for item in data["items"]:
        item["question_nl"] = NL[item["id"]]

    data["note"] = data.get("note", "") + (
        " question_nl 은 같은 gold 에 대한 자연어 질의다. "
        "법령 용어를 쓰지 않아 BM25 가 약한 조건이고, 하이브리드 효과를 재는 데 쓴다."
    )

    for item in data["items"]:
        print(f"{item['id']} [{item['expect']}]")
        print(f"  원문   {item['question']}")
        print(f"  자연어 {item['question_nl']}")

    if args.write:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n반영: {path}")
    else:
        print("\n미리보기입니다. 반영하려면 --write")


if __name__ == "__main__":
    main()