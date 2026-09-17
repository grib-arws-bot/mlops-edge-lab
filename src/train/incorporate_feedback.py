"""운영 피드백 루프(6→3번)를 닫는 스크립트 — data/processed/feedback.jsonl(웹 시뮬레이션
페이지에서 사람이 남긴 평가)을 학습 데이터 형식으로 바꿔 학습셋에 추가한다.

일부러 자동으로 재학습까지 트리거하지 않는다 — 사람이 이 스크립트를 실행하고
train.auto_retrain을 돌리는 2단계로 남겨뒀다. 실무에서도 피드백을 매번 즉시 반영하면
노이즈 하나로 모델이 흔들릴 수 있어서, 어느 정도 모아 배치로 처리하는 게 일반적이다.
"""

from __future__ import annotations

import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_FEEDBACK_PATH = _ROOT / "data" / "processed" / "feedback.jsonl"
_TRAIN_PATH = _ROOT / "data" / "processed" / "toy_sensor_alerts_train.jsonl"

_SYSTEM = "당신은 산업 현장 안전관리 보조 시스템입니다. 센서 이벤트를 받아 현장 담당자에게 보낼 간결한 한국어 알림 문장을 작성합니다."


def _to_example(feedback: dict) -> dict | None:
    event = feedback.get("event")
    if not event:
        return None
    # 사람이 "수정 필요"라고 표시하며 고친 문장이 있으면 그걸 정답으로, 없으면(=적절함
    # 평가) 원래 LLM이 낸 문장을 그대로 정답으로 삼는다.
    target = feedback.get("correction") or feedback.get("narrative")
    if not target:
        return None
    user_msg = (
        f"{event['location']} {event['category']}센서({event['substance']}), "
        f"측정값 {event['value']}{event['unit']}, 임계값 {event['threshold']}{event['unit']}"
    )
    return {
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": target},
        ],
        "kind": event["category"],
    }


def main() -> None:
    if not _FEEDBACK_PATH.exists():
        print("피드백 파일 없음 — 아직 시뮬레이션 페이지에서 피드백이 제출되지 않았습니다.")
        return

    added = 0
    with _TRAIN_PATH.open("a", encoding="utf-8") as train_f:
        for line in _FEEDBACK_PATH.read_text(encoding="utf-8").split("\n"):
            if not line.strip():
                continue
            example = _to_example(json.loads(line))
            if example:
                train_f.write(json.dumps(example, ensure_ascii=False) + "\n")
                added += 1

    print(f"{added}건을 {_TRAIN_PATH.name}에 추가했습니다.")
    if added:
        print("다음: `python -m train.auto_retrain`으로 재학습을 실행하세요.")


if __name__ == "__main__":
    main()
