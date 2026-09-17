"""파인튜닝 재학습 자동화 — 평가 점수가 기준 미달이면 에폭을 늘려 자동으로 다시 학습한다.

이전까지는 docs/전략.md 설계도에만 있고 "설계됨, 자동화는 미구현"으로 남겨뒀던 3번(파인튜닝)
내부의 재학습 루프(의사결정_로그 22/30/35번)를 실제로 자동화한다.

⚠️ 지금 데이터(toy_sensor_alerts)는 템플릿 기반 합성 데이터라 적은 에폭에도 금방
과적합될 수 있다 — 그래도 "기준 미달 시 실제로 다시 도는가"를 눈으로 확인하려고
일부러 1에폭부터 시작하는 일정을 짰다. 통과한 어댑터만 정식 경로
(experiments/toy-sensor-lora/final)로 승격해서, merge_lora.py 등 기존 스크립트가
그대로 최신 어댑터를 쓰게 한다.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from train.evaluate import run_eval
from train.finetune_lora import run_finetune

_ROOT = Path(__file__).resolve().parents[2]
_CANONICAL_DIR = _ROOT / "experiments" / "toy-sensor-lora"

THRESHOLD = 0.90
EPOCH_SCHEDULE = [1, 3, 5]


def auto_retrain(threshold: float = THRESHOLD, epoch_schedule: list[int] | None = None) -> dict:
    epoch_schedule = epoch_schedule or EPOCH_SCHEDULE
    history: list[dict] = []

    for attempt, epochs in enumerate(epoch_schedule, start=1):
        run_name = f"toy-sensor-lora-auto-attempt{attempt}"
        print(f"\n[재학습 시도 {attempt}/{len(epoch_schedule)}] epochs={epochs}")

        adapter_dir = run_finetune(num_train_epochs=epochs, run_name=run_name, output_subdir=run_name)
        score = run_eval(adapter_dir=adapter_dir, run_name=f"{run_name}-eval")
        history.append({"attempt": attempt, "epochs": epochs, "rougeL": score})
        print(f"  평가 ROUGE-L = {score:.3f} (기준 {threshold})")

        if score >= threshold:
            final_dir = _CANONICAL_DIR / "final"
            print(f"기준 통과 — 이 어댑터를 정식 경로({final_dir})로 승격하고 종료")
            if final_dir.exists():
                shutil.rmtree(final_dir)
            shutil.copytree(adapter_dir, final_dir)
            return {"success": True, "history": history, "final_dir": str(final_dir)}

    print(f"\n최대 시도 횟수({len(epoch_schedule)}회) 도달 — 기준 미달인 채로 종료. 사람 개입 필요.")
    return {"success": False, "history": history, "final_dir": None}


def main() -> None:
    result = auto_retrain()
    print("\n=== 재학습 자동화 결과 ===")
    for h in result["history"]:
        print(f"  시도 {h['attempt']}: epochs={h['epochs']}, ROUGE-L={h['rougeL']:.3f}")
    print(f"성공: {result['success']}")


if __name__ == "__main__":
    main()
