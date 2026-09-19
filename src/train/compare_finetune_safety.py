"""산업안전 파인튜닝(toy-sensor-lora)도 베이스 모델과 비교한다 — `compare_finetune.py`
(AI튜터 edu-social-lora용)와 같은 패턴.

**정직하게 밝히는 한계**: 이 도메인의 학습 데이터(toy_sensor_alerts)는 실제 센서
로그가 아니라 템플릿 기반 합성 데이터다(의사결정_로그 21번 — 값은 placeholder로
계속 쓰기로 이미 결정됨). 그래서 이 비교의 절대 수치는 "진짜 품질"이 아니라
"파이프라인이 실제로 동작하는가"의 증거에 가깝다 — edu-social-lora 비교(99번,
실제 문서 기반이라 의미 있는 수치)와는 무게가 다르다는 걸 화면에서도 밝힌다.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from train.evaluate import run_eval

_ROOT = Path(__file__).resolve().parents[2]
_ADAPTER_DIR = _ROOT / "experiments" / "toy-sensor-lora" / "final"
_VAL_PATH = _ROOT / "data" / "processed" / "toy_sensor_alerts_val.jsonl"
_OUT_PATH = _ROOT / "data" / "processed" / "finetune_comparison_safety.json"

os.environ.setdefault("MLFLOW_TRACKING_URI", "http://127.0.0.1:8082")


def run_comparison() -> dict[str, float]:
    print("=== 베이스 모델(파인튜닝 없음) ===")
    base_score = run_eval(
        adapter_dir=None, run_name="safety-base-eval",
        val_path=_VAL_PATH, out_name="eval_safety_base.jsonl",
    )

    print("\n=== 파인튜닝된 어댑터(toy-sensor-lora) ===")
    finetuned_score = run_eval(
        adapter_dir=_ADAPTER_DIR, run_name="safety-finetuned-eval",
        val_path=_VAL_PATH, out_name="eval_safety_finetuned.jsonl",
    )

    delta = finetuned_score - base_score
    result = {
        "base_rougeL": base_score, "finetuned_rougeL": finetuned_score, "delta": delta,
        "synthetic_data_caveat": True,
    }

    print(f"\n=== 비교 결과 ===")
    print(f"베이스:      {base_score:.3f}")
    print(f"파인튜닝:    {finetuned_score:.3f}")
    print(f"개선폭:      {delta:+.3f}")
    print("주의: 합성 데이터 기반 — 절대 수치는 품질 지표가 아니라 배관 동작 증거(21번)")

    with _OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def main() -> None:
    run_comparison()


if __name__ == "__main__":
    main()
