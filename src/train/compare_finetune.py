"""파인튜닝이 실제로 도움이 되는가 — 베이스 모델 vs LoRA 파인튜닝 모델을 같은
검증셋·같은 채점(ROUGE-L, CharTokenizer)으로 직접 비교한다.

**왜 지금까지 없었나**: evaluate.py는 파인튜닝된 어댑터의 절대 점수만 봐왔다.
toy-sensor 데이터는 합성이라 "베이스 대비 개선폭"을 재는 게 의미 없다고 보고
보류해왔는데(finetune_pipeline 메모), edu-social-lora는 실제 문서(kdi_econ_edu·
nationalatlas_youth, 42건)로 만든 학습 데이터라 이 비교가 실제로 의미를 가진다.

compare_quantization.py와 같은 패턴 — 변수 하나(어댑터 유무)만 격리해서 비교한다.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import mlflow

from train.evaluate import run_eval

_ROOT = Path(__file__).resolve().parents[2]
_ADAPTER_DIR = _ROOT / "experiments" / "edu-social-lora" / "final"
_VAL_PATH = _ROOT / "data" / "processed" / "edu_social_val.jsonl"
_OUT_PATH = _ROOT / "data" / "processed" / "finetune_comparison.json"

os.environ.setdefault("MLFLOW_TRACKING_URI", "http://127.0.0.1:8082")


def run_comparison() -> dict[str, float]:
    mlflow.set_experiment("sllm-finetune")

    print("=== 베이스 모델(파인튜닝 없음) ===")
    with mlflow.start_run(run_name="edu-social-base-eval"):
        base_score = run_eval(
            adapter_dir=None, run_name="edu-social-base-eval",
            val_path=_VAL_PATH, out_name="eval_edu_social_base.jsonl",
        )
        mlflow.log_metric("avg_rougeL", base_score)
        mlflow.log_param("adapter", "none")

    print("\n=== 파인튜닝된 어댑터(edu-social-lora) ===")
    with mlflow.start_run(run_name="edu-social-finetuned-eval"):
        finetuned_score = run_eval(
            adapter_dir=_ADAPTER_DIR, run_name="edu-social-finetuned-eval",
            val_path=_VAL_PATH, out_name="eval_edu_social_finetuned.jsonl",
        )
        mlflow.log_metric("avg_rougeL", finetuned_score)
        mlflow.log_param("adapter", "edu-social-lora")

    delta = finetuned_score - base_score
    result = {"base_rougeL": base_score, "finetuned_rougeL": finetuned_score, "delta": delta}

    print(f"\n=== 비교 결과 ===")
    print(f"베이스:      {base_score:.3f}")
    print(f"파인튜닝:    {finetuned_score:.3f}")
    print(f"개선폭:      {delta:+.3f}")

    with _OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def main() -> None:
    run_comparison()


if __name__ == "__main__":
    main()
