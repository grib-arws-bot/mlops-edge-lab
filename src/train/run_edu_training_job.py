"""운영 콘솔(웹)의 "학습 시작" 버튼이 서브프로세스로 띄우는 작업.

왜 서브프로세스인가: 웹 서비스(src/web/app.py)는 이미 인프로세스로 llama_cpp GPU
모델을 하나 띄워놓고 있다(/simulate, /control-room용). 여기서 transformers로 또
다른 모델을 GPU에 올려 학습시키면 그 프로세스와 메모리·CUDA 컨텍스트를 다투게 되고,
학습은 몇 초~몇 분씩 걸려서 웹 요청 핸들러 안에서 동기로 돌리면 타임아웃난다.
그래서 완전히 별도 프로세스로 띄우고, 웹은 상태 파일(job_status.json)만 폴링한다
— .health_state.json(healthcheck_alert.py)과 같은 패턴.

단계마다 상태 파일을 갱신해서, 운영 콘솔이 "지금 뭘 하고 있는지"를 보여줄 수 있게 한다.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_STATUS_PATH = _ROOT / "experiments" / "edu-social-lora" / "job_status.json"


def _write_status(**fields) -> None:
    _STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    current = {}
    if _STATUS_PATH.exists():
        try:
            current = json.loads(_STATUS_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            current = {}
    current.update(fields)
    current["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _STATUS_PATH.write_text(json.dumps(current, ensure_ascii=False), encoding="utf-8")


def run(source_ids: list[str]) -> None:
    from train.evaluate import run_eval
    from train.finetune_lora import run_finetune
    from train.make_edu_dataset import build_and_save

    _write_status(status="running", stage="generating_dataset", sources=source_ids, error=None, result=None)
    try:
        train_n, val_n = build_and_save(source_ids=source_ids)

        _write_status(stage="finetuning", train_examples=train_n, val_examples=val_n)
        adapter_dir = run_finetune(
            num_train_epochs=5,
            run_name="edu-social-lora",
            output_subdir="edu-social-lora",
            train_file="edu_social_train.jsonl",
            val_file="edu_social_val.jsonl",
        )

        _write_status(stage="evaluating")
        score = run_eval(
            adapter_dir=adapter_dir,
            run_name="edu-social-lora-eval",
            val_path=_ROOT / "data" / "processed" / "edu_social_val.jsonl",
            out_name="eval_edu_social.jsonl",
        )

        _write_status(
            status="done", stage="done",
            result={"avg_rougeL": score, "train_examples": train_n, "val_examples": val_n, "adapter_dir": str(adapter_dir)},
        )
    except Exception as exc:  # noqa: BLE001 — 운영 콘솔에 그대로 보여주기 위해 잡아서 기록
        _write_status(status="error", stage="error", error=f"{exc}\n{traceback.format_exc()[-2000:]}")
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", required=True, help="쉼표로 구분한 source_id 목록")
    args = parser.parse_args()
    run(args.sources.split(","))


if __name__ == "__main__":
    main()
