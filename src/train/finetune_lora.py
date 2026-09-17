"""LoRA 파인튜닝 — "학습→MLflow 기록→평가" 배관이 실제로 도는지 검증하는 1차 실험.

⚠️ 데이터는 make_toy_dataset.py가 만든 임시 데이터다. 이 스크립트 자체는 실제 센서
데이터가 오면 (거의) 그대로 재사용할 수 있게 짰다 — 바뀌는 건 데이터 경로뿐이어야 한다.

선택 이유(docs/모델_비교.md, docs/의사결정_로그.md 19번 참고):
- 베이스 모델 Qwen3-4B-Instruct-2507: Qwen 계열 중 최신 세대, Apache-2.0(재배포 자유),
  동급 크기 대비 다국어(한국어 포함) 성능 최상위권. Qwen2.5-3B로 배관 검증을 마친 뒤
  같은 라이선스 계열의 최신 모델로 교체 — 교체 비용이 모델 이름 한 줄뿐이라 순수 개선
- LoRA(양자화 없음, bf16): 4B 모델도 20GB VRAM에 bf16으로 충분히 들어가 QLoRA까지는 불필요.
  더 큰 모델을 시도할 때 QLoRA(4bit)를 꺼내는 걸로 미룸
"""

from __future__ import annotations

import os
from pathlib import Path

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _ROOT / "data" / "processed"

os.environ.setdefault("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "sllm-finetune")


def run_finetune(
    num_train_epochs: int = 5,
    run_name: str = "toy-sensor-lora",
    output_subdir: str = "toy-sensor-lora",
    train_file: str = "toy_sensor_alerts_train.jsonl",
    val_file: str = "toy_sensor_alerts_val.jsonl",
) -> Path:
    """LoRA 학습 1회 실행. auto_retrain.py가 에폭 수를 바꿔가며 여러 번 호출한다 —
    함수로 빼둬야 재학습 자동화 루프에서 재사용할 수 있다.

    output_subdir 기본값이 기존 경로(experiments/toy-sensor-lora)와 같다 — merge_lora.py 등
    기존 스크립트가 참조하는 고정 경로를 그대로 유지하기 위함. 재시도별로 다른 폴더에
    쓰고 싶으면 output_subdir을 바꿔서 호출한다(auto_retrain.py가 그렇게 함)."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype="bfloat16", device_map="auto")

    train_ds = load_dataset("json", data_files=str(_DATA_DIR / train_file), split="train")
    val_ds = load_dataset("json", data_files=str(_DATA_DIR / val_file), split="train")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    out_dir = _ROOT / "experiments" / output_subdir
    sft_config = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=2e-4,
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        report_to=["mlflow"],
        bf16=True,
        run_name=run_name,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=lora_config,
        processing_class=tokenizer,
    )
    trainer.train()

    final_dir = out_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"어댑터 저장 완료: {final_dir}")
    return final_dir


def main() -> None:
    run_finetune()


if __name__ == "__main__":
    main()
