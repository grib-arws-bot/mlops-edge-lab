"""LoRA 어댑터를 베이스 모델에 병합해 통짜 모델로 저장한다.

엣지 런타임(llama.cpp)이 "베이스+어댑터를 실행 시점에 따로 얹는" 방식을 안정적으로
지원하지 않아서, GGUF로 변환하기 전에 미리 합쳐둔다(docs/의사결정_로그.md 22번).
"""

from __future__ import annotations

from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
_ROOT = Path(__file__).resolve().parents[2]
_ADAPTER_DIR = _ROOT / "experiments" / "toy-sensor-lora" / "final"
_MERGED_DIR = _ROOT / "experiments" / "toy-sensor-lora" / "merged"


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype="bfloat16", device_map="cpu")
    model = PeftModel.from_pretrained(base_model, str(_ADAPTER_DIR))

    merged = model.merge_and_unload()
    _MERGED_DIR.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(_MERGED_DIR), safe_serialization=True)
    tokenizer.save_pretrained(str(_MERGED_DIR))
    print(f"병합 완료: {_MERGED_DIR}")


if __name__ == "__main__":
    main()
