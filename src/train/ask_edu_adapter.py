"""edu-social-lora 어댑터에 RAG 없이 직접 질문 — "파인튜닝만으로는 사실을 지어낸다"를
운영 콘솔의 학생 체험 화면에서 RAG 답변과 나란히 비교해서 보여주기 위한 비교용 경로.

별도 프로세스로 도는 이유: 웹 서비스가 이미 인프로세스 GPU 모델(llama_cpp)을 띄워놓고
있는데, 이 어댑터는 아직 GGUF로 변환 안 해서 transformers+PEFT로 불러와야 한다 —
같은 프로세스에 얹으면 GPU 메모리·CUDA 컨텍스트를 다투게 된다(finetune_lora.py에서
실제로 겪은 문제와 같은 종류). 매번 모델을 새로 불러와서 느리지만(비교 기능이라
자주 안 쓰임), 안전하다.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
_ROOT = Path(__file__).resolve().parents[2]
_ADAPTER_DIR = _ROOT / "experiments" / "edu-social-lora" / "final"
_SYSTEM_PROMPT = "당신은 중학교 사회 선생님입니다. 학생 질문에 학생 눈높이로 친절하게 답합니다."


def ask(question: str) -> str:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype="bfloat16", device_map={"": 0})
    model = PeftModel.from_pretrained(base_model, str(_ADAPTER_DIR))
    model.eval()

    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": _SYSTEM_PROMPT}, {"role": "user", "content": question}],
        tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    args = parser.parse_args()
    print(json.dumps({"answer": ask(args.question)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
