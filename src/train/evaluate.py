"""파인튜닝된 어댑터로 검증셋을 생성해보고, 자동 채점(ROUGE-L)해서 MLflow에 기록한다.

이게 "자동 평가" 단계다 — 사람이 매번 눈으로 확인하는 대신, 정답과 생성 결과를 자동으로
비교해서 숫자로 남긴다. 지금은 장난감 데이터라 점수 자체는 의미가 크지 않지만, 다음에
진짜 데이터가 들어왔을 때 같은 스크립트로 "이번 체크포인트가 이전보다 나은가"를 비교할 수
있어야 한다는 게 핵심이다.

**2026-09-19 수정**: rouge_score 기본 토크나이저가 `[a-z0-9]+` 정규식 기반이라 한글을
전부 버리고 숫자·영문 단위만 비교한다는 걸 compare_quantization.py 작업 중 발견했다
(실측: "2층 사무실에서 CO2 농도가 1186ppm..." → ['2','co2','1186ppm',...]). 즉 지금까지
이 스크립트는 문장이 실제로 달라져도 숫자만 같으면 만점을 주는, 한국어에는 사실상
무의미한 채점을 auto_retrain.py의 재학습 통과 기준으로 써왔다. CharTokenizer(문자 단위
분리)로 교체 — 한국어는 조사가 어절에 붙어 공백 기준 단어 분리가 잘 안 맞아서, CJK 언어
ROUGE 평가에서 흔히 쓰는 방식이다.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import mlflow
import torch
from peft import PeftModel
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
_ROOT = Path(__file__).resolve().parents[2]
_ADAPTER_DIR = _ROOT / "experiments" / "toy-sensor-lora" / "final"
_VAL_PATH = _ROOT / "data" / "processed" / "toy_sensor_alerts_val.jsonl"

os.environ.setdefault("MLFLOW_TRACKING_URI", "http://127.0.0.1:8082")


class CharTokenizer:
    """공백 제거 후 문자 단위로 쪼갠다 — 한국어는 조사가 어절에 붙어서 공백 기준 단어
    분리로는 "농도가"와 "농도는"이 아예 다른 토큰이 돼버려 LCS가 과소평가된다. 문자
    단위로 보면 두 표현이 얼마나 겹치는지가 훨씬 정직하게 드러난다."""

    def tokenize(self, text: str) -> list[str]:
        return list(text.replace(" ", ""))


def load_val_examples(val_path: Path = _VAL_PATH) -> list[dict]:
    # .splitlines()는 쓰지 않는다 — 유니코드 줄경계 문자가 텍스트에 섞이면 오작동한다
    # (src/rag/query.py에서 실제로 겪은 문제, docs/의사결정_로그.md 26번 참고)
    text = val_path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.split("\n") if line.strip()]


def run_eval(
    adapter_dir: Path | None = _ADAPTER_DIR,
    run_name: str = "toy-sensor-lora-eval",
    val_path: Path = _VAL_PATH,
    out_name: str = "eval_toy_sensor.jsonl",
) -> float:
    """어댑터 하나를 평가해서 평균 ROUGE-L을 반환한다. auto_retrain.py가 이 반환값으로
    "이번 학습이 기준을 통과했는가"를 판단한다 — 그래서 함수로 뺐다(원래는 main()에 다
    들어있었음). val_path/out_name은 도메인별 검증셋을 재사용하려고 뺐다(2026-09-19,
    edu_social 검증에서 처음 필요해짐) — 기본값은 기존 toy-sensor 경로 그대로라
    auto_retrain.py 호출부는 안 바뀜.

    adapter_dir=None이면 LoRA 없이 베이스 모델 그대로 평가한다(2026-09-20,
    compare_finetune.py에서 처음 필요해짐) — "파인튜닝이 실제로 도움이 되는가"를
    같은 검증셋·같은 채점 방식으로 베이스와 직접 비교하기 위함. 지금까지 이 함수는
    파인튜닝된 어댑터의 절대 점수만 봤지 베이스 대비 개선폭은 한 번도 측정한 적이
    없었다."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype="bfloat16", device_map="auto")
    model = PeftModel.from_pretrained(base_model, str(adapter_dir)) if adapter_dir is not None else base_model
    model.eval()

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False, tokenizer=CharTokenizer())
    examples = load_val_examples(val_path)

    rows = []
    for ex in examples:
        system_msg, user_msg, reference = ex["messages"][0], ex["messages"][1], ex["messages"][2]["content"]
        prompt = tokenizer.apply_chat_template(
            [system_msg, user_msg], tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=120, do_sample=False)
        generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        score = scorer.score(reference, generated)["rougeL"].fmeasure
        rows.append({
            "kind": ex["kind"],
            "input": user_msg["content"],
            "reference": reference,
            "generated": generated,
            "rougeL": score,
        })

    avg_rouge = sum(r["rougeL"] for r in rows) / len(rows)

    print("=== 검증셋 생성 결과 ===")
    for r in rows:
        print(f"\n[{r['kind']}] ROUGE-L={r['rougeL']:.2f}")
        print(f"  입력: {r['input']}")
        print(f"  정답: {r['reference']}")
        print(f"  생성: {r['generated']}")
    print(f"\n평균 ROUGE-L: {avg_rouge:.3f}")

    out_path = _ROOT / "data" / "processed" / out_name
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    mlflow.set_experiment("sllm-finetune")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_metric("avg_rougeL", avg_rouge)
        mlflow.log_artifact(str(out_path))

    return avg_rouge


def main() -> None:
    run_eval()


if __name__ == "__main__":
    main()
