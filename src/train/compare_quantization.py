"""양자화 전후 품질 비교 — model-f16.gguf(무양자화) vs model-Q4_K_M.gguf.

기존 evaluate.py는 "transformers+LoRA 어댑터"로 채점하지만, 여기서는 일부러 그 경로를
안 쓴다. transformers(fp16 HF 모델)와 llama.cpp(양자화 GGUF)를 직접 비교하면 런타임
차이(토크나이즈·샘플링 구현 차이 등)가 양자화 효과와 섞여버린다. 그래서 두 GGUF 파일
(model-f16.gguf, model-Q4_K_M.gguf)을 똑같이 llama-cpp-python으로 로드해서, "양자화"라는
변수 하나만 격리해서 비교한다.

정답셋은 evaluate.py와 동일한 것을 재사용하되, 채점 토크나이저는 다르게 한다 —
rouge_score 기본 토크나이저는 `[a-z0-9]+` 정규식 기반이라 한글을 전부 버리고 숫자·영문
단위만 남긴다(실측: "2층 사무실에서 CO2 농도가 1186ppm..." → ['2','co2','1186ppm',...]).
그래서 문장 자체가 달라져도 숫자만 같으면 ROUGE-L이 1.0이 나오는, 한국어에는 사실상
무의미한 채점이 된다. 이 문제는 evaluate.py(→auto_retrain.py 재학습 게이트)에도 그대로
있지만 이 스크립트의 스코프가 아니라 별도로 다룬다 — 여기서는 문자 단위 토크나이저로
직접 교체해서 실제로 한글 차이를 반영하는 채점을 한다(CJK 언어에서 ROUGE를 쓸 때
일반적인 방식 — 공백 기준 단어 분리가 한국어 조사 결합 특성상 잘 안 맞기 때문).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import mlflow
from llama_cpp import Llama
from rouge_score import rouge_scorer

from train.evaluate import load_val_examples


class _CharTokenizer:
    """공백 제거 후 문자 단위로 쪼갠다 — 한국어는 조사가 어절에 붙어서 공백 기준 단어
    분리로는 "농도가"와 "농도는"이 아예 다른 토큰이 돼버려 LCS가 과소평가된다. 문자
    단위로 보면 두 표현이 얼마나 겹치는지가 훨씬 정직하게 드러난다."""

    def tokenize(self, text: str) -> list[str]:
        return list(text.replace(" ", ""))


_ROOT = Path(__file__).resolve().parents[2]
_MODELS = {
    "f16": _ROOT / "experiments" / "toy-sensor-lora" / "model-f16.gguf",
    "Q4_K_M": _ROOT / "experiments" / "toy-sensor-lora" / "model-Q4_K_M.gguf",
}
_OUT_PATH = _ROOT / "data" / "processed" / "quant_comparison.jsonl"

os.environ.setdefault("MLFLOW_TRACKING_URI", "http://127.0.0.1:8082")


def _generate(llm: Llama, system_msg: dict, user_msg: dict) -> str:
    result = llm.create_chat_completion(
        messages=[system_msg, user_msg],
        temperature=0.0,  # evaluate.py의 do_sample=False(그리디)와 동일 조건
        max_tokens=120,   # evaluate.py의 max_new_tokens=120과 동일
    )
    return result["choices"][0]["message"]["content"]


def run_comparison() -> dict[str, float]:
    examples = load_val_examples()
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False, tokenizer=_CharTokenizer())

    rows = []
    avg_scores: dict[str, float] = {}

    for tag, path in _MODELS.items():
        llm = Llama(model_path=str(path), n_ctx=4096, n_threads=8, verbose=False)
        scores = []
        for i, ex in enumerate(examples):
            system_msg, user_msg, reference = ex["messages"][0], ex["messages"][1], ex["messages"][2]["content"]
            generated = _generate(llm, system_msg, user_msg)
            score = scorer.score(reference, generated)["rougeL"].fmeasure
            scores.append(score)

            if len(rows) <= i:
                rows.append({"kind": ex["kind"], "input": user_msg["content"], "reference": reference})
            rows[i][f"generated_{tag}"] = generated
            rows[i][f"rougeL_{tag}"] = score

        avg_scores[tag] = sum(scores) / len(scores)
        del llm  # 다음 모델 로드 전에 메모리 반환

    print("=== 양자화 전후 비교 (model-f16.gguf vs model-Q4_K_M.gguf) ===")
    for r in rows:
        print(f"\n[{r['kind']}] f16 ROUGE-L={r['rougeL_f16']:.4f}  Q4_K_M ROUGE-L={r['rougeL_Q4_K_M']:.4f}")
        print(f"  입력: {r['input']}")
        print(f"  정답: {r['reference']}")
        print(f"  f16    : {r['generated_f16']}")
        print(f"  Q4_K_M : {r['generated_Q4_K_M']}")

    delta = avg_scores["Q4_K_M"] - avg_scores["f16"]
    print(f"\n평균 ROUGE-L — f16: {avg_scores['f16']:.3f}  Q4_K_M: {avg_scores['Q4_K_M']:.3f}  차이: {delta:+.3f}")

    with _OUT_PATH.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    mlflow.set_experiment("sllm-finetune")
    with mlflow.start_run(run_name="quantization-comparison"):
        mlflow.log_metric("avg_rougeL_f16", avg_scores["f16"])
        mlflow.log_metric("avg_rougeL_q4km", avg_scores["Q4_K_M"])
        mlflow.log_metric("rougeL_delta_q4km_minus_f16", delta)
        mlflow.log_artifact(str(_OUT_PATH))

    return avg_scores


def main() -> None:
    run_comparison()


if __name__ == "__main__":
    main()
