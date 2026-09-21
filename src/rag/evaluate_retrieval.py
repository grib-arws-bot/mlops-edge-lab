"""RAG 검색 품질 회귀 테스트 — rag_golden_set.jsonl(사람이 검수한 질문+정답 출처)로
hybrid retrieve()가 실제로 정답 문서를 top-k 안에서 찾아내는지 자동 채점한다.

evaluate.py/compare_quantization.py와 달리 LLM 생성 없이 검색만 채점한다 — 지금
검증하려는 게 "검색이 올바른 문서를 찾는가"이지 "생성된 답변이 좋은가"가 아니라서다.
LLM 호출이 없어 몇 초 안에 끝나고, 프롬프트나 모델 교체와 무관하게 BM25/임베딩/
sub_domain 보정 같은 검색 로직만 바뀌었을 때 "이전보다 나아졌는가"를 즉시 비교할
수 있다.

evaluate.py와 마찬가지로 지금은 골든셋이 3문항뿐이라 절대 수치 자체보다 "회귀
테스트가 실제로 동작하는가"에 의미가 있다 — 소스가 늘어나면(현재 10개 등록 소스 중
2개만 인덱스에 있음, 92번 참고 예정) 같은 스크립트로 그대로 확장된다.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import mlflow
from sentence_transformers import SentenceTransformer

from collect import registry as collect_registry
from rag import query_edu

_ROOT = Path(__file__).resolve().parents[2]
_GOLDEN_SET_PATH = _ROOT / "data" / "processed" / "rag_golden_set.jsonl"

os.environ.setdefault("MLFLOW_TRACKING_URI", "http://127.0.0.1:8082/mlflow")


def _load_golden_set(path: Path = _GOLDEN_SET_PATH) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_eval(golden_set_path: Path = _GOLDEN_SET_PATH, top_k: int = 4) -> dict:
    if not query_edu.index_available():
        raise RuntimeError("edu RAG 인덱스가 없습니다 — build_edu_index.py를 먼저 실행하세요")

    embed_model = SentenceTransformer(query_edu.EMBED_MODEL)
    meta = query_edu.load_meta()
    index = query_edu.load_index()
    bm25 = query_edu.load_bm25()

    examples = _load_golden_set(golden_set_path)
    rows = []
    for ex in examples:
        allowed = None
        if ex.get("school_level") and ex.get("subject"):
            allowed = {c.source_id for c in collect_registry.sources_for(ex["school_level"], ex["subject"])}

        hits = query_edu.retrieve(ex["question"], embed_model, index, meta, bm25, top_k=top_k, allowed_source_ids=allowed)
        hit_source_ids = [h["source_id"] for h, _ in hits]
        hit = ex["expected_source_id"] in hit_source_ids
        rank = hit_source_ids.index(ex["expected_source_id"]) + 1 if hit else None
        rows.append({
            "question": ex["question"], "expected_source_id": ex["expected_source_id"],
            "hit": hit, "rank": rank, "retrieved_source_ids": hit_source_ids,
        })
        mark = f"O (rank {rank})" if hit else "X"
        print(f"[{mark}] {ex['question']}")
        if not hit:
            print(f"    기대: {ex['expected_source_id']} / 실제 top-{top_k}: {hit_source_ids}")

    hit_rate = sum(r["hit"] for r in rows) / len(rows) if rows else 0.0
    print(f"\nhit_rate@{top_k}: {hit_rate:.3f} ({sum(r['hit'] for r in rows)}/{len(rows)})")

    mlflow.set_experiment("rag-retrieval-eval")
    with mlflow.start_run(run_name="golden-set-eval"):
        mlflow.log_metric(f"hit_rate_at_{top_k}", hit_rate)
        mlflow.log_metric("n_examples", len(rows))
        out_path = _ROOT / "data" / "processed" / "rag_retrieval_eval_result.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        mlflow.log_artifact(str(out_path))

    return {"hit_rate": hit_rate, "rows": rows}


def main() -> None:
    run_eval()


if __name__ == "__main__":
    main()
