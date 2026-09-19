"""산업안전 RAG 검색 품질 회귀 테스트 — safety_golden_set.jsonl(사람이 검수한
질문+정답 출처)로 query.retrieve()가 실제로 정답 문서를 top-k 안에서 찾아내는지
자동 채점한다.

`rag/evaluate_retrieval.py`(AI튜터용)와 같은 패턴 — LLM 생성 없이 검색만 채점해서
몇 초 안에 끝난다. 이쪽은 하이브리드(BM25+임베딩)가 아니라 임베딩 단독 검색이라는
점이 AI튜터 쪽과의 실질적 차이 — RAG 레이어 추가 당시(로드맵 5번) 하이브리드 없이도
검증됐던 도메인이라 지금 이 골든셋으로 그 수치를 처음 정량화한다.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import faiss
import mlflow
from sentence_transformers import SentenceTransformer

from rag import query

_ROOT = Path(__file__).resolve().parents[2]
_GOLDEN_SET_PATH = _ROOT / "data" / "processed" / "safety_golden_set.jsonl"
_INDEX_PATH = _ROOT / "data" / "processed" / "rag_index.faiss"

os.environ.setdefault("MLFLOW_TRACKING_URI", "http://127.0.0.1:8082")


def _load_golden_set(path: Path = _GOLDEN_SET_PATH) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_eval(golden_set_path: Path = _GOLDEN_SET_PATH, top_k: int = 4) -> dict:
    if not _INDEX_PATH.exists():
        raise RuntimeError("안전 RAG 인덱스가 없습니다 — rag/build_index.py를 먼저 실행하세요")

    embed_model = SentenceTransformer(query.EMBED_MODEL)
    meta = query.load_meta()
    index = faiss.read_index(str(_INDEX_PATH))

    examples = _load_golden_set(golden_set_path)
    rows = []
    for ex in examples:
        hits = query.retrieve(ex["question"], embed_model, index, meta, top_k=top_k)
        hit_sources = [h["source"] for h, _ in hits]
        hit = ex["expected_source"] in hit_sources
        rank = hit_sources.index(ex["expected_source"]) + 1 if hit else None
        rows.append({
            "question": ex["question"], "expected_source": ex["expected_source"],
            "hit": hit, "rank": rank, "retrieved_sources": hit_sources,
        })
        mark = f"O (rank {rank})" if hit else "X"
        print(f"[{mark}] {ex['question']}")
        if not hit:
            print(f"    기대: {ex['expected_source']} / 실제 top-{top_k}: {hit_sources}")

    hit_rate = sum(r["hit"] for r in rows) / len(rows) if rows else 0.0
    print(f"\nhit_rate@{top_k}: {hit_rate:.3f} ({sum(r['hit'] for r in rows)}/{len(rows)})")

    mlflow.set_experiment("rag-retrieval-eval")
    with mlflow.start_run(run_name="safety-golden-set-eval"):
        mlflow.log_metric(f"hit_rate_at_{top_k}", hit_rate)
        mlflow.log_metric("n_examples", len(rows))
        out_path = _ROOT / "data" / "processed" / "safety_retrieval_eval_result.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        mlflow.log_artifact(str(out_path))

    return {"hit_rate": hit_rate, "rows": rows}


def main() -> None:
    run_eval()


if __name__ == "__main__":
    main()
