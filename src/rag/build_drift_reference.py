"""드리프트 감지(rag/drift_check.py)용 "기준(reference) 분포"를 더 두텁게 다시 만든다.

**배경(의사결정_로그 125번)**: 처음엔 `evaluate_retrieval_safety.py`가 골든셋(사람이
검수한 7문항) 채점 시점의 top-1 유사도를 그대로 reference로 썼다 — "표본을 부풀려
만들지 않는다"는 원칙 때문(의사결정_로그 124번). 그런데 실측해보니 n=7짜리 reference는
PSI 계산 자체를 못 쓰게 만드는 수준의 문제였다: current 쪽에 어떤 값을 넣어도(심지어
reference와 거의 겹치는 값이어도) PSI가 항상 13 근방으로 나왔다 — evidently의 구간(bin)
계산이 reference 표본이 너무 성겨서 대부분의 구간에서 reference 밀도가 0에 가까워지고,
log(current비율/reference비율) 항이 발산하기 때문(서버에서 직접 재현·확인).

**이번 방식**: `rag/build_golden_set_safety.build_draft()`로 실제 코퍼스 청크 N개에서
LLM이 질문을 초안한다(질문은 LLM이 만들지만 내용은 실제 문서 그대로, 답 정확도는
검증하지 않는다 — 이 reference는 "이 질문이 정답을 찾았는가"가 아니라 "이런 질문들을
던지면 top-1 점수가 보통 어느 대역에 분포하는가"만 필요하기 때문에 사람 검수 없이도
목적에 맞다). 각 질문으로 실제 `query.retrieve()`를 돌려 top-1 점수를 모은다 — 점수
자체는 전부 실측값이고 지어낸 숫자가 아니다.

**`safety_golden_set.jsonl`(7문항, hit-rate 회귀 테스트용)은 건드리지 않는다** — 그쪽은
여전히 사람이 검수한 정확도 평가 전용이고, 이 스크립트는 드리프트 reference만 별도로
다시 만든다."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

from rag import query
from rag.build_golden_set_safety import build_draft

_ROOT = Path(__file__).resolve().parents[2]
_INDEX_PATH = _ROOT / "data" / "processed" / "rag_index.faiss"
_REFERENCE_PATH = _ROOT / "data" / "processed" / "retrieval_score_reference.json"

os.environ.setdefault("MLFLOW_TRACKING_URI", "http://127.0.0.1:8082/mlflow")


def build(n_questions: int = 120, seed: int = 7) -> dict:
    if not _INDEX_PATH.exists():
        raise RuntimeError("안전 RAG 인덱스가 없습니다 — rag/build_index.py를 먼저 실행하세요")

    drafts = build_draft(n_per_source=1, seed=seed, max_sources=n_questions)
    print(f"LLM 초안 질문 {len(drafts)}개 생성 완료 — top-1 점수 채점 시작")

    embed_model = SentenceTransformer(query.EMBED_MODEL)
    meta = query.load_meta()
    index = faiss.read_index(str(_INDEX_PATH))

    scores = []
    for d in drafts:
        hits = query.retrieve(d["question"], embed_model, index, meta, top_k=1)
        if hits:
            scores.append(hits[0][1])

    if not scores:
        raise RuntimeError("점수를 하나도 얻지 못했습니다 — retrieve() 결과 확인 필요")

    reference = {
        "top1_scores": scores,
        "n": len(scores),
        "mean": sum(scores) / len(scores),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": (
            f"rag/build_golden_set_safety.build_draft()로 생성한 LLM 초안 질문 {len(scores)}건 "
            "— 정답 정확도는 검증하지 않음(점수 분포 전용). safety_golden_set.jsonl(7건, "
            "hit-rate 회귀 테스트용)과는 별개 — 그쪽은 변경하지 않았음."
        ),
    }
    _REFERENCE_PATH.write_text(json.dumps(reference, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"reference 저장 완료: {_REFERENCE_PATH} (n={reference['n']}, mean={reference['mean']:.4f})")
    return reference


def main() -> None:
    build()


if __name__ == "__main__":
    main()
