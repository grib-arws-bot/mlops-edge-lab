"""RAG 검색 품질을 회귀 테스트하기 위한 골든셋 초안 생성.

실제 코퍼스(edu/processed/edu_chunks.jsonl, 738문서를 청킹한 결과)에서 소스별로
청크를 몇 개씩 뽑아 LLM에게 "이 글에서만 답할 수 있는 질문"을 만들게 한다.

**초안일 뿐이다** — LLM이 만든 질문이 실제로 그 문서에만 근거하는지, 질문 자체가
말이 되는지는 보장이 없다. 반드시 사람이 data/processed/rag_golden_set_draft.jsonl을
검수해서 골라낸 뒤 rag_golden_set.jsonl로 옮겨야 evaluate_retrieval.py가 신뢰할 수
있는 회귀 테스트로 쓸 수 있다(교육 자료 프로젝트 CLAUDE.md 원칙 — 결과물만 던지지
않는다, 여기서는 "LLM 초안 + 사람 검수" 원칙, 퀴즈 grounding 검증과 같은 철학).
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from llama_cpp import Llama

from collect import registry as collect_registry

_ROOT = Path(__file__).resolve().parents[2]
_META_PATH = _ROOT / "edu" / "processed" / "edu_chunks.jsonl"
_GGUF_PATH = _ROOT / "experiments" / "toy-sensor-lora" / "model-Q4_K_M.gguf"
_OUT_PATH = _ROOT / "data" / "processed" / "rag_golden_set_draft.jsonl"

_DRAFT_SYSTEM_PROMPT = (
    "다음 글에서만 답할 수 있는, 구체적인 사실 확인 질문을 한국어로 하나 만드세요. "
    "이 글이 없으면 답을 알 수 없을 정도로 구체적이어야 하고, 일반 상식만으로 "
    "답할 수 있는 질문은 안 됩니다. 설명 없이 반드시 아래 JSON 형식으로만 답하세요:\n"
    '{"question": "...", "answer_gist": "한두 문장 정답 요지"}'
)

_MIN_CHUNK_LEN = 200


def _registry_hint(source_id: str) -> tuple[str, str]:
    """이 소스가 속한 (학교급, 과목) 하나를 대표값으로 뽑는다 — 실제 학생이 질문할 때
    쓰는 필터(allowed_source_ids)와 같은 조건으로 채점하기 위함(evaluate_retrieval.py
    참고). 한 소스가 여러 학교급/과목에 걸치면 첫 번째 것만 대표로 쓴다 — 골든셋
    채점은 "이 조합으로 검색해도 찾아지는가"를 보는 것이지 모든 조합을 다 검증하는
    게 목적이 아니라서 충분하다."""
    for cfg in collect_registry.SOURCES:
        if cfg.source_id == source_id:
            return cfg.school_levels[0], cfg.subjects[0]
    return "", ""


def _load_chunks_by_source(n_per_source: int, seed: int) -> dict[str, list[dict]]:
    by_source: dict[str, list[dict]] = defaultdict(list)
    with _META_PATH.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                by_source[rec["source_id"]].append(rec)

    rng = random.Random(seed)
    picked: dict[str, list[dict]] = {}
    for source_id, records in by_source.items():
        candidates = [r for r in records if len(r["text"]) >= _MIN_CHUNK_LEN]
        if not candidates:
            continue
        picked[source_id] = rng.sample(candidates, min(n_per_source, len(candidates)))
    return picked


def build_draft(n_per_source: int = 2, seed: int = 42) -> list[dict]:
    picked = _load_chunks_by_source(n_per_source, seed)
    if not picked:
        raise RuntimeError(f"{_META_PATH}에 청크가 없습니다 — build_edu_index.py를 먼저 실행하세요")

    llm = Llama(model_path=str(_GGUF_PATH), n_ctx=4096, n_threads=8, n_gpu_layers=-1, verbose=False)

    drafts: list[dict] = []
    for source_id, chunks in picked.items():
        school_level, subject = _registry_hint(source_id)
        for rec in chunks:
            messages = [
                {"role": "system", "content": _DRAFT_SYSTEM_PROMPT},
                {"role": "user", "content": f"[{rec['title']}]\n{rec['text']}"},
            ]
            result = llm.create_chat_completion(messages=messages, temperature=0.3, max_tokens=200)
            raw = result["choices"][0]["message"]["content"]
            try:
                parsed = json.loads(raw)
                question = str(parsed["question"]).strip()
                answer_gist = str(parsed.get("answer_gist", "")).strip()
            except (json.JSONDecodeError, KeyError, TypeError):
                print(f"[skip] JSON 파싱 실패 — source={source_id}, raw={raw[:80]!r}")
                continue
            if not question:
                continue
            drafts.append({
                "question": question,
                "answer_gist": answer_gist,
                "expected_source_id": source_id,
                "school_level": school_level,
                "subject": subject,
                "source_title": rec["title"],
                "source_text_preview": rec["text"][:200],
            })
            print(f"[{source_id}] {question}")

    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _OUT_PATH.open("w", encoding="utf-8") as f:
        for d in drafts:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"\n초안 {len(drafts)}개 저장: {_OUT_PATH}")
    print("사람이 검수해서 골라낸 뒤 rag_golden_set.jsonl로 옮겨야 회귀 테스트로 쓸 수 있습니다.")
    return drafts


def main() -> None:
    build_draft()


if __name__ == "__main__":
    main()
