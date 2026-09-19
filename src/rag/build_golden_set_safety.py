"""산업안전 RAG 검색 품질을 회귀 테스트하기 위한 골든셋 초안 생성.

`rag/build_golden_set.py`(AI튜터용)와 같은 방식 — 실제 코퍼스(data/processed/
rag_chunks.jsonl, 738건 안전문서를 청킹한 결과)에서 청크를 몇 개 뽑아 LLM에게
"이 글에서만 답할 수 있는 질문"을 만들게 한다. 여기도 LLM 초안은 반드시 사람이
검수해서 채택해야 한다(퀴즈 grounding 검증과 같은 "LLM 초안 + 사람 판정" 철학).

AI튜터 쪽과의 차이: 이쪽 코퍼스는 738개 문서·소스가 훨씬 많아서(AI튜터는 2개
소스뿐) 소스 전체가 아니라 무작위로 일부(max_sources)만 샘플링한다. 또 이 도메인
인덱스는 school_level/subject 같은 필터 개념이 없어(collect.registry 미사용,
source_id 대신 파일명 그대로인 "source" 필드) 그 부분은 뺐다.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from llama_cpp import Llama

_ROOT = Path(__file__).resolve().parents[2]
_META_PATH = _ROOT / "data" / "processed" / "rag_chunks.jsonl"
_GGUF_PATH = _ROOT / "experiments" / "toy-sensor-lora" / "model-Q4_K_M.gguf"
_OUT_PATH = _ROOT / "data" / "processed" / "safety_golden_set_draft.jsonl"

_DRAFT_SYSTEM_PROMPT = (
    "다음 글에서만 답할 수 있는, 구체적인 사실 확인 질문을 한국어로 하나 만드세요. "
    "이 글이 없으면 답을 알 수 없을 정도로 구체적이어야 하고, 일반 상식만으로 "
    "답할 수 있는 질문은 안 됩니다. 설명 없이 반드시 아래 JSON 형식으로만 답하세요:\n"
    '{"question": "...", "answer_gist": "한두 문장 정답 요지"}'
)

_MIN_CHUNK_LEN = 200


def _load_chunks_by_source(n_per_source: int, seed: int, max_sources: int) -> dict[str, list[dict]]:
    by_source: dict[str, list[dict]] = defaultdict(list)
    with _META_PATH.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                by_source[rec["source"]].append(rec)

    rng = random.Random(seed)
    sources = sorted(by_source.keys())
    # 738개 문서 전부에서 뽑으면 초안이 너무 많아진다 — 무작위로 일부 문서만 샘플링
    sampled_sources = rng.sample(sources, min(max_sources, len(sources)))

    picked: dict[str, list[dict]] = {}
    for source in sampled_sources:
        candidates = [r for r in by_source[source] if len(r["text"]) >= _MIN_CHUNK_LEN]
        if not candidates:
            continue
        picked[source] = rng.sample(candidates, min(n_per_source, len(candidates)))
    return picked


def build_draft(n_per_source: int = 1, seed: int = 42, max_sources: int = 10) -> list[dict]:
    picked = _load_chunks_by_source(n_per_source, seed, max_sources)
    if not picked:
        raise RuntimeError(f"{_META_PATH}에 청크가 없습니다 — rag/build_index.py를 먼저 실행하세요")

    llm = Llama(model_path=str(_GGUF_PATH), n_ctx=4096, n_threads=8, n_gpu_layers=-1, verbose=False)

    drafts: list[dict] = []
    for source, chunks in picked.items():
        for rec in chunks:
            messages = [
                {"role": "system", "content": _DRAFT_SYSTEM_PROMPT},
                {"role": "user", "content": f"[{source}]\n{rec['text']}"},
            ]
            result = llm.create_chat_completion(messages=messages, temperature=0.3, max_tokens=200)
            raw = result["choices"][0]["message"]["content"]
            try:
                parsed = json.loads(raw)
                question = str(parsed["question"]).strip()
                answer_gist = str(parsed.get("answer_gist", "")).strip()
            except (json.JSONDecodeError, KeyError, TypeError):
                print(f"[skip] JSON 파싱 실패 — source={source}, raw={raw[:80]!r}")
                continue
            if not question:
                continue
            drafts.append({
                "question": question,
                "answer_gist": answer_gist,
                "expected_source": source,
                "chunk_id": rec["chunk_id"],
                "source_text_preview": rec["text"][:200],
            })
            print(f"[{source}] {question}")

    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _OUT_PATH.open("w", encoding="utf-8") as f:
        for d in drafts:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"\n초안 {len(drafts)}개 저장: {_OUT_PATH}")
    print("사람이 검수해서 골라낸 뒤 safety_golden_set.jsonl로 옮겨야 회귀 테스트로 쓸 수 있습니다.")
    return drafts


def main() -> None:
    build_draft()


if __name__ == "__main__":
    main()
