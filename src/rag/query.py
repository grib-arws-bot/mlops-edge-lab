"""RAG 질의응답 — FAISS로 관련 문서 조각을 찾고, sLLM(GGUF, llama-cpp-python)이 그
조각만 근거로 답하게 한다.

BidRadar의 "근거 없는 판정은 화면에 내보내지 않는다" 원칙(CLAUDE.md 참고)과 같은 이유로,
프롬프트에서 "근거에 없으면 모른다고 답해라"를 명시하고 답변 끝에 출처 문서명을 붙인다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import faiss
import numpy as np
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer

from rag.build_index import EMBED_MODEL

_ROOT = Path(__file__).resolve().parents[2]
_INDEX_PATH = _ROOT / "data" / "processed" / "rag_index.faiss"
_META_PATH = _ROOT / "data" / "processed" / "rag_chunks.jsonl"
_GGUF_PATH = _ROOT / "experiments" / "toy-sensor-lora" / "model-Q4_K_M.gguf"

_TOP_K = 4
_FETCH_K_WITH_FILTER = 200
"""source_filter가 있을 때 FAISS에서 미리 가져올 후보 개수 — MSDS 청크가 전체
인덱스의 대다수를 차지하게 된 뒤(2026-09-23), "산업안전 원본만" 같은 좁은 필터는
상위 top_k(4)만 보면 걸릴 게 거의 없다. query_edu.py의 allowed_source_ids 패턴과
같은 이유로 넉넉히 가져온 다음 걸러서 자른다."""
_SYSTEM_PROMPT = (
    "당신은 산업안전 가이드라인 문서를 근거로 답변하는 도우미입니다. "
    "아래 [참고 문서]에 있는 내용만 근거로 답하세요. "
    "근거에 없는 내용은 답하지 말고 '문서에서 근거를 찾지 못했습니다'라고 답하세요. "
    "답변 끝 줄에 참고한 문서명을 '(출처: ...)' 형식으로 표시하세요."
)


def load_meta() -> list[dict]:
    # .splitlines()는 쓰지 않는다 — 청크 텍스트 안에 U+2028 같은 유니코드 줄경계 문자가
    # 섞여 있으면 splitlines()가 그걸 줄바꿈으로 오인해서 JSON을 반쪽으로 잘라버린다.
    # 쓸 때 "\n"으로만 구분했으니 읽을 때도 "\n"만 기준으로 나눈다.
    text = _META_PATH.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.split("\n") if line.strip()]


def retrieve(
    question: str, embed_model: SentenceTransformer, index, meta: list[dict], top_k: int = _TOP_K,
    source_filter=None,
):
    """source_filter(선택): meta 항목 하나(dict)를 받아 True/False를 돌려주는 함수
    — 예: `lambda m: m["source"].startswith("msds_")`. 지정하면 top_k보다 훨씬 많이
    가져온 뒤(_FETCH_K_WITH_FILTER) 걸러서 top_k개로 자른다 — 필터 없이 top_k만
    가져오면 원하는 소스의 청크가 그 안에 하나도 없을 수 있다."""
    q_vec = embed_model.encode([f"query: {question}"], normalize_embeddings=True)
    fetch_k = _FETCH_K_WITH_FILTER if source_filter is not None else top_k
    scores, idxs = index.search(np.asarray(q_vec, dtype="float32"), min(fetch_k, index.ntotal))
    hits = [(meta[i], float(s)) for i, s in zip(idxs[0], scores[0]) if i != -1]
    if source_filter is not None:
        hits = [h for h in hits if source_filter(h[0])]
    return hits[:top_k]


def answer(question: str, embed_model: SentenceTransformer, index, meta: list[dict], llm: Llama) -> str:
    hits = retrieve(question, embed_model, index, meta)
    context = "\n\n".join(f"[{h['source']} #{h['chunk_id']}]\n{h['text']}" for h, _ in hits)

    print("=== 검색된 근거 ===")
    for h, score in hits:
        print(f"  score={score:.3f}  {h['source']} #{h['chunk_id']}")

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"[참고 문서]\n{context}\n\n[질문]\n{question}"},
    ]
    result = llm.create_chat_completion(messages=messages, temperature=0.0, max_tokens=400)
    return result["choices"][0]["message"]["content"]


def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else "화기 작업을 할 때 필요한 허가 절차는 무엇인가요?"

    embed_model = SentenceTransformer(EMBED_MODEL)
    index = faiss.read_index(str(_INDEX_PATH))
    meta = load_meta()
    llm = Llama(model_path=str(_GGUF_PATH), n_ctx=4096, n_threads=8, verbose=False)

    print(f"질문: {question}\n")
    result = answer(question, embed_model, index, meta, llm)
    print("\n=== 답변 ===")
    print(result)


if __name__ == "__main__":
    main()
