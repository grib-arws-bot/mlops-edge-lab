"""교육 도메인 RAG 질의응답 — 안전 도메인 query.py와 같은 패턴, 완전히 별도 인덱스.

웹 앱(src/web/app.py)이 embed_model/index/meta/llm을 시작할 때 한 번만 로드해서
넘겨준다 — 요청마다 다시 로드하면 느려서(임베딩 모델 로드만 수 초) 안 된다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer

from rag.build_edu_index import EMBED_MODEL

_ROOT = Path(__file__).resolve().parents[2]
_INDEX_PATH = _ROOT / "edu" / "processed" / "edu_index.faiss"
_META_PATH = _ROOT / "edu" / "processed" / "edu_chunks.jsonl"

_TOP_K = 4
_SYSTEM_PROMPT = (
    "당신은 중학교 사회 선생님입니다. 아래 [참고 자료]에 있는 내용만 근거로, 학생 눈높이로 "
    "친절하게 답변하세요. 근거에 없는 내용은 답하지 말고 '자료에서 근거를 찾지 못했습니다'라고 "
    "답하세요. 답변 끝 줄에 참고한 자료명을 '(출처: ...)' 형식으로 표시하세요."
)


def index_available() -> bool:
    return _INDEX_PATH.exists() and _META_PATH.exists()


def load_meta() -> list[dict]:
    text = _META_PATH.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.split("\n") if line.strip()]


def load_index():
    import faiss

    return faiss.read_index(str(_INDEX_PATH))


def retrieve(question: str, embed_model: SentenceTransformer, index, meta: list[dict], top_k: int = _TOP_K):
    q_vec = embed_model.encode([f"query: {question}"], normalize_embeddings=True)
    scores, idxs = index.search(np.asarray(q_vec, dtype="float32"), top_k)
    return [(meta[i], float(s)) for i, s in zip(idxs[0], scores[0]) if i != -1]


def answer(question: str, embed_model: SentenceTransformer, index, meta: list[dict], llm: Llama) -> dict:
    hits = retrieve(question, embed_model, index, meta)
    context = "\n\n".join(f"[{h['title']}]\n{h['text']}" for h, _ in hits)

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"[참고 자료]\n{context}\n\n[질문]\n{question}"},
    ]
    result = llm.create_chat_completion(messages=messages, temperature=0.0, max_tokens=400)

    return {
        "answer": result["choices"][0]["message"]["content"],
        "sources": [
            {"title": h["title"], "sub_domain": h["sub_domain"], "score": round(score, 3)}
            for h, score in hits
        ],
    }
