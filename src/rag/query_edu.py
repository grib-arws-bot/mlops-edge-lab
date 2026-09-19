"""교육 도메인 RAG 질의응답 — 안전 도메인 query.py와 같은 패턴, 완전히 별도 인덱스.

웹 앱(src/web/app.py)이 embed_model/index/meta/bm25/llm을 시작할 때 한 번만 로드해서
넘겨준다 — 요청마다 다시 로드하면 느려서(임베딩 모델 로드만 수 초) 안 된다.

**하이브리드 검색(2026-09-19)**: 임베딩(FAISS) 단독으로 "부산의 인구에 대해 알려줘"를
검색했더니, 무관한 경제 자료(0.87)가 진짜 인구 통계 자료(0.858)보다 높게 나왔다 — 짧은
핵심어("인구")가 700자 청크 임베딩에서 희석되는 게 원인. BM25(어휘 정확매칭, 실제로
"인구"라는 단어가 그 청크에 있는지를 봄)를 병행해서 Reciprocal Rank Fusion(RRF)으로
합친다. 프롬프트로 "무관한 근거 무시"를 시켜본 시도는 효과가 없어서(4B 모델이 그 판단을
못 함) 되돌렸고, 검색 단계 자체를 고치는 이 방식이 진짜 해결책이라고 판단했다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer

from rag.build_edu_index import EMBED_MODEL, korean_tokenize

_ROOT = Path(__file__).resolve().parents[2]
_INDEX_PATH = _ROOT / "edu" / "processed" / "edu_index.faiss"
_META_PATH = _ROOT / "edu" / "processed" / "edu_chunks.jsonl"
_BM25_INDEX_DIR = _ROOT / "edu" / "processed" / "edu_bm25"

_TOP_K = 4
_FETCH_K = 20  # RRF로 합치기 전, 각 방식에서 넉넉히 뽑아두는 후보 수(리서치 권고안)
_RRF_K = 60    # RRF 상수 — 순위 차이를 완만하게 반영(표준값)
_SYSTEM_PROMPT = (
    "당신은 중학교 사회 선생님입니다. 아래 [참고 자료]에 있는 내용만 근거로, 학생 눈높이로 "
    "친절하게 답변하세요. 근거에 없는 내용은 답하지 말고 '자료에서 근거를 찾지 못했습니다'라고 "
    "답하세요. 답변 끝 줄에 참고한 자료명을 '(출처: ...)' 형식으로 표시하세요."
)


def index_available() -> bool:
    return _INDEX_PATH.exists() and _META_PATH.exists() and _BM25_INDEX_DIR.exists()


def load_meta() -> list[dict]:
    text = _META_PATH.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.split("\n") if line.strip()]


def load_index():
    import faiss

    return faiss.read_index(str(_INDEX_PATH))


def load_bm25():
    import bm25s

    return bm25s.BM25.load(str(_BM25_INDEX_DIR))


_SUB_DOMAIN_BOOST = 1.15  # sub_domain 소프트 보정 배율 — 하드 필터가 아니라 동점에 가까울 때만 순위를 바꾸는 정도로 작게 잡음


def _infer_majority_sub_domain(dense_order: list[int], bm25_order: list[int], meta: list[dict]) -> str | None:
    """dense·BM25 상위 후보 풀의 sub_domain 다수결로 질문의 주제 영역을 추정한다.
    수작업 키워드 목록(예: "경제"→[돈,시장,...])을 쓰지 않은 이유: 이 프로젝트는 특정
    교과에 종속되지 않는 범용 수집·검색 체계를 목표로 설계했다(의사결정_로그 참고) —
    교과가 늘어날 때마다 키워드 목록을 손으로 유지보수해야 하면 그 목표에 어긋난다.
    검색 결과 자체의 신호(어느 sub_domain 청크가 상위 후보에 더 많이 걸렸는가)를 쓰면
    새 교과가 추가돼도 코드 변경 없이 그대로 동작한다."""
    counts: dict[str, int] = {}
    for idx in dense_order + bm25_order:
        sd = meta[idx]["sub_domain"]
        counts[sd] = counts.get(sd, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _rrf_combine(dense_order: list[int], bm25_order: list[int], meta: list[dict], top_k: int) -> list[int]:
    """두 순위 목록을 Reciprocal Rank Fusion으로 합친다 — 점수 스케일이 서로 다른
    코사인 유사도(dense)와 BM25 점수를 직접 비교할 수 없어서, "몇 등이었는가"만
    가지고 합치는 표준적인 방법을 쓴다. 그 위에 sub_domain 소프트 보정을 더한다 —
    "부산 인구" 실패 사례처럼 무관한 다른 교과 청크가 근소한 차이로 상위에 낄 때,
    다수결로 추정한 주제 영역과 같은 sub_domain 청크를 살짝 밀어올려 동점 근처의
    순위를 바로잡는다. 절대 점수 임계값으로 '근거 없음'을 판정하는 게이팅은 시도했으나
    보류했다(2026-09-19) — dense 코사인 유사도도, 필터링한 BM25 점수도, 두 방식의
    후보 교집합 크기도, 실측 결과 관련 질문과 무관 질문을 안정적으로 못 갈랐다(예:
    무관한 "세종대왕은 한글을 언제 만들었나요"의 BM25 최고점(4.02)이 실제 관련 있는
    "부산의 인구에 대해 알려줘"(4.06)보다 낮지도 않았다). 이 작은 코퍼스·경량 모델
    조합에서는 신뢰할 만한 절대 임계값을 잡을 근거가 없다고 판단해 억지로 넣지 않았다."""
    scores: dict[int, float] = {}
    for rank, idx in enumerate(dense_order):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (_RRF_K + rank + 1)
    for rank, idx in enumerate(bm25_order):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (_RRF_K + rank + 1)

    majority_sub_domain = _infer_majority_sub_domain(dense_order, bm25_order, meta)
    if majority_sub_domain is not None:
        for idx in scores:
            if meta[idx]["sub_domain"] == majority_sub_domain:
                scores[idx] *= _SUB_DOMAIN_BOOST

    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    return [idx for idx, _ in ranked[:top_k]]


def retrieve(
    question: str, embed_model: SentenceTransformer, index, meta: list[dict], bm25,
    top_k: int = _TOP_K, allowed_source_ids: set[str] | None = None,
):
    q_vec = embed_model.encode([f"query: {question}"], normalize_embeddings=True)
    dense_scores, dense_idxs = index.search(np.asarray(q_vec, dtype="float32"), _FETCH_K)
    dense_order = [int(i) for i in dense_idxs[0] if i != -1]
    dense_score_map = {int(i): float(s) for i, s in zip(dense_idxs[0], dense_scores[0]) if i != -1}

    tokenized_query = korean_tokenize([question])[0]
    bm25_results, _bm25_scores = bm25.retrieve([tokenized_query], k=min(_FETCH_K, len(meta)))
    bm25_order = [int(i) for i in bm25_results[0]]

    # 학교급/과목 선택(사용자 요청, 2026-09-19) — collect.registry의 source_id 기준
    # 허용 목록으로 후보를 거른다. 청크마다 별도 school_level/subject 필드를 새로
    # 두지 않고 이미 있는 source_id → registry 조회로 해결한 이유: 교과가 늘어도
    # 인덱스를 다시 만들 필요 없이 registry 설정만 보면 되게 하기 위함.
    if allowed_source_ids is not None:
        dense_order = [i for i in dense_order if meta[i]["source_id"] in allowed_source_ids]
        bm25_order = [i for i in bm25_order if meta[i]["source_id"] in allowed_source_ids]

    combined = _rrf_combine(dense_order, bm25_order, meta, top_k)
    # 화면에 보여주는 score는 해석하기 쉬운 코사인 유사도(dense)를 그대로 쓴다 —
    # BM25로만 뽑힌 항목(dense 상위 _FETCH_K 밖)은 0.0으로 표시된다는 한계는 있지만,
    # RRF 점수(0.03대 소수) 자체를 보여주는 것보다 직관적이라 이렇게 정했다.
    return [(meta[i], dense_score_map.get(i, 0.0)) for i in combined]


def answer(
    question: str, embed_model: SentenceTransformer, index, meta: list[dict], bm25, llm: Llama,
    allowed_source_ids: set[str] | None = None,
) -> dict:
    hits = retrieve(question, embed_model, index, meta, bm25, allowed_source_ids=allowed_source_ids)
    if not hits:
        return {"answer": "선택한 학교급·과목에는 아직 수집된 자료가 없습니다.", "sources": []}
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
