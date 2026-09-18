"""RAG 유무 비교 실험 — 산업안전 도메인과 달리 사회과 일반 지식은 LLM이 사전학습으로
이미 어느 정도 알고 있을 것이라는 가설을 실제로 검증한다.

산업안전 RAG(src/rag/)와 완전히 별개 코퍼스를 쓴다 — 교육과정 문서를 안전 문서 인덱스에
섞으면 두 도메인의 검색 품질이 서로 오염된다. 그래서 이 스크립트는 그때그때 메모리에서
자체적으로 청킹·임베딩하고 별도 인덱스 파일을 남기지 않는다(1회성 실험이라 인덱스를
영구 저장할 필요가 아직 없음 — 커리큘럼 봇으로 발전시킬 때 build_index.py처럼 정식화).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import faiss
import numpy as np
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer

from extract.core import extract_text
from rag.chunk import chunk_text

_ROOT = Path(__file__).resolve().parents[1]
_GGUF_PATH = _ROOT / "experiments" / "toy-sensor-lora" / "model-Q4_K_M.gguf"
_CURRICULUM_PDF = Path(__file__).resolve().parent / "raw" / "[별책7] 사회과 교육과정.pdf"
_EMBED_MODEL = "intfloat/multilingual-e5-small"

QUESTIONS = [
    "3·1 운동에 대해 중학생이 이해할 수 있게 설명해줘.",
    "중학교 사회 교육과정에서 법과 관련된 내용은 몇 학년 때 배우나요?",
]


def build_index(text: str, embed_model: SentenceTransformer):
    chunks = chunk_text(text)
    passages = [f"passage: {c}" for c in chunks]
    emb = embed_model.encode(passages, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(np.asarray(emb, dtype="float32"))
    return index, chunks


def retrieve(question: str, embed_model: SentenceTransformer, index, chunks: list[str], top_k: int = 4):
    q_vec = embed_model.encode([f"query: {question}"], normalize_embeddings=True)
    scores, idxs = index.search(np.asarray(q_vec, dtype="float32"), top_k)
    return [(chunks[i], float(s)) for i, s in zip(idxs[0], scores[0]) if i != -1]


def ask_no_rag(llm: Llama, question: str) -> str:
    result = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "당신은 친절한 중학교 사회 선생님입니다. 학생 눈높이에 맞춰 설명하세요."},
            {"role": "user", "content": question},
        ],
        temperature=0.3, max_tokens=500,
    )
    return result["choices"][0]["message"]["content"]


def ask_with_rag(llm: Llama, question: str, embed_model: SentenceTransformer, index, chunks: list[str]) -> tuple[str, list[tuple[str, float]]]:
    hits = retrieve(question, embed_model, index, chunks)
    context = "\n\n".join(c for c, _ in hits)
    result = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": (
                "당신은 친절한 중학교 사회 선생님입니다. 아래 [교육과정 자료]에 명시된 "
                "학년·단원 범위와 성취기준에 맞춰 학생 눈높이로 설명하세요."
            )},
            {"role": "user", "content": f"[교육과정 자료]\n{context}\n\n[질문]\n{question}"},
        ],
        temperature=0.3, max_tokens=500,
    )
    return result["choices"][0]["message"]["content"], hits


def main() -> None:
    print("PDF 추출 중...")
    data = _CURRICULUM_PDF.read_bytes()
    result = extract_text(data, ".pdf")
    print(f"추출 완료: {len(result.text)}자\n")

    print("임베딩 모델 로딩 + 인덱스 구축 중...")
    embed_model = SentenceTransformer(_EMBED_MODEL)
    index, chunks = build_index(result.text, embed_model)
    print(f"청크 {len(chunks)}개\n")

    llm = Llama(model_path=str(_GGUF_PATH), n_ctx=4096, n_threads=8, verbose=False)

    for question in QUESTIONS:
        print("=" * 78)
        print(f"질문: {question}")
        print("=" * 78)

        print("\n--- ① RAG 없이 (LLM 자체 지식만) ---")
        print(ask_no_rag(llm, question))

        print("\n--- ② RAG 사용 (2022 개정 교육과정 문서 근거) ---")
        answer, hits = ask_with_rag(llm, question, embed_model, index, chunks)
        print("[검색된 근거 청크]")
        for chunk, score in hits:
            print(f"  score={score:.3f}  {chunk[:60].strip()}...")
        print()
        print(answer)
        print()


if __name__ == "__main__":
    main()
