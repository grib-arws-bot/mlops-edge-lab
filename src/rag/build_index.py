"""data/processed/text/*.txt 전체를 청킹 → 임베딩 → FAISS 인덱스로 만든다.

임베딩 모델 선택 이유(docs/의사결정_로그.md 26번): 질문 임베딩은 엣지 디바이스에서 매번
실시간으로 계산해야 해서 가벼워야 한다. `intfloat/multilingual-e5-small`(~470MB, 한국어
지원)로 시작 — 검색 품질이 부족하면 더 큰 모델(BAAI/bge-m3)로 교체할 수 있게 설계한다.

E5 계열은 질문/문서에 각각 "query: " / "passage: " 접두어를 붙여야 성능이 제대로 나온다
(모델 카드 권장 사항).
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from rag.chunk import chunk_text

EMBED_MODEL = "intfloat/multilingual-e5-small"
_ROOT = Path(__file__).resolve().parents[2]
_TEXT_DIR = _ROOT / "data" / "processed" / "text"
_INDEX_PATH = _ROOT / "data" / "processed" / "rag_index.faiss"
_META_PATH = _ROOT / "data" / "processed" / "rag_chunks.jsonl"


def main() -> None:
    model = SentenceTransformer(EMBED_MODEL)

    records: list[dict] = []
    for path in sorted(_TEXT_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        for i, chunk in enumerate(chunk_text(text)):
            records.append({"source": path.stem, "chunk_id": i, "text": chunk})

    print(f"문서 {len(list(_TEXT_DIR.glob('*.txt')))}건 -> 청크 {len(records)}개")

    passages = [f"passage: {r['text']}" for r in records]
    embeddings = model.encode(
        passages, batch_size=64, show_progress_bar=True, normalize_embeddings=True
    )

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # 정규화된 벡터의 내적 = 코사인 유사도
    index.add(np.asarray(embeddings, dtype="float32"))

    faiss.write_index(index, str(_INDEX_PATH))
    with _META_PATH.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"인덱스 저장: {_INDEX_PATH} (벡터 {index.ntotal}개, 차원 {dim})")
    print(f"메타데이터 저장: {_META_PATH}")


if __name__ == "__main__":
    main()
