"""cosmetics/sop/*.txt(합성 SOP 문서)를 청킹→임베딩→FAISS 인덱스로 만든다.

기존 rag/build_index.py(산업안전)와 완전히 같은 패턴 — 도메인 섞임 방지 원칙 그대로
유지, 별도 인덱스 파일로 저장한다. concept 검증용이라 문서 4건뿐이지만 파이프라인
자체는 실제 코퍼스가 늘어나도 그대로 쓸 수 있게 동일 구조로 맞춘다.
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from rag.build_index import EMBED_MODEL
from rag.chunk import chunk_text

_ROOT = Path(__file__).resolve().parents[1]
_SOP_DIR = _ROOT / "cosmetics" / "sop"
_INDEX_PATH = _ROOT / "data" / "processed" / "cosmetics_index.faiss"
_META_PATH = _ROOT / "data" / "processed" / "cosmetics_chunks.jsonl"


def main() -> None:
    model = SentenceTransformer(EMBED_MODEL)

    records: list[dict] = []
    for path in sorted(_SOP_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        for i, chunk in enumerate(chunk_text(text, target_size=500, overlap=80)):
            records.append({"source": path.stem, "chunk_id": i, "text": chunk})

    print(f"SOP 문서 {len(list(_SOP_DIR.glob('*.txt')))}건 -> 청크 {len(records)}개")

    passages = [f"passage: {r['text']}" for r in records]
    embeddings = model.encode(passages, batch_size=32, show_progress_bar=True, normalize_embeddings=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(np.asarray(embeddings, dtype="float32"))

    _INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(_INDEX_PATH))
    with _META_PATH.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"인덱스 저장: {_INDEX_PATH} (벡터 {index.ntotal}개, 차원 {dim})")
    print(f"메타데이터 저장: {_META_PATH}")


if __name__ == "__main__":
    main()
