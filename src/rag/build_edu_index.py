"""edu/collected/*/items.jsonl 전체를 청킹 → 임베딩 → FAISS 인덱스로 만든다.

안전 도메인의 build_index.py와 완전히 같은 패턴(같은 임베딩 모델·청킹 함수)이지만
소스가 다르고, 완전히 별도 인덱스 파일로 저장한다 — 도메인 섞임 방지 원칙(이번
스마트교육 트랙 전체에서 계속 지켜온 것) 그대로 유지.

**라이선스 참고**: 여기엔 allows_modification=False인 소스(KDI 등)도 포함한다 —
RAG는 원문을 그대로 인용하는 것이라 "변경"이 아니라서 라이선스 문제가 없다(의사결정_로그
79번에서 정리한 구분). 파인튜닝 학습 데이터(make_edu_dataset.py)만 그 플래그로
걸러야 한다.
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
_COLLECTED_DIR = _ROOT / "edu" / "collected"
_INDEX_PATH = _ROOT / "edu" / "processed" / "edu_index.faiss"
_META_PATH = _ROOT / "edu" / "processed" / "edu_chunks.jsonl"


def _load_records() -> list[dict]:
    records: list[dict] = []
    if not _COLLECTED_DIR.exists():
        return records

    for source_dir in sorted(_COLLECTED_DIR.iterdir()):
        items_path = source_dir / "items.jsonl"
        if not items_path.exists():
            continue
        for line in items_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            raw_path = _COLLECTED_DIR / item["raw_path"]
            from extract.core import extract_text

            data = raw_path.read_bytes()
            result = extract_text(data, raw_path.suffix)
            for chunk_id, chunk in enumerate(chunk_text(result.text)):
                records.append({
                    "source_id": item["source_id"],
                    "title": item["title"],
                    "sub_domain": item["sub_domain"],
                    "school_level": item["school_level"],
                    "license_type": item["license_type"],
                    "chunk_id": chunk_id,
                    "text": chunk,
                })
    return records


def main() -> None:
    model = SentenceTransformer(EMBED_MODEL)
    records = _load_records()
    print(f"수집 항목 청크 {len(records)}개")

    if not records:
        raise RuntimeError("edu/collected/에 데이터가 없음 — src/collect/run.py 먼저 실행 필요")

    passages = [f"passage: {r['text']}" for r in records]
    embeddings = model.encode(passages, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

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
