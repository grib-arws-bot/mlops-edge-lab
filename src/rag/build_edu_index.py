"""edu/collected/*/items.jsonl 전체를 청킹 → 임베딩(FAISS) + 키워드(BM25) 이중 인덱스로 만든다.

안전 도메인의 build_index.py와 완전히 같은 패턴(같은 임베딩 모델·청킹 함수)이지만
소스가 다르고, 완전히 별도 인덱스 파일로 저장한다 — 도메인 섞임 방지 원칙(이번
스마트교육 트랙 전체에서 계속 지켜온 것) 그대로 유지.

**하이브리드 검색(BM25+임베딩) 추가 이유(2026-09-19)**: "부산의 인구에 대해 알려줘"
질문에서 임베딩 단독 검색이 무관한 경제 자료(0.87)를 진짜 인구 통계 자료(0.858)보다
높게 매긴 실패를 실제로 겪었다 — "인구"라는 짧은 핵심어가 700자 청크 임베딩에서
희석되는 게 원인. BM25(어휘 정확매칭)는 이 문제에 구조적으로 강하다 — "인구"가 실제로
그 청크에 있는지를 직접 본다. 한국어는 조사가 어절에 붙어 공백 토큰화가 무의미해서
`kiwipiepy`로 형태소 분석 후 토큰화한다.

**라이선스 참고**: 여기엔 allows_modification=False인 소스(KDI 등)도 포함한다 —
RAG는 원문을 그대로 인용하는 것이라 "변경"이 아니라서 라이선스 문제가 없다(의사결정_로그
79번에서 정리한 구분). 파인튜닝 학습 데이터(make_edu_dataset.py)만 그 플래그로
걸러야 한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import bm25s
import faiss
import numpy as np
from kiwipiepy import Kiwi
from sentence_transformers import SentenceTransformer

from rag.chunk import chunk_text

EMBED_MODEL = "intfloat/multilingual-e5-small"
_ROOT = Path(__file__).resolve().parents[2]
_COLLECTED_DIR = _ROOT / "edu" / "collected"
_INDEX_PATH = _ROOT / "edu" / "processed" / "edu_index.faiss"
_META_PATH = _ROOT / "edu" / "processed" / "edu_chunks.jsonl"
_BM25_INDEX_DIR = _ROOT / "edu" / "processed" / "edu_bm25"


_CONTENT_TAGS = {
    "NNG", "NNP", "NNB",  # 명사(일반/고유/의존)
    "VV", "VA",  # 동사/형용사
    "MAG",  # 부사
    "XR",  # 어근
    "SL", "SH", "SN",  # 외국어/한자/숫자
}


def korean_tokenize(texts: list[str]) -> list[list[str]]:
    """bm25s가 요구하는 '토큰 리스트의 리스트' 형태로 형태소 분석한다 — 인덱스 빌드와
    질의 시점(query_edu.py) 둘 다 이 함수를 그대로 써야 토큰화 기준이 어긋나지 않는다.

    조사(JX/JKO/JKG/JKB 등)·어미(EF/EC/EP 등)·구두점(SF 등)은 제외하고 명사·동사·
    형용사·부사 등 내용어만 남긴다(2026-09-19, 실측 확인) — 조사는 거의 모든 한국어
    문장에 붙어서 BM25 점수를 오염시킨다. 실제로 "부산의 인구에 대해 알려줘"(코퍼스에
    실존하는 내용)의 BM25 최고점(4.49)이 "세종대왕은 한글을 언제 만들었나요"(코퍼스에
    없는 역사 질문)의 최고점(6.64)보다 낮게 나오는 걸 확인했다 — 조사·어미가 공통으로
    걸리며 점수에 노이즈를 더한 게 원인. 내용어만 남기면 이 잡음이 사라진다."""
    kiwi = Kiwi()
    return [
        [t.form for t in kiwi.tokenize(text) if t.tag in _CONTENT_TAGS]
        for text in texts
    ]


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

    print("BM25 인덱스 빌드 중(형태소 분석)...")
    tokenized = korean_tokenize([r["text"] for r in records])
    bm25 = bm25s.BM25()
    bm25.index(tokenized)
    bm25.save(str(_BM25_INDEX_DIR))
    print(f"BM25 인덱스 저장: {_BM25_INDEX_DIR}")


if __name__ == "__main__":
    main()
