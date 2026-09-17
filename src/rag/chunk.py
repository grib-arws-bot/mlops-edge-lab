"""추출된 텍스트(data/processed/text/*.txt)를 검색 단위 조각(chunk)으로 나눈다.

너무 잘게 자르면 문맥이 끊기고, 너무 크게 자르면 검색 정확도가 떨어진다 — 500~800자를
목표 크기로 하고, 조각 경계에서 맥락이 뚝 끊기지 않게 약간의 겹침(overlap)을 둔다.
"""

from __future__ import annotations

_TARGET_SIZE = 700
_OVERLAP = 100


def chunk_text(text: str, target_size: int = _TARGET_SIZE, overlap: int = _OVERLAP) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if current and len(current) + len(para) > target_size:
            chunks.append(current)
            current = current[-overlap:] + "\n" + para
        else:
            current = f"{current}\n{para}" if current else para
    if current.strip():
        chunks.append(current)
    return chunks
