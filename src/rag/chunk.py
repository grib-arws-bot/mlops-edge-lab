"""추출된 텍스트(data/processed/text/*.txt)를 검색 단위 조각(chunk)으로 나눈다.

너무 잘게 자르면 문맥이 끊기고, 너무 크게 자르면 검색 정확도가 떨어진다 — 500~800자를
목표 크기로 하고, 조각 경계에서 맥락이 뚝 끊기지 않게 약간의 겹침(overlap)을 둔다.
"""

from __future__ import annotations

import re

_TARGET_SIZE = 700
_OVERLAP = 100

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_SECTION_MARKER = re.compile(r"^\[\d+\.\s")


def _group_by_section_markers(paragraphs: list[str]) -> list[str]:
    """`[N. 제목]` 같은 구획 마커로 시작하는 줄을 만나면 새 블록을 연다 — MSDS처럼
    문서 안에 명확한 항목 구분이 있는 텍스트(collect/msds_hf_ingest.py가 쓰는
    포맷)에서, 그 구획 하나를 쪼개지지 않는 최소 단위로 다뤄야 "[4. 응급조치
    요령]"처럼 제목만 있고 본문은 다음 청크로 넘어가버리는 식으로 잘리지 않는다
    (사용자 실측, 2026-09-23: 실제 검색 결과가 항목 제목 직후에서 끊겨 나오는 걸
    확인함 — 줄바꿈 기준 고정 길이 패킹만으로는 구획 경계를 모른다)."""
    blocks: list[str] = []
    current: list[str] = []
    for p in paragraphs:
        if _SECTION_MARKER.match(p) and current:
            blocks.append("\n".join(current))
            current = [p]
        else:
            current.append(p)
    if current:
        blocks.append("\n".join(current))
    return blocks


def _split_long_paragraph(para: str, target_size: int) -> list[str]:
    """target_size를 넘는 문단 하나를 문장 경계로, 그마저 없으면 고정 길이로 쪼갠다
    — 반환하는 조각은 항상 target_size 이하임을 보장한다."""
    if len(para) <= target_size:
        return [para]
    pieces = [s.strip() for s in _SENTENCE_SPLIT.split(para) if s.strip()] or [para]
    result: list[str] = []
    for piece in pieces:
        if len(piece) <= target_size:
            result.append(piece)
        else:
            result.extend(piece[i:i + target_size] for i in range(0, len(piece), target_size))
    return result


def chunk_text(text: str, target_size: int = _TARGET_SIZE, overlap: int = _OVERLAP) -> list[str]:
    # 줄바꿈(\n) 기준으로만 문단을 나눴더니, 줄바꿈이 아예 없는 문서(예: HTML에서
    # 추출한 한 덩어리 텍스트)가 들어오면 문단이 하나로 뭉쳐 크기 제한 없이 그대로
    # 청크가 됐다(2026-09-20 발견, 의도적으로 미뤄둔 버그) — 문단 단위로 나눈 뒤에도
    # target_size를 넘는 조각은 문장 경계로 한 번 더 쪼갠다.
    raw_paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    units = _group_by_section_markers(raw_paragraphs) if any(_SECTION_MARKER.match(p) for p in raw_paragraphs) else raw_paragraphs

    paragraphs: list[str] = []
    for p in units:
        paragraphs.extend(_split_long_paragraph(p, target_size))

    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if current and len(current) + len(para) > target_size:
            chunks.append(current)
            # 파이썬 슬라이싱 함정: overlap=0이면 current[-0:]는 빈 문자열이 아니라
            # current[0:](전체)를 가리킨다 — "겹침 없음"을 요청해도 직전 청크
            # 전체가 그대로 이어붙는 버그가 될 뻔했다(테스트로 발견).
            tail = current[-overlap:] if overlap > 0 else ""
            current = f"{tail}\n{para}" if tail else para
        else:
            current = f"{current}\n{para}" if current else para
    if current.strip():
        chunks.append(current)
    return chunks
