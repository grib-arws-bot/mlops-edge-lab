"""rag/chunk.py에 대한 최소 단위 테스트 — 외부 의존성 없음.

줄바꿈 없는 긴 문서가 크기 제한 없이 청크 1개로 뭉치던 실제 버그(2026-09-20
발견, BidRadar 병렬 풀 작업 중 "의도적으로 미뤄둠"으로 기록됨, 2026-09-23 수정)
재발 방지가 핵심 목적."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.chunk import chunk_text


def test_short_text_single_chunk():
    assert chunk_text("짧은 문장 하나.") == ["짧은 문장 하나."]


def test_paragraphs_split_on_newline():
    text = "\n".join(["문단 " + str(i) * 50 for i in range(6)])
    chunks = chunk_text(text, target_size=200)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 250  # overlap 포함 약간의 여유


def test_no_newline_long_text_still_bounded():
    # 줄바꿈이 전혀 없는 문서(HTML 추출 등) — 예전엔 문단이 하나로 뭉쳐 크기
    # 제한 없이 그대로 청크가 됐다. 문장 부호로 나뉘는지 확인한다.
    sentence = "이것은 문장입니다. "
    text = (sentence * 100).strip()  # 줄바꿈 없이 약 2000자
    assert "\n" not in text
    chunks = chunk_text(text, target_size=200)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 250


def test_no_newline_no_punctuation_hard_sliced():
    # 문장 부호조차 없는 극단적인 경우(예: 순수 숫자·코드 덤프) — 고정 길이로
    # 강제 절단되어 여전히 크기 제한을 지키는지 확인한다.
    text = "가" * 2000
    chunks = chunk_text(text, target_size=200, overlap=0)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 200
