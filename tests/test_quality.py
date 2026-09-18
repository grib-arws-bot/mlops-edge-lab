"""추출 품질 신호(src/extract/quality.py)에 대한 최소 단위 테스트 — 외부 의존성 없음."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from extract.quality import assess


def test_clean_korean_text_no_flags():
    text = "산업안전보건법에 따라 화기 작업 시 안전 관리자를 배치해야 한다. " * 5
    signals = assess(text)
    assert not signals.needs_review


def test_empty_text_flagged():
    signals = assess("")
    assert signals.needs_review
    assert "빈 텍스트" in signals.flags


def test_repeated_garbage_lines_flagged():
    text = "\n".join(["오류 오류 오류"] * 20 + ["실제 내용 한 줄"])
    signals = assess(text)
    assert signals.needs_review
    assert any("반복" in f for f in signals.flags)


def test_low_korean_ratio_flagged():
    text = "abcdefghij" * 20  # 한글이 전혀 없는 텍스트
    signals = assess(text)
    assert signals.needs_review
    assert any("한글 비율" in f for f in signals.flags)


def test_replacement_characters_flagged():
    text = ("정상적인 한국어 문장입니다. " * 10) + ("�" * 30)
    signals = assess(text)
    assert signals.needs_review
    assert any("대체" in f for f in signals.flags)
