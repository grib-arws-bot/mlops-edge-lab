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


def test_section_markers_never_split_title_from_body():
    # 실사용 버그(2026-09-23): MSDS처럼 "[N. 제목]\n본문" 구조인 문서를 줄바꿈
    # 기준 고정 길이로만 패킹하면 "[4. 응급조치 요령]"이라는 제목만 있고 본문은
    # 다음 청크로 넘어가는 식으로 잘려서 검색 결과가 "짤려 보이는" 문제가 실제
    # 사용자 검색에서 확인됐다. 짧은 구획(전체가 target_size 이내)은 항상 제목+
    # 본문이 같은 청크에 붙어 있어야 한다.
    text = "\n".join([
        "물질명: 테스트물질",
        "[1. 화학제품과 회사에 관한 정보]",
        "제품명: 테스트물질",
        "[2. 유해성·위험성]",
        "급성 독성이 있음",
        "[3. 구성성분의 명칭 및 함유량]",
        "물질명: 테스트물질 100%",
    ])
    chunks = chunk_text(text, target_size=60, overlap=0)
    # 각 구획(제목+본문)이 어느 청크에 있든, "제목만 있고 본문이 없는" 청크가
    # 없어야 한다 — 제목 다음 줄(본문)이 항상 같은 청크 안에 있는지로 검증.
    for title, body in [("[1. 화학제품과 회사에 관한 정보]", "제품명: 테스트물질"),
                         ("[2. 유해성·위험성]", "급성 독성이 있음"),
                         ("[3. 구성성분의 명칭 및 함유량]", "물질명: 테스트물질 100%")]:
        owner = next(c for c in chunks if title in c)
        assert body in owner, f"{title!r}의 본문이 다른 청크로 잘려나감: {chunks!r}"


def test_no_section_markers_unaffected():
    # 마커가 아예 없는 일반 산업안전 문서는 기존 줄바꿈 기준 패킹 그대로 동작해야
    # 한다(회귀 방지 — 이 변경이 MSDS 외 문서에 영향을 주면 안 됨).
    text = "\n".join(["문단 " + str(i) * 50 for i in range(6)])
    chunks = chunk_text(text, target_size=200)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 250
