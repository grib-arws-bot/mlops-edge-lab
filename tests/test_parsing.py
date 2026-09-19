"""src/web/parsing.py(AI토론·BidRadar 등이 쓰는 순수 파싱/판정 로직)에 대한 최소
단위 테스트 — 외부 의존성 없음(numpy만 사용). app.py 자체는 llama-cpp-python 등
무거운 의존성을 끌고 와서 CI에서 직접 import할 수 없어, 순수 로직만 이 모듈로
분리해뒀다(tests/test_rules.py·test_quality.py와 같은 목적)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from web.parsing import (
    balanced_cluster_assignment,
    dedupe_requirements,
    parse_opinion_list,
    parse_requirements,
    parse_student_list,
    parse_topic_matches,
    quiz_choice_grounded,
)


# --- quiz_choice_grounded ---------------------------------------------------

def test_grounded_choice_passes():
    context = "산업안전보건법에 따라 화기 작업 시 안전 관리자를 배치해야 한다."
    assert quiz_choice_grounded("화기 작업 시 안전 관리자를 배치해야 한다", context)


def test_fabricated_choice_fails():
    context = "산업안전보건법에 따라 화기 작업 시 안전 관리자를 배치해야 한다."
    assert not quiz_choice_grounded("서울은 인구의 90%, 부산은 70%가 해당한다", context)


def test_choice_with_no_extractable_tokens_passes_by_default():
    # 명사·숫자 토큰이 하나도 없으면(예: 조사·기호뿐) 검증할 게 없어 통과시킨다
    assert quiz_choice_grounded("음", "아무 관련 없는 문장")


# --- parse_student_list / parse_opinion_list --------------------------------

def test_parse_student_list_valid_json():
    raw = '{"students": [{"name": "민준", "opinion": "찬성합니다"}, {"name": "서연", "opinion": "반대합니다"}]}'
    result = parse_student_list(raw)
    assert len(result) == 2
    assert result[0]["name"] == "민준"


def test_parse_student_list_recovers_from_broken_json():
    # 2026-09-19 실측 — 모델이 `"}` 대신 `")`를 써서 배열 전체(json.loads) 파싱이
    # 깨진 실제 사례. 정규식 폴백은 중괄호 짝이 안 맞아도 "name"/"opinion" 텍스트
    # 자체만 온전하면 항목을 그대로 복구한다 — 그래서 이 케이스는 둘 다 살아남는다
    # (실제 15명 케이스에서 일부가 유실된 건 손상 위치가 이름/의견 문자열 안쪽까지
    # 걸쳤던 경우였을 것).
    raw = '{"students": [{"name": "민준", "opinion": "찬성합니다")}, {"name": "서연", "opinion": "반대합니다"}]}'
    result = parse_student_list(raw)
    assert len(result) == 2
    assert result[0]["name"] == "민준"
    assert result[1]["name"] == "서연"


def test_parse_student_list_loses_item_when_quote_itself_is_corrupted():
    # 손상이 값 문자열 안쪽(따옴표 자체)까지 걸치면 그 항목만 실제로 유실된다 —
    # 이게 "학생 한둘이 유실되는 건 허용 오차로 본다"고 한 실제 실패 양상이다.
    raw = '{"students": [{"name": "민준, "opinion": "찬성합니다"}, {"name": "서연", "opinion": "반대합니다"}]}'
    result = parse_student_list(raw)
    assert len(result) == 1
    assert result[0]["name"] == "서연"


def test_parse_opinion_list_valid_json():
    raw = '{"students": [{"opinion": "찬성"}, {"opinion": "반대"}]}'
    assert parse_opinion_list(raw) == ["찬성", "반대"]


def test_parse_opinion_list_recovers_from_broken_json():
    # parse_student_list와 같은 이유로 중괄호 손상은 무시하고 둘 다 복구된다
    raw = '{"students": [{"opinion": "찬성"), {"opinion": "반대"}]}'
    assert parse_opinion_list(raw) == ["찬성", "반대"]


# --- balanced_cluster_assignment --------------------------------------------

def test_balanced_cluster_assignment_group_sizes_are_even():
    rng = np.random.RandomState(1)
    vectors = rng.rand(15, 8)
    assignment = balanced_cluster_assignment(vectors, k=3, seed=0)
    assert len(assignment) == 15
    sizes = sorted(assignment.count(c) for c in range(3))
    assert sizes == [5, 5, 5]  # 15명을 3팀 — 딱 나눠떨어지는 크기까지 강제로 맞춰야 함


def test_balanced_cluster_assignment_uneven_split_stays_within_one():
    rng = np.random.RandomState(2)
    vectors = rng.rand(16, 8)  # 16 / 3 = 5.33 — 나눠떨어지지 않음
    assignment = balanced_cluster_assignment(vectors, k=3, seed=0)
    sizes = sorted(assignment.count(c) for c in range(3))
    assert sizes == [5, 5, 6]  # 최대·최소 차이가 1을 넘으면 안 됨(쏠림 방지가 핵심 목적)


def test_balanced_cluster_assignment_is_deterministic_given_seed():
    rng = np.random.RandomState(3)
    vectors = rng.rand(10, 4)
    a1 = balanced_cluster_assignment(vectors, k=2, seed=42)
    a2 = balanced_cluster_assignment(vectors, k=2, seed=42)
    assert a1 == a2


# --- parse_topic_matches -----------------------------------------------------

def test_parse_topic_matches_valid_json():
    raw = '{"matches": [{"topic_id": 1, "confidence": 0.9, "reason": "관련 있음"}]}'
    result = parse_topic_matches(raw)
    assert result == [{"topic_id": 1, "confidence": 0.9, "reason": "관련 있음"}]


def test_parse_topic_matches_recovers_from_broken_json():
    raw = '{"matches": [{"topic_id": 1, "confidence": 0.9, "reason": "관련 있음")]}'
    result = parse_topic_matches(raw)
    assert result == [{"topic_id": 1, "confidence": 0.9, "reason": "관련 있음"}]


def test_parse_topic_matches_empty_when_no_match():
    assert parse_topic_matches('{"matches": []}') == []


# --- parse_requirements ------------------------------------------------------

def test_parse_requirements_valid_json():
    raw = '{"requirements": [{"category": "성능", "req_text": "3초 이내", "req_value": "3", "req_unit": "초", "op": "lte", "cite": "3초 이내여야 한다"}]}'
    result = parse_requirements(raw)
    assert len(result) == 1
    assert result[0]["req_value"] == "3"


def test_parse_requirements_recovers_from_broken_json():
    raw = '{"requirements": [{"category": "성능", "req_text": "3초 이내", "cite": "3초 이내여야 한다")]}'
    result = parse_requirements(raw)
    assert len(result) == 1
    assert result[0]["cite"] == "3초 이내여야 한다"


# --- dedupe_requirements ------------------------------------------------------

def _req(cite: str, category: str = "성능") -> dict:
    return {"category": category, "req_text": cite, "req_value": "", "req_unit": "", "op": "manual", "cite": cite}


def test_dedupe_requirements_removes_exact_duplicate():
    # 의사결정_로그 95번 실측 사례 재현 — 청크 겹침 구간에 걸친 요구사항이 두 번 추출됨
    reqs = [_req("최근 3년간 유사 사업 수행 실적이 2건 이상이어야 한다.", "실적"),
            _req("ISO 27001 인증을 보유해야 한다.", "인증"),
            _req("최근 3년간 유사 사업 수행 실적이 2건 이상이어야 한다.", "실적")]
    result = dedupe_requirements(reqs)
    assert len(result) == 2


def test_dedupe_requirements_treats_whitespace_variants_as_duplicate():
    # 청크 경계에서 줄바꿈·공백이 살짝 다르게 잘려도 같은 문장으로 봐야 한다
    reqs = [_req("응답속도는 3초 이내여야 한다."),
            _req("응답속도는  3초\n이내여야 한다.")]
    assert len(dedupe_requirements(reqs)) == 1


def test_dedupe_requirements_keeps_genuinely_different_items():
    reqs = [_req("응답속도는 3초 이내여야 한다."), _req("응답속도는 5초 이내여야 한다.")]
    assert len(dedupe_requirements(reqs)) == 2
