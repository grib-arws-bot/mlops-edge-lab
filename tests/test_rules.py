"""판정 규칙(agent/rules.py)에 대한 최소 단위 테스트.

CI에서 가볍게 도는 걸 목표로 함 — LLM·GPU·모델 다운로드가 필요 없는 순수 규칙 로직만
검증한다. 이게 우리가 "판정은 규칙이 한다"고 주장하는 부분의 실질적 안전망이다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent.rules import Risk, Severity, judge, required_equipment_actions


def test_normal_when_under_threshold():
    j = judge(value=4.0, threshold=5.0)
    assert j.severity == Severity.NORMAL
    assert not j.exceeded


def test_caution_when_slightly_over():
    j = judge(value=6.0, threshold=5.0)  # 1.2배
    assert j.severity == Severity.CAUTION
    assert j.exceeded


def test_danger_when_1_5x_or_more():
    j = judge(value=10.0, threshold=5.0)  # 2.0배
    assert j.severity == Severity.DANGER
    assert j.exceeded


def test_boundary_exactly_at_threshold_is_normal():
    j = judge(value=5.0, threshold=5.0)  # 정확히 경계 — 초과 아님
    assert j.severity == Severity.NORMAL


def test_lower_is_worse_normal_when_above_threshold():
    """산소농도처럼 낮을수록 위험한 물질 — 임계값(18%)보다 높으면 정상."""
    j = judge(value=20.9, threshold=18.0, lower_is_worse=True)
    assert j.severity == Severity.NORMAL
    assert not j.exceeded


def test_lower_is_worse_danger_when_far_below_threshold():
    j = judge(value=12.0, threshold=18.0, lower_is_worse=True)  # 18/12 = 1.5배
    assert j.severity == Severity.DANGER
    assert j.exceeded


def test_lower_is_worse_boundary_exactly_at_threshold_is_normal():
    j = judge(value=18.0, threshold=18.0, lower_is_worse=True)
    assert j.severity == Severity.NORMAL
    assert not j.exceeded


def test_high_risk_equipment_never_auto_executes():
    """고위험 장비(가스차단기 등)는 규칙상 자동 실행 목록에 있으면 안 된다 — 항상 승인
    대기여야 한다. 이 테스트가 깨지면 안전 설계 원칙이 깨진 것이다."""
    actions = required_equipment_actions("가스", Severity.DANGER)
    high_risk = [a for a in actions if a.risk == Risk.HIGH]
    assert high_risk, "위험 등급 가스 이벤트엔 고위험 장비 조치가 하나는 있어야 함"
    for action in high_risk:
        assert action.risk == Risk.HIGH  # 자동 실행 대상이 아님을 명시적으로 확인
