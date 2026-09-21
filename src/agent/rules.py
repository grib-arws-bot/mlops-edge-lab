"""임계값 초과·위험도 판정 — 규칙(결정론), LLM이 하지 않는다.

BidRadar의 핵심 원칙(app/services/analysis/match.py, CLAUDE.md)을 그대로 가져온다:
"충족 판정은 LLM이 하지 않는다. LLM은 문구 작성·검색어 결정처럼 저위험 판단만 한다."
이 프로젝트도 로드맵 3번 평가에서 LLM이 "정상 범위인데 초과로 판단"하는 오류를 실제로
봤다(docs/의사결정_로그.md 20번) — 그 문제를 여기서 규칙으로 대체해 해결한다.

⚠️ 위험도 구간(1.5배 등)은 placeholder다 — 실제 서비스 전 정식 기준으로 교체해야 한다
(docs/의사결정_로그.md 18번 경고와 동일한 맥락).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Severity(str, Enum):
    NORMAL = "정상"
    CAUTION = "주의"  # 임계값은 넘었지만 1.5배 미만
    DANGER = "위험"  # 임계값의 1.5배 이상


@dataclass
class Judgement:
    exceeded: bool
    severity: Severity
    ratio: float  # 측정값 / 임계값


def judge(value: float, threshold: float, lower_is_worse: bool = False) -> Judgement:
    """lower_is_worse=True면 값이 threshold보다 낮을수록 위험하다고 판정한다(예: 산소농도 —
    18% 미만이면 산소결핍, sensors.py의 LOWER_IS_WORSE 참고). ratio 계산 방향만 뒤집고
    1.0/1.5 경계는 그대로 재사용 — 기존 물질(높을수록 위험)과 판정 로직을 통일해서 유지한다."""
    if lower_is_worse:
        ratio = threshold / value if value else float("inf")
    else:
        ratio = value / threshold if threshold else float("inf")
    if ratio <= 1.0:
        return Judgement(exceeded=False, severity=Severity.NORMAL, ratio=ratio)
    if ratio < 1.5:
        return Judgement(exceeded=True, severity=Severity.CAUTION, ratio=ratio)
    return Judgement(exceeded=True, severity=Severity.DANGER, ratio=ratio)


class Risk(str, Enum):
    LOW = "낮음"  # 되돌리기 쉬움 -> 자동 실행
    HIGH = "높음"  # 오작동 시 비용/피해가 큼 -> 사람 승인 필요, 자동 실행 안 함


@dataclass
class EquipmentAction:
    name: str
    risk: Risk
    description: str


# 카테고리·위험도별로 어떤 장비를 건드릴지 — 이것도 규칙이다. LLM은 "이 장비를 켤지 말지"를
# 절대 스스로 판단하지 않는다(잘못 판단하면 실제 설비가 오작동하는 것이라 문구 오류보다 훨씬
# 위험하다). ⚠️ 실제 설비 종류·매핑은 placeholder다 — 실서비스 전 현장 안전 담당자 검토 필요.
_EQUIPMENT_RULES: dict[tuple[str, Severity], list[EquipmentAction]] = {
    ("가스", Severity.CAUTION): [
        EquipmentAction("환풍기", Risk.LOW, "환기 가동"),
    ],
    ("가스", Severity.DANGER): [
        EquipmentAction("환풍기", Risk.LOW, "환기 가동"),
        EquipmentAction("가스차단기", Risk.HIGH, "가스 공급 차단"),
    ],
    ("조리흄", Severity.CAUTION): [
        EquipmentAction("환풍기", Risk.LOW, "환기 가동"),
    ],
    ("조리흄", Severity.DANGER): [
        EquipmentAction("환풍기", Risk.LOW, "환기 가동"),
        EquipmentAction("소화설비(대기)", Risk.LOW, "대기 상태로 전환(분사 아님)"),
        EquipmentAction("소화설비(방출)", Risk.HIGH, "소화약제 방출"),
    ],
    ("공기질", Severity.CAUTION): [
        EquipmentAction("환풍기", Risk.LOW, "환기 가동"),
    ],
    ("공기질", Severity.DANGER): [
        EquipmentAction("환풍기", Risk.LOW, "환기 가동"),
    ],
}


def required_equipment_actions(category: str, severity: Severity) -> list[EquipmentAction]:
    return _EQUIPMENT_RULES.get((category, severity), [])


def equipment_for_category(category: str) -> list[str]:
    """이 카테고리가 어떤 위험도에서든 건드릴 수 있는 장비 전체 목록(중복 제거, 등장 순서
    유지) — 정상 상태 화면에 "이 현장이 갖춘 장비들은 전부 꺼져 있다"를 보여줄 때 쓴다
    (_EQUIPMENT_RULES에는 NORMAL 등급 항목이 아예 없어서 이렇게 위험도 전체를 합쳐야 한다)."""
    names: list[str] = []
    for (cat, _severity), actions in _EQUIPMENT_RULES.items():
        if cat != category:
            continue
        for action in actions:
            if action.name not in names:
                names.append(action.name)
    return names
