"""수집 데이터의 공통 자료구조 — 모든 커넥터(법령/지도/통계 등)가 이 형태로 반환한다.

핵심은 `LicenseInfo.allows_modification`이다. 오늘 소스 리서치에서 KDI 경제정보센터가
공공누리 3유형(변경금지)이라는 걸 실제로 확인했다 — 이런 소스는 원문 그대로 RAG 인용에는
써도, 파인튜닝용으로 LLM이 다시 쓰는(=변경하는) 건 라이선스 위반이다. 운영 콘솔에서
"이 소스는 학습에 못 쓴다"를 자동으로 구분하려면, 수집 시점에 이 정보를 같이 저장해둬야
나중에 사람이 소스별 약관을 다시 뒤져보지 않아도 된다.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LicenseInfo:
    license_type: str
    """예: "공공누리 제1유형", "자유이용(공공데이터법 근거)", "확인 못 함"."""

    allows_modification: bool | None
    """파인튜닝 데이터로 가공(=원문 변경)해도 되는지. None이면 아직 확인 안 됨 —
    확인 전까지는 파인튜닝 파이프라인이 이 소스를 자동으로 제외해야 한다(안전 기본값)."""

    allows_commercial: bool | None
    source_policy_note: str
    """어디서 이 판단을 내렸는지 원문 근거(약관 문구·robots.txt 등) — 나중에 재검증할 때 필요."""


@dataclass
class CollectedItem:
    source_id: str
    school_level: str
    subject: str
    sub_domain: str
    title: str
    source_url: str
    fetch_method: str
    """"http_download" | "api" — 스크래핑은 의도적으로 없음(robots.txt 차단 소스는 커넥터를 아예 안 만듦)."""

    fetched_at: str
    license: LicenseInfo
    raw_path: str
    """저장된 원문 파일의 edu/collected/ 기준 상대경로."""

    text_char_count: int
    achievement_standard_tags: list[str] = field(default_factory=list)
    """성취기준 코드 매핑 — 수집 시점엔 대부분 비어있고, 후처리 분류 단계에서 채워진다
    (오늘 리서치에서 확인: 어떤 소스도 성취기준과 직접 매핑되는 구조가 아니었음)."""

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "school_level": self.school_level,
            "subject": self.subject,
            "sub_domain": self.sub_domain,
            "title": self.title,
            "source_url": self.source_url,
            "fetch_method": self.fetch_method,
            "fetched_at": self.fetched_at,
            "license_type": self.license.license_type,
            "allows_modification": self.license.allows_modification,
            "allows_commercial": self.license.allows_commercial,
            "license_note": self.license.source_policy_note,
            "raw_path": self.raw_path,
            "text_char_count": self.text_char_count,
            "achievement_standard_tags": self.achievement_standard_tags,
        }
