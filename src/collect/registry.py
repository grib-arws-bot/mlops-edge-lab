"""(학교급, 과목/영역) → 어떤 소스를 쓸지 매핑하는 설정.

전 학년·전 과목에 재사용 가능한 체계로 만들라는 요구사항(2026-09-19)을 반영해, 소스를
추가할 때 커넥터 코드를 새로 만들고 여기 한 줄만 추가하면 되는 구조로 잡았다. 지금은
중학교 사회과만 채워져 있다.

각 SourceConfig의 license는 "이 소스 전체에 적용되는 기본 라이선스"다 — 실제로는 항목별로
다를 수 있는 소스(예: 국편 data.go.kr 배포본은 데이터셋마다 공공누리 유형이 다름)는
커넥터가 항목 단위로 다시 판단해서 덮어쓴다. 여기 값은 "커넥터가 항목별 판단을 못 했을 때"의
안전한 기본값 역할도 한다.
"""

from __future__ import annotations

from dataclasses import dataclass

from collect.models import LicenseInfo


@dataclass
class SourceConfig:
    source_id: str
    name: str
    school_levels: list[str]
    subjects: list[str]
    sub_domain: str
    requires_auth: bool
    license: LicenseInfo
    notes: str
    implemented: bool
    """False면 아직 커넥터 코드가 없음(예: API 키 승인 대기 중) — registry에는 문서화
    목적으로 미리 등록해두되, run.py가 건너뛴다."""


SOURCES: list[SourceConfig] = [
    SourceConfig(
        source_id="nationalatlas_youth",
        name="대한민국 국가지도집 청소년판(국토지리정보원)",
        school_levels=["중학교"],
        subjects=["사회"],
        sub_domain="지리",
        requires_auth=False,
        license=LicenseInfo(
            license_type="공공누리 제1유형(출처표시)",
            allows_modification=True,
            allows_commercial=True,
            source_policy_note="data.go.kr 15066387 등재 기준(2026-09-19 리서치 확인). robots.txt User-agent:* Allow:/ — 전면 허용.",
        ),
        notes="HTTPS 인증서가 다른 호스트(stat.ngii.go.kr) 것이라 불일치 — HTTP로만 접근. 2022년판(kor_hi_2022) PDF 7종.",
        implemented=True,
    ),
    SourceConfig(
        source_id="law_go_kr",
        name="국가법령정보센터 Open API",
        school_levels=["중학교", "고등학교"],
        subjects=["사회"],
        sub_domain="법",
        requires_auth=True,
        license=LicenseInfo(
            license_type="자유이용(공공데이터법 근거)",
            allows_modification=True,
            allows_commercial=True,
            source_policy_note="open.law.go.kr 법적효력/저작권 안내: '영리 목적 포함 자유로운 활용 보장'(위·변조 금지). robots.txt 전면 허용.",
        ),
        notes="OC(이메일 ID 기반 인증값) 발급 필요 — 회원가입 후 API 활용신청, 담당자 심의 1~2일. 인증키 발급 전까지 implemented=False.",
        implemented=False,
    ),
    SourceConfig(
        source_id="easylaw_go_kr",
        name="찾기쉬운 생활법령정보 Open API",
        school_levels=["중학교", "고등학교"],
        subjects=["사회"],
        sub_domain="법",
        requires_auth=True,
        license=LicenseInfo(
            license_type="공공누리 제1유형(출처표시)",
            allows_modification=True,
            allows_commercial=True,
            source_policy_note="data.go.kr 15000215 등재 기준. 중학생 수준 해설형 텍스트 — 법령 원문보다 적합.",
        ),
        notes="서비스키 인증, 개발계정 일 100건 제한. 인증키 발급 전까지 implemented=False.",
        implemented=False,
    ),
    SourceConfig(
        source_id="fss_edu",
        name="금융감독원 금융교육 Open API",
        school_levels=["중학교"],
        subjects=["사회"],
        sub_domain="경제",
        requires_auth=True,
        license=LicenseInfo(
            license_type="확인 못 함",
            allows_modification=None,
            allows_commercial=None,
            source_policy_note="응답에 항목별 cpyrhtPermCode(저작권 허용 코드)가 포함됨 — 실제 수집 시 항목별로 재판단 필요. eduTrgtCntnt=H(청소년기) 필터 가능.",
        ),
        notes="authKey(32자리) 발급 필요. 인증키 발급 전까지 implemented=False.",
        implemented=False,
    ),
    SourceConfig(
        source_id="ecos_bok",
        name="한국은행 ECOS OpenAPI",
        school_levels=["중학교", "고등학교"],
        subjects=["사회"],
        sub_domain="경제",
        requires_auth=True,
        license=LicenseInfo(
            license_type="확인 못 함(실호출 성공만 확인)",
            allows_modification=None,
            allows_commercial=None,
            source_policy_note="통계 수치 위주 — 설명문이 아니라 근거 수치용. robots.txt 전면 허용.",
        ),
        notes="인증키 회원가입 후 즉시 자동발급. 인증키 발급 전까지 implemented=False.",
        implemented=False,
    ),
    SourceConfig(
        source_id="kdi_econ_edu",
        name="KDI 경제정보센터 교육과정별 경제교육",
        school_levels=["중학교"],
        subjects=["사회"],
        sub_domain="경제",
        requires_auth=False,
        license=LicenseInfo(
            license_type="공공누리 제3유형(출처표시+변경금지)",
            allows_modification=False,
            allows_commercial=None,
            source_policy_note="eiec.kdi.re.kr '교육과정별 경제교육' 62건 PDF, 학교급 필터 있음. 변경금지 — 파인튜닝 가공 대상에서 반드시 제외, RAG 원문 인용에만 사용.",
        ),
        notes="아직 커넥터 미구현.",
        implemented=False,
    ),
]


def sources_for(school_level: str, subject: str) -> list[SourceConfig]:
    return [
        s for s in SOURCES
        if school_level in s.school_levels and subject in s.subjects
    ]
