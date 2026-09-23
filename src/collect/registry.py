"""(학교급, 과목/영역) → 어떤 소스를 쓸지 매핑하는 설정.

전 학년·전 과목에 재사용 가능한 체계로 만들라는 요구사항(2026-09-19)을 반영해, 소스를
추가할 때 커넥터 코드를 새로 만들고 여기 한 줄만 추가하면 되는 구조로 잡았다. 지금은
중학교 사회과만 채워져 있다.

각 SourceConfig의 license는 "이 소스 전체에 적용되는 기본 라이선스"다 — 실제로는 항목별로
다를 수 있는 소스(예: 국편 data.go.kr 배포본은 데이터셋마다 공공누리 유형이 다름)는
커넥터가 항목 단위로 다시 판단해서 덮어쓴다. 여기 값은 "커넥터가 항목별 판단을 못 했을 때"의
안전한 기본값 역할도 한다.

**data.go.kr(공공데이터포털) 인증키 메모(2026-09-19 리서치)**: 계정 하나로 인증키
(Encoding/Decoding 한 쌍)는 공유되지만, API별 "활용신청" 승인은 각각 따로 받아야 한다
— 키 재사용은 되고 신청 절차 통합은 안 된다. 상세페이지의 `API 유형`이 REST/SOAP가
아니라 LINK인 항목(한국은행·법제처 일부)은 실제 호출이 기관 자체 사이트로 넘어가서
기관 직접 가입을 못 피할 가능성이 큼 — 실제 신청해서 확인 전까지는 추정.
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
        notes=(
            "2026-09-19 공식 가이드 문서(사용자 제공, edu/raw/OpenAPI/) 직접 읽고 확인 — "
            "REST 아니라 SOAP. 엔드포인트 http://www.easylaw.go.kr/OPENAPI/soap/LifeLawInfoService "
            "(WSDL: 같은 URL+?wsdl). 인증은 SOAP Header의 ServiceKey 필드(공공데이터포털 발급키). "
            "12개 오퍼레이션이 계층형으로 연결됨: getLifeClassList(생활분류 17종, 코드 고정 — 아동청소년/"
            "교육=2, 근로/노동=12, 소비자=8, 금융/금전=4 등 사회과 관련 코드 있음) → getLifeAreaList "
            "(csmAstSeq로 조회) → getLifeInterrestRuleAreaClassList(csmSeq) → "
            "getLifeInterrestRuleSummaryItem(ccfNo) 순으로 파고들어야 실제 콘텐츠(관심규정 개요)에 도달. "
            "그 외 getLifeLawsInterpretList(법령해석례)·getLifeAskNoticeList(주요궁금사항) 등 8개 추가 "
            "오퍼레이션 있음. SOAP 클라이언트라 실제 ServiceKey로 응답 스키마를 검증하기 전엔 구현해도 "
            "신뢰 못 함(폴트 처리가 조용히 틀릴 수 있음) — 인증키 발급 후 구현 예정."
        ),
        implemented=False,
    ),
    SourceConfig(
        source_id="easylaw_qna100_soap",
        name="법제처 생활법령 백문백답 서비스(SOAP)",
        school_levels=["중학교", "고등학교"],
        subjects=["사회"],
        sub_domain="법",
        requires_auth=True,
        license=LicenseInfo(
            license_type="공공누리 제1유형(출처표시)",
            allows_modification=True,
            allows_commercial=True,
            source_policy_note="data.go.kr 15000335. 내집마련·금전거래·소비자보호·근로 등 Q&A — instruction 학습셋에 가장 잘 맞는 형태.",
        ),
        notes=(
            "2026-09-19 공식 가이드 문서(사용자 제공) 확인 — easylaw_go_kr과 같은 플랫폼, "
            "엔드포인트만 다름: http://www.easylaw.go.kr/OPENAPI/soap/ManyAskManyAnswerService "
            "(WSDL: 같은 URL+?wsdl). SOAP Header ServiceKey 인증, 오퍼레이션명도 "
            "getMaskMAnswerClassList/getMaskMAnswerAreaList 식으로 easylaw_go_kr과 동일 계층 구조 "
            "패턴. 인증키 발급 후 easylaw_go_kr과 함께 구현 예정 — 코드 상당 부분(SOAP 클라이언트, "
            "계층 탐색 로직) 재사용 가능. 개발·운영 단계 모두 심의승인 필요(자동승인 아님) — 승인 대기 예상."
        ),
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
        notes="authKey(32자리) 발급 필요. 2026-09-19 재조사 결과 data.go.kr 등재분은 대부분 DART 기업공시(84건)라 중학생 경제교육용 가치 낮음으로 판명 — 우선순위 하향, 구현 보류.",
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
        notes="API 없음, 사이트 직접 다운로드(robots 허용). WAF가 브라우저 흉내 User-Agent 없으면 차단 — 커넥터에서 처리함. ⚠️ 변경금지 라이선스 — RAG 인용에만 사용, 파인튜닝 가공 절대 금지.",
        implemented=True,
    ),
    SourceConfig(
        source_id="kosis_desc",
        name="국가데이터처(구 통계청) KOSIS 통계표설명·지표명별 설명자료·챗봇 학습데이터",
        school_levels=["중학교", "고등학교"],
        subjects=["사회"],
        sub_domain="지리/경제",
        requires_auth=True,
        license=LicenseInfo(
            license_type="공공누리 제1유형 또는 제한없음",
            allows_modification=True,
            allows_commercial=True,
            source_policy_note="data.go.kr 등재, 2026-09-19 리서치 확인. 수치가 아니라 설명문/용어정의 — 지리·경제 지표의 '무슨 뜻인가'를 채워줌.",
        ),
        notes=(
            "data.go.kr 활용신청, 개발·운영 단계 모두 자동승인 확인. 2026-09-21: BidRadar 프로젝트가 "
            "이미 승인받아 쓰던 계정의 인증키를 재사용하기로 사용자 결정(서로 다른 프로젝트지만 "
            "동일인 소유 data.go.kr 계정, CLAUDE.md의 '인프라 분리' 원칙과 별개로 키 재사용만 허용된 "
            "케이스 — 의사결정_로그 참고). 키는 `grib-ai-server`의 "
            "`~/secrets/mlops-edge-lab/education_api_keys.env`(저장소 밖, git 미추적)에 "
            "`MLOPS_EDU_KOSIS_DESC_KEY`로 있음. "
            "2026-09-23: BidRadar 스프레드시트에 이 소스의 endpoint 칸이 비어 있었음(다른 두 소스는 "
            "채워져 있었는데 이것만 없음) — BidRadar도 이 API는 실제로 호출까지 성공한 적이 없을 "
            "가능성. KOSIS는 data.go.kr 키 체계와 kosis.kr 자체 openAPI 키 체계가 별개라(문서 조사로 "
            "확인) 정확한 apis.data.go.kr 엔드포인트를 특정 못 함. 아래 constitutional_court/"
            "law_terms_kb에서 발견한 IP 화이트리스트 문제까지 겹칠 가능성이 높아 커넥터 구현을 "
            "보류함 — 엔드포인트 확정 + IP 등록 여부 확인이 먼저 필요."
        ),
        implemented=False,
    ),
    SourceConfig(
        source_id="constitutional_court",
        name="헌법재판소 판례정보 조회 서비스",
        school_levels=["중학교", "고등학교"],
        subjects=["사회"],
        sub_domain="법",
        requires_auth=True,
        license=LicenseInfo(
            license_type="공공누리 제1유형(출처표시)",
            allows_modification=True,
            allows_commercial=True,
            source_policy_note="data.go.kr 15141085. 기본권·인권·헌법 성취기준 직결, 판례요지가 해설문 형태.",
        ),
        notes=(
            "data.go.kr 활용신청, 개발단계 자동승인 확인(운영단계는 심의). 2026-09-21: BidRadar 계정의 "
            "인증키를 재사용하기로 사용자 결정(kosis_desc와 동일 배경). "
            "`~/secrets/mlops-edge-lab/education_api_keys.env`에 `MLOPS_EDU_CONSTITUTIONAL_COURT_KEY`로 "
            "있음. 엔드포인트 베이스는 BidRadar 스프레드시트로 확인: "
            "`https://apis.data.go.kr/9750000/PrecedentInfomationService`. "
            "2026-09-23: grib-ai-server에서 실제 키로 후보 오퍼레이션 7개를 라이브 테스트함 — "
            "`getRealmMainPrcdntList`만 다른 오퍼레이션(reasonCode 12 'NO_OPENAPI_SERVICE_ERROR', "
            "즉 경로 자체가 없음)과 달리 reasonCode 30 'SERVICE_KEY_IS_NOT_REGISTERED_ERROR'를 "
            "반환함 — 이 오퍼레이션 경로 자체는 실재하지만, 이 서비스키가 등록 안 된 것으로 판정됨. "
            "law_terms_kb에서 확인한 것과 같은 IP 화이트리스트 문제로 강하게 추정(아래 참고). "
            "**막힌 지점**: BidRadar가 활용신청 시 등록한 서버 IP가 grib-ai-server(공인 IP "
            "1.220.120.74, 2026-09-23 확인)와 다를 가능성이 큼 — data.go.kr 활용신청 상세보기에서 "
            "등록된 IP를 확인하거나, grib-ai-server IP를 추가 등록해야 진행 가능."
        ),
        implemented=False,
    ),
    SourceConfig(
        source_id="law_terms_kb",
        name="법제처 법령정보지식베이스(법령용어-일상용어 연계 조회)",
        school_levels=["중학교", "고등학교"],
        subjects=["사회"],
        sub_domain="법",
        requires_auth=True,
        license=LicenseInfo(
            license_type="제한 없음",
            allows_modification=True,
            allows_commercial=True,
            source_policy_note="어려운 법률용어를 일상어로 매핑한 데이터를 기관이 직접 제공 — 중학생 눈높이 변환에 직접 쓸모 있음.",
        ),
        notes=(
            "API 유형이 LINK로 표시됨 — 2026-09-21 BidRadar 계정 스프레드시트에서 실제 확인해보니 "
            "호출 주소가 open.law.go.kr(OC 인증키 방식)이었음, 예상대로 law_go_kr과 같은 인증 체계로 "
            "보임. BidRadar 계정의 키를 재사용하기로 사용자 결정(kosis_desc와 동일 배경) — "
            "`~/secrets/mlops-edge-lab/education_api_keys.env`에 `MLOPS_EDU_LAW_TERMS_KB_KEY`로 있음. "
            "엔드포인트는 BidRadar 스프레드시트로 확인: `http://www.law.go.kr/DRF/lawSearch.do`. "
            "2026-09-23: grib-ai-server에서 실제 키로 라이브 호출해봄 — target 파라미터(lstrmAI 등 "
            "4종 후보) 무엇을 넣어도 전부 동일한 응답: '사용자 정보 검증에 실패하였습니다 — OPEN API "
            "호출 시 사용자 검증을 위하여 정확한 서버장비의 IP주소 및 도메인주소를 등록해 주세요.' "
            "law.go.kr의 OC 키는 호출 서버의 IP/도메인을 미리 등록해야 동작하는 화이트리스트 방식임을 "
            "실측으로 확인함 — target 코드가 맞는지 여부와 무관하게 이 에러가 먼저 막는다. "
            "**막힌 지점**: BidRadar가 등록한 IP가 grib-ai-server(공인 IP 1.220.120.74, 2026-09-23 "
            "확인)와 다름. law.go.kr 계정(open.law.go.kr 마이페이지)에서 이 IP를 추가 등록해야 진행 "
            "가능 — 또는 그립 회사 네트워크의 고정 아웃바운드 IP를 대신 등록해뒀다면 grib-ai-server가 "
            "그 IP로 나가는지 확인 필요."
        ),
        implemented=False,
    ),
]


def sources_for(school_level: str, subject: str) -> list[SourceConfig]:
    return [
        s for s in SOURCES
        if school_level in s.school_levels and subject in s.subjects
    ]
