import sys
from pathlib import Path

# grib-ppt는 사내 공용 Claude 스킬이라 동기화 경로가 사용자마다 다르다.
# 로컬에서 `.claude/skills/synced/*/grib-ppt` 를 찾아 재사용하면 됨.
SKILL_DIR = Path(r"C:\Users\jyahn\.claude\skills\synced\e1336d5b-00ae-4578-bcb3-20eed59b9c78_3719b39e-e72c-4ba0-bdde-9b8c5c919051\grib-ppt")
sys.path.insert(0, str(SKILL_DIR / "scripts"))

from grib_ppt_base import GribPpt

# ── 스킬 버그 우회 헬퍼 3종 (마스터 스킬 파일 자체는 건드리지 않고 여기서만 보정) ──

def fix_title_brackets(slide, title):
    """set_cover의 매핑 키가 "[ 문서명 ]"(공백 포함)인데 실제 마스터 텍스트는
    "[문서명]"(공백 없음)이라 안쪽 단어만 치환되고 대괄호가 그대로 남는다."""
    GribPpt._replace_text_in_slide(slide, {f"[{title}]": title})


def fix_chapter_label(slide, correct_rn):
    """_apply_chapter_context의 로마숫자 매핑 루프가 첫 번째로 안 맞는 default를 찾으면
    바로 break해버려서, 템플릿에 실제로 남아있는 다른 로마숫자는 못 고치는 경우가 있다.
    안 맞는 숫자를 전부 correct_rn으로 매핑해서 확실하게 고친다."""
    mapping = {rn: correct_rn for rn in ["Ⅰ", "Ⅱ", "Ⅲ"] if rn != correct_rn}
    GribPpt._replace_text_in_slide(slide, mapping)


def fill_table(slide, rows):
    """add_body_table의 rows 인자가 _replace_text_in_slide로 넘어가는데, 그 함수는
    shape.has_text_frame인 도형만 훑어서 표(GraphicFrame)의 셀 텍스트는 절대 못 건드린다
    (표는 has_text_frame이 아니라 has_table). 셀을 직접 채우되, 기존 런의 서식(헤더 행
    남색 배경·흰 글씨 등)을 유지하려고 run[0]의 텍스트만 바꾸는 방식을 그대로 따른다."""
    for shape in slide.shapes:
        if not shape.has_table:
            continue
        table = shape.table
        for r, row_vals in enumerate(rows):
            for c, val in enumerate(row_vals):
                if r >= len(table.rows) or c >= len(table.columns):
                    continue
                cell = table.cell(r, c)
                for para in cell.text_frame.paragraphs:
                    if para.runs:
                        para.runs[0].text = val
                        for run in para.runs[1:]:
                            run.text = ""
                    else:
                        para.text = val


deck = GribPpt()

COVER_TITLE = "MLOps-Edge-Lab"
deck.set_cover(
    main_title=COVER_TITLE,
    slogan="산업안전 sLLM 파이프라인 — 개발자 아키텍처 & 사업 활용 제안",
    date="2026년 09월",
)
fix_title_brackets(deck.prs.slides[0], COVER_TITLE)

deck.set_toc(["프로젝트 개요", "개발자 레퍼런스 아키텍처", "사업 활용 제안"])

# ── Ⅰ. 프로젝트 개요 ──────────────────────────────
deck.add_chapter("Ⅰ", "프로젝트 개요")

s = deck.add_body_stats(
    headerSub="1. 프로젝트 한눈에",
    headline="사내 GPU 서버 하나로 완결하는 온프레미스 sLLM 파이프라인",
    stats=[
        {"num": "738건", "label": "산업안전 문서 처리 (100%)"},
        {"num": "2.3GB", "label": "양자화 모델 크기 (원본 7.5GB)"},
        {"num": "3.7배", "label": "GPU 가속 (CPU 대비)"},
        {"num": "4종", "label": "표준 MLOps 요소 구현"},
    ],
)
fix_chapter_label(s, "Ⅰ")

s = deck.add_body_3level(
    headerSub="2. 전체 파이프라인",
    headline="8단계로 구성된 엔드투엔드 MLOps 파이프라인",
    items=[
        {"level": 0, "text": "1~3단계 — 데이터에서 모델까지"},
        {"level": 1, "text": "① 인프라: uv · MLflow · tmux"},
        {"level": 1, "text": "② 데이터 추출: 문서 738건 → 텍스트"},
        {"level": 2, "text": "HWP·PDF·이미지·ZIP 폴백 사슬"},
        {"level": 2, "text": "③ 파인튜닝: Qwen3-4B + LoRA"},
        {"level": 0, "text": "4~6단계 — 경량화에서 서빙까지"},
        {"level": 1, "text": "④ 양자화: GGUF Q4_K_M (7.5GB→2.3GB)"},
        {"level": 2, "text": "⑤ RAG: 문서 근거 기반 응답 생성"},
        {"level": 1, "text": "⑥ Agent: 규칙(판정) · LLM(문구) 역할 분리"},
        {"level": 0, "text": "7~8단계 — 엣지 검증 + 표준 MLOps 4요소"},
    ],
)
fix_chapter_label(s, "Ⅰ")

s = deck.add_body_2col(
    headerSub="3. 왜 온프레미스인가",
    headline="산업 현장의 네트워크·프라이버시 제약이 곧 존재 이유",
    left_title="산업안전 현장의 제약",
    left_items=[
        "현장 네트워크 단절·보안망 다수",
        "클라우드 LLM 사용 불가 구간 존재",
        "엣지 디바이스 사양 다양 (GPU 유무)",
        "실시간 장비 제어 필요",
    ],
    right_title="온프레미스 sLLM의 강점",
    right_items=[
        "완전 로컬 추론, 외부 전송 없음",
        "경량화로 저사양 디바이스 대응",
        "GPU 유무 무관, 동일 파일로 서빙",
        "규칙+LLM 분리로 안전성 확보",
    ],
)
fix_chapter_label(s, "Ⅰ")

# ── Ⅱ. 개발자 레퍼런스 아키텍처 ──────────────────
deck.add_chapter("Ⅱ", "개발자 레퍼런스 아키텍처")

table1_rows = [
    ["모델", "라이선스", "한국어", "크기", "비고"],
    ["Qwen3", "Apache 2.0", "최상위권", "0.6B~32B", "채택 — 엣지~서버 한 계열"],
    ["Gemma 3", "Gemma 라이선스", "양호", "1B~27B", "RAM 효율 최고 (4B=4.2GB)"],
    ["DeepSeek V4", "MIT", "검증 부족", "다양", "한국어 품질 불확실"],
    ["EXAONE 4.0", "NC (비상업)", "최상", "1.2B~32B", "한국어 최상, 상업 불가"],
]
s = deck.add_body_table(
    headerSub="1. 베이스 모델 선택 기준",
    headline="상용 가능 오픈웨이트 모델 비교 — Qwen3 채택",
)
fill_table(s, table1_rows)
fix_chapter_label(s, "Ⅱ")

s = deck.add_body_2col(
    headerSub="2. 양자화 방식 선택 기준",
    headline="Q4_K_M — 엣지 배포용 범용 기본값으로 채택",
    left_title="후보 비교 (4B 기준)",
    left_items=[
        "Q8_0 — 거의 무손실, 4.3GB",
        "Q5_K_M — 품질 우선, 2.7GB",
        "Q4_K_M — 범용 기본값, 2.3GB",
        "Q3_K_M — 초경량, 1.9GB",
    ],
    right_title="채택 근거",
    right_items=[
        "크기 절반 이하, 품질 92~95% 유지",
        "K-quant 계열이 legacy보다 우수",
        "실측상 CPU·GPU 모두 실용 속도",
        "RAM 빠듯하면 Q3_K_M로 대체 가능",
    ],
)
fix_chapter_label(s, "Ⅱ")

s = deck.add_body_stats(
    headerSub="3. 성능 실측치",
    headline="같은 GGUF 파일 하나로 CPU·GPU 모두 대응",
    stats=[
        {"num": "23.6tok/s", "label": "CPU 8스레드 생성 속도"},
        {"num": "87.0tok/s", "label": "GPU RTX4000 Ada 생성 속도"},
        {"num": "3.7배", "label": "GPU 가속 배율"},
        {"num": "7.3tok/s", "label": "엣지 에뮬레이션(2코어) 실측"},
    ],
)
fix_chapter_label(s, "Ⅱ")

s = deck.add_body_3col(
    headerSub="4. 3대 핵심 기술 스택",
    headline="파인튜닝 · RAG · Agent — 역할이 다른 세 기술의 조합",
    cards=[
        {"title": "sLLM 파인튜닝", "items": ["Qwen3-4B + LoRA(r=16)", "도메인 문체 학습", "판정은 LLM에 안 맡김"]},
        {"title": "RAG", "items": ["e5-small 임베딩 + FAISS", "안전문서 12,964개 조각", "근거 없으면 '모른다'"]},
        {"title": "Agent", "items": ["규칙: 위험도·장비 제어", "LLM: 검색·문구 작성", "결정마다 권한 태그 기록"]},
    ],
)
fix_chapter_label(s, "Ⅱ")

s = deck.add_body_3level(
    headerSub="5. 설계 원칙",
    headline="위험 판정은 항상 규칙, LLM은 언어만 담당",
    items=[
        {"level": 0, "text": "원칙 1 — 판정은 규칙, 서술은 LLM"},
        {"level": 1, "text": "위험도(정상·주의·위험)는 임계값 규칙이 판정"},
        {"level": 1, "text": "LLM은 검색 여부·문구 작성만 자율판단"},
        {"level": 2, "text": "파인튜닝 평가서 LLM 오판 실제 발견 → 이 원칙으로 해결"},
        {"level": 2, "text": "장비 제어(가스차단기 등)도 규칙이 결정"},
        {"level": 0, "text": "원칙 2 — 고위험 장비는 자동 실행 금지"},
        {"level": 1, "text": "저위험(환풍기)만 즉시 자동 실행"},
        {"level": 2, "text": "고위험(가스차단·소화방출)은 승인 대기만"},
        {"level": 1, "text": "모든 결정에 규칙/LLM 권한 태그로 추적"},
        {"level": 0, "text": "원칙 3 — 표준 MLOps 4요소(레지스트리·모니터링·버저닝·CI/CD) 구현"},
    ],
)
fix_chapter_label(s, "Ⅱ")

s = deck.add_body_3level(
    headerSub="6. 레퍼런스 아키텍처 재현 가이드",
    headline="이 저장소 구조를 그대로 따라하면 재현 가능",
    items=[
        {"level": 0, "text": "소스 구조 (src/)"},
        {"level": 1, "text": "extract/ — 문서 추출 폴백 사슬"},
        {"level": 1, "text": "train/ — 파인튜닝·재학습·평가"},
        {"level": 2, "text": "rag/ — 청킹·임베딩·FAISS"},
        {"level": 2, "text": "agent/ — 규칙+도구호출 오케스트레이션"},
        {"level": 0, "text": "인프라 (infra/, scripts/, .github/)"},
        {"level": 1, "text": "monitoring/ — Prometheus + Grafana"},
        {"level": 2, "text": "deploy.sh — git pull 기반 배포"},
        {"level": 1, "text": "CI/CD — GitHub Actions 워크플로"},
        {"level": 0, "text": "저장소: github.com/grib-arws-bot/mlops-edge-lab"},
    ],
)
fix_chapter_label(s, "Ⅱ")

# ── Ⅲ. 사업 활용 제안 ────────────────────────────
deck.add_chapter("Ⅲ", "사업 활용 제안")

s = deck.add_body_stats(
    headerSub="1. 시장 기회",
    headline="산업안전관리 시장 — 완전 오프라인이 곧 차별점",
    stats=[
        {"num": "통신차단", "label": "현장 다수가 네트워크 제한 환경"},
        {"num": "완전로컬", "label": "데이터 외부 전송 없는 보안"},
        {"num": "하드웨어", "label": "기존 엣지 장비와 결합 가능"},
        {"num": "신뢰설계", "label": "과대판정 방지로 오작동 최소화"},
    ],
)
fix_chapter_label(s, "Ⅲ")

s = deck.add_body_2col(
    headerSub="2. 경쟁 대비 차별화",
    headline="클라우드 LLM 기반 솔루션 대비 강점",
    left_title="일반 클라우드 LLM 솔루션",
    left_items=[
        "인터넷 연결 필수",
        "데이터 외부 전송 (보안 우려)",
        "사용량 기반 과금 부담",
        "네트워크 단절 시 서비스 중단",
    ],
    right_title="우리 온프레미스 sLLM",
    right_items=[
        "완전 오프라인 동작",
        "데이터는 현장·사내에만 보관",
        "1회 구축, 종량제 부담 없음",
        "기존 엣지 하드웨어 활용 가능",
    ],
)
fix_chapter_label(s, "Ⅲ")

s = deck.add_body_3col(
    headerSub="3. 확장 가능 사업 영역",
    headline="산업안전에서 스마트교육까지, 같은 기술로 확장",
    cards=[
        {"title": "산업안전 (1차)", "items": ["CCTV·센서 이상 알림", "안전수칙 즉시 검색", "장비 자동 제어 연동"]},
        {"title": "스마트교육 (2차 후보)", "items": ["오프라인 학습 튜터", "AI 토론 배틀", "학생 데이터 로컬 보관"]},
        {"title": "기존 사업 연계", "items": ["산업안전 입찰 다수 추적 중", "엣지 하드웨어 제품군과 결합", "실증 데모로 즉시 활용 가능"]},
    ],
)
fix_chapter_label(s, "Ⅲ")

table2_rows = [
    ["단계", "내용", "기간(안)", "산출물", "비고"],
    ["1", "현장 안전 PoC 고도화", "1~2개월", "실증 데모", "이번 프로젝트 결과물 기반"],
    ["2", "실제 센서·임계값 확보", "1개월", "정식 기준 반영", "현장 안전 담당자 협업 필요"],
    ["3", "엣지 하드웨어 탑재 검증", "1~2개월", "엣지 실기 테스트", "현재는 자원 에뮬레이션 단계"],
    ["4", "고객 파일럿", "2~3개월", "레퍼런스 사례", "기존 입찰 추적 대상과 연계"],
]
s = deck.add_body_table(
    headerSub="4. 도입 로드맵 제안",
    headline="단계별 제품화 로드맵",
)
fill_table(s, table2_rows)
fix_chapter_label(s, "Ⅲ")

s = deck.add_body_3level(
    headerSub="5. 논의가 필요한 지점",
    headline="사업화 전 확인이 필요한 사항",
    items=[
        {"level": 0, "text": "기술적 확인사항"},
        {"level": 1, "text": "실제 센서·임계값 확보 시점"},
        {"level": 1, "text": "실제 엣지 하드웨어 사양 확인"},
        {"level": 2, "text": "엣지 실기 성능 — 현재는 자원 에뮬레이션뿐"},
        {"level": 2, "text": "장비 제어 매핑, 현장 검증 필요"},
        {"level": 0, "text": "사업적 확인사항"},
        {"level": 1, "text": "목표 고객군 — 신규 vs 기존 고객"},
        {"level": 2, "text": "가격 정책 — 1회성 vs 유지보수"},
        {"level": 1, "text": "경쟁 제품 벤치마킹 필요"},
        {"level": 0, "text": "다음 논의 — 파일럿 고객 1곳 선정"},
    ],
)
fix_chapter_label(s, "Ⅲ")

deck.set_closing(
    message="MLOps와 엣지 sLLM, 처음부터 끝까지 직접 만들어본 여정입니다.",
    contact="jeffrey@grib.co.kr · github.com/grib-arws-bot/mlops-edge-lab",
)

out = Path(r"C:\Users\jyahn\AppData\Local\Temp\claude\d--Code-CLI-MLOps-Edge-Lab\87e19a71-963e-4bbb-8d62-ff7f5989f822\scratchpad\MLOps-Edge-Lab_교육자료.pptx")
deck.save(out)
print("saved ok")
