# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, r"C:\Users\jyahn\.claude\skills\synced\e1336d5b-00ae-4578-bcb3-20eed59b9c78_3719b39e-e72c-4ba0-bdde-9b8c5c919051\grib-ppt\scripts")
from grib_ppt_base import GribPpt, SLIDE_BODY_TABLE
from pptx.util import Inches, Pt
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

deck = GribPpt()

# ── 표지 ──────────────────────────────────────
deck.set_cover(main_title="MLOps-Edge-Lab", date="2026년 09월")
GribPpt._replace_text_in_slide(deck.prs.slides[0], {"[MLOps-Edge-Lab]": "MLOps-Edge-Lab"})
# 제목이 기본 박스 폭(3.77in)보다 길어서 2줄로 줄바꿈됐다(사용자 지적, 2026-09-21) —
# 박스를 넓혀서 한 줄에 들어가게 한다. 로고/원형 이미지는 6.5in 부근부터 시작해서
# 6in까지는 안전하다.
for shape in deck.prs.slides[0].shapes:
    if shape.has_text_frame and "MLOps-Edge-Lab" in shape.text_frame.text:
        shape.width = Inches(6.0)

deck.set_toc(["시스템 개요", "기술 구성과 선택 근거", "서비스 적용 사례"])


def add_flow_diagram_slide(headerSub, headline, stages, note):
    """마스터에 흐름도 레이아웃이 없어서, '본문—표' 슬라이드를 복제한 뒤 표를 지우고
    도형(사각형+화살표)으로 직접 그린다 — 챕터 배지·헤드라인 배너 같은 공통 헤더는
    그대로 재사용해서 다른 본문 슬라이드와 톤이 어긋나지 않게 한다."""
    slide = deck._copy_slide(deck.prs, SLIDE_BODY_TABLE)
    deck._apply_chapter_context(slide, headerSub)
    deck._apply_headline(slide, headline)
    for shape in list(slide.shapes):
        if shape.has_table:
            shape._element.getparent().remove(shape._element)

    n = len(stages)
    box_w, box_h, gap = Inches(1.38), Inches(1.05), Inches(0.12)
    total_w = box_w * n + gap * (n - 1)
    start_x = int((Inches(13.333) - total_w) / 2)
    y = Inches(3.0)
    prev = None
    for i, label in enumerate(stages):
        x = start_x + i * (box_w + gap)
        box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, box_w, box_h)
        box.fill.solid()
        box.fill.fore_color.rgb = RGBColor(0x1F, 0x4E, 0x79)
        box.line.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        box.shadow.inherit = False
        tf = box.text_frame
        tf.word_wrap = True
        tf.text = label
        for para in tf.paragraphs:
            para.alignment = PP_ALIGN.CENTER
            for run in para.runs:
                run.font.size = Pt(11)
                run.font.bold = True
                run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        if prev is not None:
            conn = slide.shapes.add_connector(
                MSO_CONNECTOR.STRAIGHT, prev.left + prev.width, y + box_h // 2, x, y + box_h // 2,
            )
            conn.line.color.rgb = RGBColor(0x1F, 0x4E, 0x79)
            conn.line.width = Pt(2.25)
        prev = box

    note_box = slide.shapes.add_textbox(start_x, y + box_h + Inches(0.5), total_w, Inches(0.6))
    ntf = note_box.text_frame
    ntf.text = note
    ntf.paragraphs[0].alignment = PP_ALIGN.CENTER
    for run in ntf.paragraphs[0].runs:
        run.font.size = Pt(13)
        run.font.italic = True
        run.font.color.rgb = RGBColor(0x59, 0x59, 0x59)
    return slide


# ── 챕터 Ⅰ. 시스템 개요 ─────────────────────────
deck.add_chapter("Ⅰ", "시스템 개요")

deck.add_body_stats(
    headerSub="1. 핵심 지표",
    headline="온프레미스·엣지 환경에서 규칙 기반 판정과 LLM을 결합한 sLLM 플랫폼",
    stats=[
        {"num": "738건", "label": "안전문서 추출 (99.9%)"},
        {"num": "12,964개", "label": "RAG 검색 청크"},
        {"num": "3.2배", "label": "양자화 압축 (7.5GB→2.3GB)"},
        {"num": "3.7배", "label": "GPU 추론 가속"},
    ],
)

deck.add_body_2col(
    headerSub="2. 사업 배경",
    headline="정부과제·잠재고객 수요와 온프레미스 제약이 만나는 지점",
    left_title="왜 시작했나",
    left_items=[
        "정부과제·잠재고객 sLLM 수요 증가",
        "AI 성능·데이터 보안 관점 온프레미스 요구",
        "sLLM이 부가 기능으로 요청되는 경우 증가",
        "1인 개발, 온프레미스 GPU 서버 1대",
    ],
    right_title="왜 온프레미스·오프라인 우선인가",
    right_items=[
        "산업 현장은 네트워크 단절·통신차단 흔함",
        "클라우드 LLM을 아예 못 쓰는 조건",
        "안전 가이드라인 문서를 다수 확보 가능",
        "완전 폐쇄망에서도 동작하는 게 목표",
    ],
)

add_flow_diagram_slide(
    headerSub="3. 전체 흐름",
    headline="데이터 추출부터 운영까지 — 8단계가 순서대로 이어진다",
    stages=["1.개발환경", "2.데이터추출", "3.모델+파인튜닝", "4.양자화", "5.RAG", "6.Agent", "7.엣지에뮬", "8.운영인프라"],
    note="↺ 6→2/3단계: 운영 중 쌓인 피드백을 다시 학습 데이터로 반영(자동화 완료)",
)

deck.add_body_3level(
    headerSub="4. 목표와 원칙",
    headline="두 가지 목표를 네 가지 설계 원칙이 관통한다",
    items=[
        {"text": "목표 ① MLOps 파이프라인 구축"},
        {"text": "실험관리(MLflow) → 학습 → 평가"},
        {"text": "배포(모델 레지스트리) → 모니터링"},
        {"text": "Prometheus+Grafana로 운영 지표 실시간 확인"},
        {"text": "평가 기준 미달 시 자동 재학습"},
        {"text": "목표 ② 엣지 sLLM 구축"},
        {"text": "양자화(Q4_K_M)로 경량화, GPU 없이도 동작"},
        {"text": "cgroup 에뮬레이션으로 저사양 환경 하한선 추정"},
        {"text": "규칙+LLM 역할 분리로 안전한 자동 판단"},
        {"text": "설계 철학 — 온프레미스·오프라인 우선"},
    ],
)

# ── 챕터 Ⅱ. 기술 구성과 단계별 선택 근거 ─────────
deck.add_chapter("Ⅱ", "기술 구성과 선택 근거")

deck.add_body_table(
    headerSub="1. 개발 환경",
    headline="실험 재현성과 배포 안정성을 우선한 개발 환경",
    rows=[
        ["결정", "선택", "이유", "비고", "상태"],
        ["실행 위치", "AI 서버 하나로 통일", "로컬에서 학습·추론·서빙 안 함", "grib-ai-server", "완료"],
        ["환경관리자", "uv (venv/conda 대신)", "lock 파일로 실험 재현성 확보", "-", "완료"],
        ["MLflow backend", "SQLite", "조건 검색에 유리, 백업 쉬움", "mlflow.db", "완료"],
        ["상시 실행", "tmux → systemd --user", "재부팅에도 살아남게", "Restart=always", "완료"],
    ],
)

deck.add_body_table(
    headerSub="2. 데이터 추출",
    headline="포맷별 폴백 사슬과 조용한 실패 금지 원칙",
    rows=[
        ["결정", "선택", "이유", "비고", "실측"],
        ["PDF 라이브러리", "pymupdf(fitz)", "OCR용 렌더링 겸용, 외부도구 불필요", "-", "-"],
        ["레거시 포맷", "LibreOffice→PDF 변환 후 재사용", "포맷별 전용 파서 불필요", "HWP/DOC/XLS", "-"],
        ["HWP", "pyhwp 1차, LibreOffice 2차", "더 가볍고 변환 품질 안정적", "-", "-"],
        ["확장자 오표기", "매직바이트로 실제 구조 재확인", "확장자만으로는 신뢰 불가", ".hwpx가 실제 OLE인 사례", "738/738건 성공"],
    ],
)

deck.add_body_3level(
    headerSub="3. 모델 선정",
    headline="라이선스·한국어 품질·크기 — 세 기준으로 Qwen3 채택",
    items=[
        {"text": "채택 기준 3가지 — 라이선스·한국어 품질·크기"},
        {"text": "① 상업적 이용 가능한 라이선스"},
        {"text": "② 한국어 품질"},
        {"text": "③ 엣지 배포 가능한 크기(1~3B급)"},
        {"text": "GPU 서버용 대형 모델도 같은 계열이면 유리"},
        {"text": "제외된 후보"},
        {"text": "EXAONE — 한국어 최고, 비상업 전용이라 제외"},
        {"text": "Upstage Solar — 최소 10.7B, 엣지에 부적합"},
        {"text": "DeepSeek V4 — 라이선스는 깨끗하나 한국어 불확실"},
        {"text": "채택 — Qwen/Qwen3-4B-Instruct-2507"},
    ],
)

deck.add_body_table(
    headerSub="3. 모델 비교표",
    headline="4개 후보의 라이선스·한국어 품질 정량 비교",
    rows=[
        ["모델", "기업/국가", "라이선스", "한국어 품질", "결과"],
        ["Qwen3", "Alibaba·중국", "Apache 2.0", "다국어 최상위권", "채택"],
        ["EXAONE 4.0", "LG AI연구원·한국", "비상업 전용", "한국어 최상", "제외(라이선스)"],
        ["Upstage Solar", "Upstage·한국", "조건부 상업", "우수", "제외(최소10.7B)"],
        ["Gemma 3/4", "Google·미국", "조건부 상업", "양호", "차선책"],
    ],
)

deck.add_body_2col(
    headerSub="3-2. 파인튜닝",
    headline="현장 알림체를 소량 예시로 가르치는 LoRA",
    left_title="왜 필요했나",
    left_items=[
        "기본 모델은 일반적 문장만 생성",
        "\"3동 2층 화기작업구역...\" 같은 간결한 알림체 필요",
        "LoRA — 모델 전체가 아닌 보정 레이어만 학습",
        "r=16, alpha=32, bf16(양자화 없음)",
    ],
    right_title="효과 (정직하게 두 가지로)",
    right_items=[
        "배관검증용 — 합성 32건 5에포크, 품질지표 아님",
        "실제 개선폭 — 안전 도메인 ROUGE-L 0.525→0.719",
        "한계 — 합성 데이터라 실사용 품질보증 아님",
        "에듀 도메인은 별도 실측 0.239→0.373",
    ],
)

deck.add_body_table(
    headerSub="4. 양자화 방식",
    headline="K-quant 계열에서 크기·품질 균형점을 찾는다",
    rows=[
        ["방식", "계열", "크기(4B 기준)", "품질 손실", "비고"],
        ["F16(무양자화)", "-", "~8.0GB", "없음(기준)", "GPU 서버 학습용"],
        ["Q5_K_M", "K-quant", "~2.7GB", "적음", "품질 우선 대안"],
        ["Q4_K_M", "K-quant", "~2.4GB(실측)", "적당", "채택 · 범용 기본값"],
        ["Q3_K_M", "K-quant", "~1.9GB", "눈에 띔", "RAM 빠듯한 엣지용"],
    ],
)

deck.add_body_2col(
    headerSub="4. 실측 결과",
    headline="같은 GGUF 파일 하나로 CPU·GPU 둘 다 동작",
    left_title="압축 전후",
    left_items=[
        "f16 7.5GB → Q4_K_M 2.3GB",
        "약 3.2배 압축",
        "CPU에서 정확한 문장 생성 확인",
        "범용 기본값으로 유지",
    ],
    right_title="CPU vs GPU 벤치마크",
    right_items=[
        "CPU(Xeon Silver, 8스레드): 23.6 tok/s",
        "GPU(RTX 4000 Ada): 87.0 tok/s",
        "배율 약 3.7배",
        "같은 파일을 두 백엔드가 그대로 사용",
    ],
)

deck.add_body_table(
    headerSub="5. RAG 파이프라인",
    headline="근거 없이는 답하지 않는 검색 결합 생성",
    rows=[
        ["구성요소", "기술", "역할", "실측", "비고"],
        ["청킹", "문단 슬라이딩윈도우", "문서를 검색 단위로 분할", "12,964개 청크", "738건 문서 기준"],
        ["임베딩", "multilingual-e5-small", "의미 기반 벡터화", "-", "다국어 지원"],
        ["검색", "FAISS", "최근접 벡터 검색", "hit_rate@4 = 0.857", "골든셋 평가"],
        ["생성", "sLLM(Qwen3)", "검색결과 근거로 답변 생성", "근거 없으면 모른다고 답함", "할루시네이션 방지"],
    ],
)

deck.add_body_3col(
    headerSub="6. Agent 원칙",
    headline="판정은 규칙, 서술은 LLM, 고위험은 사람이 승인한다",
    cards=[
        {"title": "규칙 우선", "items": ["초과·위험도 판정은 코드", "장비 제어도 규칙이 즉시 실행", "LLM은 판정을 재수행하지 않음"]},
        {"title": "근거 기반", "items": ["RAG 검색 결과 없이 답하지 않음", "도구 호출 여부는 LLM 자율판단", "할루시네이션 방지 핵심 장치"]},
        {"title": "승인 체계", "items": ["저위험 장비는 자동 실행", "고위험은 사람 승인 대기", "Human-in-the-loop 전 도메인 공통"]},
    ],
)

deck.add_body_table(
    headerSub="6. Agent 역할 분리",
    headline="같은 상황도 담당 주체에 따라 처리 방식이 다르다",
    rows=[
        ["역할", "담당", "설명", "비고", "권한"],
        ["판정", "규칙(rules.judge)", "초과 여부·위험도 계산", "LLM이 재판정 안 함", "코드"],
        ["도구호출", "LLM", "search_guidelines 호출여부·검색어", "자율판단", "LLM"],
        ["저위험 장비", "규칙", "즉시 자동 실행", "환풍기 등", "코드"],
        ["고위험 장비", "사람 승인", "자동 실행 안 함", "가스차단기 등", "사람"],
    ],
)

# 6-1. 센서 데이터 — Agent의 입력
# 원래 2단계(문서 추출) 아래 "2-2. 센서 데이터"로 있었으나, 사용자 지적(2026-09-21:
# "2-2 센서데이터가 데이터 수집에 있는게 맞나? Agent의 입력이 아닌가")으로 6단계 아래로
# 이동했다 — 실제로 이 값을 소비하는 건 문서 기반 RAG가 아니라 Agent의 규칙 판정
# (rules.judge)이기 때문. src/sensors.py에 정의를 모아두고 학습데이터·웹시뮬레이터·
# Agent가 전부 공유한다.
deck.add_body_table(
    headerSub="6-1. 센서 데이터",
    headline="Agent 규칙판정(rules.judge)이 소비하는 3개 카테고리 정의",
    rows=[
        ["카테고리", "측정 물질", "단위 예", "임계값 근거", "비고"],
        ["공기질", "온도·습도·PM10·PM2.5·CO2·TVOC", "℃/%/㎍/m³/ppm", "KOSHA 사무실 공기관리지침·실내공기질관리법 시행규칙", "6개 물질"],
        ["가스", "산소(O2)·수소", "%/%LEL", "산업안전보건기준에 관한 규칙 제618조", "산소는 낮을수록 위험(판정 로직 방향 반대)"],
        ["조리흄", "포름알데히드 등 9종", "ppm/ng·mg/m³", "고용노동부 노출기준 고시", "4종(아세트알데히드·벤조에이피렌 등)은 근사값"],
        ["구조 변경", "헬륨·질소·아르곤 → 산소(O2) 통합", "%", "산업안전보건기준 제618조(산소 18%미만=산소결핍)", "단순 질식제라 개별기준 없음, 3물질→1센서 통합"],
    ],
)

deck.add_body_2col(
    headerSub="6-1. 센서 데이터",
    headline="2단계가 아니라 6단계 — 실제 소비자를 반영해 재배치했다",
    left_title="왜 6단계로 옮겼나",
    left_items=[
        "소비 주체는 문서추출·RAG가 아니라 Agent 규칙판정(rules.judge)",
        "src/sensors.py에 정의를 두고 학습데이터·웹시뮬레이터·Agent가 공유",
        "사용자 지적(2026-09-21): \"Agent의 입력 아닌가\"",
        "원래 2단계 아래 있던 정의를 6단계로 재배치",
    ],
    right_title="정직한 현황",
    right_items=[
        "임계값 — 처음엔 placeholder, 법령·고시·ACGIH/NIOSH 조사로 대부분 교체",
        "남은 한계 — 4개 물질은 국내외 공식기준 자체가 없어 근사값",
        "아직 실제 센서 장비 없음 — 값은 무작위(웹데모)·합성(학습데이터)",
        "3곳에서 공유 — Agent 판정, 파인튜닝 데이터, /simulate·/control-room UI",
    ],
)

deck.add_body_3level(
    headerSub="7. 엣지 에뮬레이션",
    headline="실제 하드웨어 없이 성능 하한선을 추정한다",
    items=[
        {"text": "cgroup(systemd-run)으로 CPU·메모리 제한"},
        {"text": "재현 가능 — CPU 코어 수, 메모리 상한"},
        {"text": "재현 불가 — 명령어셋(AVX-512), 디스크 I/O, 발열"},
        {"text": "그래서 정확한 예측이 아니라 하한선"},
        {"text": "실측치"},
        {"text": "실측 결과"},
        {"text": "2코어/32GB → tg 7.32 tok/s"},
        {"text": "48코어 서버 CPU 벤치(19.1 tok/s)보다 낙관적일 수 있음"},
        {"text": "진짜 엣지 하드웨어 확보 전까지는 참고치"},
        {"text": "다음 단계 — 실제 하드웨어 확보 후 재검증"},
    ],
)

deck.add_body_table(
    headerSub="8. 운영 인프라",
    headline="모델 레지스트리부터 CI/CD까지 표준 MLOps 갭을 채운다",
    rows=[
        ["갭", "도구", "역할", "비고", "상태"],
        ["모델 레지스트리", "MLflow Model Registry", "성능 기준 통과 모델만 Production 승격", "train/auto_retrain.py", "완료"],
        ["모니터링", "Prometheus + Grafana", "서비스 상태·요청량·지연시간 실시간 확인", "/metrics 엔드포인트", "완료"],
        ["데이터 버저닝", "DVC", "대용량 학습 데이터 버전 관리", "로컬 원격", "완료"],
        ["CI/CD", "GitHub Actions", "push마다 테스트→자동 배포", "테스트~배포 27초", "완료"],
    ],
)

# ── 챕터 Ⅲ. 서비스 적용 사례 ─────────────────────
deck.add_chapter("Ⅲ", "서비스 적용 사례")

deck.add_body_3col(
    headerSub="1. 사업 기회",
    headline="산업안전·제조AX·에듀테크 세 영역에서 실제 수요 확인",
    cards=[
        {"title": "한국수자원공사", "items": ["온프레미스 실험실 안전 대응", "자연어 기반 안내·질의응답", "산업안전 Agent와 동일 구조"]},
        {"title": "강원 AX 화장품", "items": ["AX 팩토리 Layer-4 AI Agent", "운영자용 자연어 대시보드", "개념검증(PoC) 완료"]},
        {"title": "클랙스", "items": ["단순 에듀테크가 아닌 AI융합교육", "AI튜터·AI토론이 실제 사례", "완전 오프라인 동작"]},
    ],
)

deck.add_body_3level(
    headerSub="2. 스마트팩토리",
    headline="설비·SOP만 교체하면 같은 Agent 구조를 재사용한다",
    items=[
        {"text": "목적 — 화장품 AX 패턴의 일반화"},
        {"text": "설비군·센서 종류·SOP만 교체"},
        {"text": "EQUIPMENT_MAP·SOP_RANGES가 이미 설정 기반 구조"},
        {"text": "제안서에도 Config 기반 아키텍처로 확장 반영 명시"},
        {"text": "Human-in-the-loop 원칙 동일 적용"},
        {"text": "단계별 변경"},
        {"text": "데이터 = PLC/MES 시계열, RAG = 설비 SOP"},
        {"text": "Agent = 예지보전 알람 + 승인 워크플로"},
        {"text": "나머지(개발환경·양자화·엣지에뮬·운영인프라)는 동일"},
        {"text": "구현 상태 — 개념검증(화장품 AX PoC)"},
    ],
)

deck.add_body_3level(
    headerSub="3. AX 에듀테크",
    headline="AI튜터·AI토론이 이미 실제 사례다",
    items=[
        {"text": "목적 — 단순 에듀테크가 아닌 AI융합교육"},
        {"text": "AI튜터(RAG+LoRA)·AI토론이 실제 사례"},
        {"text": "완전 오프라인 동작(학교 네트워크 제약 대응)"},
        {"text": "학생 개인정보 서버 미전송(localStorage)"},
        {"text": "실측 ROUGE-L 0.239→0.373"},
        {"text": "단계별 변경"},
        {"text": "데이터 = 교과서·교육과정, RAG = 경제·지리 지식"},
        {"text": "Agent = 오답노트+학습진도 추적(장비제어 없음)"},
        {"text": "나머지는 동일 플랫폼 재사용"},
        {"text": "구현 상태 — 완전 구축"},
    ],
)

# 3(추가). AI 토론 — 팀 배정 메커니즘
# 학생 의견을 임베딩으로 벡터화 → 균형 클러스터링으로 팀을 나누고(판정=코드),
# LLM은 그 결과에 라벨만 붙인다. 2026-09-21 실측 예시(8명→3팀)를 그대로 반영.
deck.add_body_2col(
    headerSub="3. AX 에듀테크 — AI 토론",
    headline="판정은 코드(균형 클러스터링), 라벨링만 LLM이 한다",
    left_title="배정 메커니즘",
    left_items=[
        "학생 의견을 임베딩으로 벡터화(RAG와 같은 다국어 모델)",
        "균형 잡힌 클러스터링으로 팀 수만큼 분할(인원을 고르게)",
        "LLM은 클러스터 결과에 주제 관련 이름만 붙임",
        "판정=코드, 라벨링만=LLM — 프로젝트 전체 원칙과 동일",
    ],
    right_title="실제 예시 개요(2026-09-21 실측)",
    right_items=[
        "주제 — 급식실 식재료: 현지 농산물 vs 도시 외식업체 조달",
        "학생 8명 → 3개 팀으로 자동 분리",
        "찬성 5명이 \"지역경제\" vs \"환경·품질\" 두 갈래로 세분화",
        "같은 입장이어도 강조점이 다르면 다른 팀으로 갈릴 수 있음",
    ],
)

deck.add_body_table(
    headerSub="3. AX 에듀테크 — AI 토론",
    headline="찬성 5명도 \"지역경제 vs 환경·품질\"로 자동 분리됐다",
    rows=[
        ["팀(라벨)", "배정 학생", "인원", "핵심 근거", "구분"],
        ["현지 농산물 보호와 지역경제 연결", "서민서·김지영·정하늘", "3명", "\"농장·지역경제에 도움\"이 핵심 근거", "찬성"],
        ["환경 보호와 식재료 질 강조", "최지민·박지훈·윤지현", "3명", "같은 찬성이나 환경·품질 관점 우세", "찬성(강조점 다름)"],
        ["간편성과 맛을 우선하는 조달 시스템", "강민호·이승재", "2명", "유일하게 도시 외식업체 지지, 편의성·전문성 강조", "반대"],
        ["참고", "찬성 측 5명은 하나로 뭉치지 않고 의미적 유사도만으로 두 갈래 자동 분리됐다", "", "", ""],
    ],
)

# 4. BidRadar — AI 공고 매칭 (구 "9-3. AI 추천"을 완전히 대체)
# 이전(v4)까지는 미구현 사고실험이었으나, 현재 doc에서 실제 운영 중인 사례로
# 전면 재작성됐다 — 사내 별도 프로젝트 BidRadar가 이 프로젝트의 sLLM 서빙 API를
# 실제로 호출한다(인프라는 완전히 별개).
deck.add_body_2col(
    headerSub="4. BidRadar — 개요",
    headline="사고실험이 아니라 실제 운영 중인 사례다",
    left_title="무엇을 하나",
    left_items=[
        "BidRadar(사내 별도 프로젝트, 인프라 완전 분리)가 호출",
        "이 프로젝트의 sLLM 서빙 API(8단계, 8081/28081 포트)를 그대로 사용",
        "공고 제목 ↔ 고객 관심주제를 의미적으로 매칭(/v1/classify-topic)",
        "첨부문서가 공통서식인지도 판단(/v1/classify-doc)",
    ],
    right_title="원칙과 한계",
    right_items=[
        "confidence는 절대 임계값이 아니라 \"사람이 검토할 후보\"로만 사용",
        "기존 키워드 매칭을 대체하지 않음 — \"판정은 규칙\" 원칙과 동일",
        "정직한 한계 — \"추천\"보다 \"매칭 후보 제시\"에 가까움, 최종 결정은 사람",
        "classify-doc 실패율(185건 중 88건)이 아직 높음",
    ],
)

deck.add_body_table(
    headerSub="4. BidRadar — 실측",
    headline="공고 매칭 5,299건 호출·성공률 100%",
    rows=[
        ["항목", "내용", "실측", "API", "비고"],
        ["공고 매칭", "공고 제목 ↔ 고객 관심주제 의미적 매칭", "5,299건 호출, 성공률 100%", "/v1/classify-topic", "확정 매칭 아님, 후보 제시"],
        ["보조 신호", "키워드 매칭을 대체하지 않음", "평균 지연 1.6초", "/v1/classify-topic", "confidence는 사람 검토용"],
        ["병렬 처리", "전용 GPU 풀(GPU 2·3)로 동시 처리", "4워커", "-", "실시간 서비스 GPU와 분리"],
        ["문서 분류", "첨부문서가 공통서식인지 판단", "185건 호출(성공97·실패88)", "/v1/classify-doc", "실패율 47.6% — 아직 높음"],
    ],
)

# 5. 세 사례 종합 비교 — 파이프라인 단계별 동일/변경 (신규, doc 9-4)
# 스마트팩토리·AX에듀테크는 "같은 플랫폼에 다른 도메인 데이터를 넣은" 확장인 반면,
# BidRadar는 이미 만들어진 5·6단계 결과물(서빙 API)을 완전히 별개 시스템이 외부에서
# 호출하는 구조라, 1·2·3·4·7·8단계는 "변경"이 아니라 "해당없음"이 정확하다 — 표가
# 8행×4열이라 5열 고정 표 마스터에 맞추기 위해 2장으로 나누고 "패턴" 요약열을 더했다.
deck.add_body_table(
    headerSub="5. 세 사례 종합 비교 (1/2)",
    headline="스마트팩토리·AX에듀테크는 확장, BidRadar는 대부분 해당없음",
    rows=[
        ["단계", "스마트팩토리", "AX 에듀테크", "BidRadar", "패턴"],
        ["1. 개발환경", "동일", "동일", "해당없음(별도 인프라)", "SF·AX 동일, BR만 별도"],
        ["2. 데이터추출", "변경(설비 매뉴얼·SOP)", "변경(교과서·공공데이터)", "해당없음", "SF·AX 변경, BR 해당없음"],
        ["3. 파인튜닝", "변경(설비 이상 서술체)", "변경(학생 눈높이 설명체)", "해당없음(재학습 없이 API만 호출)", "SF·AX 변경, BR 해당없음"],
        ["4. 양자화", "동일", "동일", "해당없음(이미 양자화된 모델 호출)", "SF·AX 동일, BR 해당없음"],
    ],
)

deck.add_body_table(
    headerSub="5. 세 사례 종합 비교 (2/2)",
    headline="BidRadar는 확장이 아니라 이미 만든 API를 외부에서 호출하는 구조다",
    rows=[
        ["단계", "스마트팩토리", "AX 에듀테크", "BidRadar", "패턴"],
        ["5. RAG", "변경(SOP 검색)", "변경(교과서 하이브리드 검색)", "변형(RAG 대신 classify-topic 임베딩 매칭)", "SF·AX 변경, BR 변형"],
        ["6. Agent", "동일 구조(임계값 판정+승인)", "변경(장비제어→오답노트/토론팀 배정)", "변형(장비제어→키워드매칭 보조신호, 최종판정은 사람)", "셋 다 다른 양상"],
        ["7. 엣지에뮬", "동일(현장 PC 흉내)", "동일(학교 PC/태블릿)", "해당없음(엣지 배포 없음, 서버 GPU 풀만 사용)", "SF·AX 동일, BR 해당없음"],
        ["8. 운영인프라", "동일", "동일", "별도(분리된 인프라, 네트워크로 API만 호출)", "SF·AX 동일, BR 별도"],
    ],
)

deck.add_body_2col(
    headerSub="6. 관통 원칙",
    headline="규칙 우선 · 근거 기반 · 실측 우선 · 정직한 기록",
    left_title="관통한 설계 원칙",
    left_items=[
        "규칙 우선, LLM은 보조",
        "근거 없으면 모른다고 답함",
        "주장보다 실측 수치",
        "실패도 숨기지 않고 기록",
    ],
    right_title="앞으로의 확장 방향",
    right_items=[
        "실제 엣지 하드웨어 확보 후 재검증",
        "센서 임계값을 정식 안전보건기준으로 교체",
        "운영 피드백 루프의 자동 재학습 확장",
        "스마트팩토리 등 추가 실증 검토(BidRadar는 이미 실측 확보)",
    ],
)

deck.set_closing(
    message="MLOps-Edge-Lab — 규칙 기반 판정과 LLM을 결합한 온프레미스 AI 플랫폼",
)

out_path = r"D:\Code-CLI\MLOps-Edge-Lab\.claude\worktrees\agent-a39a852996d55e779\docs\교육자료\MLOps-Edge-Lab_교육자료_v5.pptx"
deck.save(out_path)
print("saved:", out_path)
