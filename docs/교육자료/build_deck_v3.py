# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, r"C:\Users\jyahn\.claude\skills\synced\e1336d5b-00ae-4578-bcb3-20eed59b9c78_3719b39e-e72c-4ba0-bdde-9b8c5c919051\grib-ppt\scripts")
from grib_ppt_base import GribPpt

deck = GribPpt()

deck.set_cover(
    main_title="MLOps-Edge-Lab",
    date="2026년 09월",
)
# set_cover()의 기본 매핑 키("[ 문서명 ]"/"문서명")가 마스터의 실제 런 텍스트
# "[문서명]"(공백 없음)과 정확히 안 맞아서 "문서명"만 치환되고 대괄호가 그대로
# 남았다(1차 렌더 확인, 2026-09-21) — 실제 남은 문자열을 직접 다시 치환해서 고친다.
GribPpt._replace_text_in_slide(deck.prs.slides[0], {"[MLOps-Edge-Lab]": "MLOps-Edge-Lab"})

deck.set_toc(["시스템 개요", "기술 구성과 선택 근거", "서비스 적용 사례"])

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

deck.add_body_3level(
    headerSub="2. 목표와 원칙",
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
    headerSub="1. 1~4단계",
    headline="개발 환경부터 양자화까지 — 채택 근거와 실측치",
    rows=[
        ["단계", "핵심 기술/도구", "채택 근거", "실측 수치", "상태"],
        ["1. 개발 환경", "uv + MLflow + systemd", "실험 재현성(lock 파일)", "-", "완료"],
        ["2. 데이터 추출", "PDF/HWP/OCR 폴백 체인", "포맷별 실패 시 다음 경로로", "738/738건 성공", "완료"],
        ["3. 모델선정+파인튜닝", "Qwen3-4B, LoRA", "Apache 2.0 · 한국어 상위권 · 크기 다양", "ROUGE-L 0.525→0.719", "완료"],
        ["4. 양자화", "GGUF Q4_K_M", "품질 92~95% 유지, 절반 이하 압축", "7.5GB→2.3GB, GPU 3.7배", "완료"],
    ],
)

deck.add_body_table(
    headerSub="2. 5~8단계",
    headline="검색부터 운영까지 — 채택 근거와 실측치",
    rows=[
        ["단계", "핵심 기술/도구", "채택 근거", "실측 수치", "상태"],
        ["5. RAG", "다국어 임베딩 + FAISS", "청킹→임베딩→검색→생성", "hit_rate@4 = 0.857", "완료"],
        ["6. Agent", "규칙 판정 + LLM 도구호출", "판정은 코드, 서술은 LLM", "정상/주의/위험 3단계 검증", "완료"],
        ["7. 엣지 에뮬레이션", "cgroup(systemd-run)", "실 하드웨어 없이 하한선 추정", "2코어/32GB → 7.32 tok/s", "진행중"],
        ["8. 운영 인프라", "MLflow+Prometheus+Grafana+CI/CD", "표준 MLOps 갭 4가지 충족", "배포 자동화 27초", "완료"],
    ],
)

# 마스터의 3단계 슬롯은 위치별로 레벨이 고정돼 있다(1=L0,2=L1,3=L1,4=L2,5=L2,
# 6=L0,7=L1,8=L2,9=L1,10=L0) — 1차 렌더에서 이 순서를 무시하고 "기준①②③+채택"
# 4개를 전부 L0로 쓰려다 5번 슬롯(L2)에 "기준②"가 들어가 작은 글씨로 밀려나는 등
# 위계가 뒤섞였다(2026-09-21 발견). L0 슬롯이 딱 3개(1,6,10)뿐이라 "기준 소개 →
# 제외된 후보 → 채택" 3단 구성으로 다시 짰다.
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

deck.add_body_2col(
    headerSub="4. 파인튜닝·양자화",
    headline="가볍게 다듬고, 가볍게 압축한다",
    left_title="파인튜닝 (LoRA)",
    left_items=[
        "r=16, alpha=32, bf16",
        "목적 — 현장 알림체로 스타일 학습",
        "안전 도메인 ROUGE-L 0.525→0.719",
        "한계 — 합성 데이터, 실사용 품질보증 아님",
    ],
    right_title="양자화 (GGUF)",
    right_items=[
        "Q4_K_M (K-quant 계열)",
        "7.5GB→2.3GB (약 3.2배 압축)",
        "CPU 23.6 tok/s · GPU 87 tok/s (3.7배)",
        "같은 파일로 CPU/GPU 겸용",
    ],
)

deck.add_body_3col(
    headerSub="5. Agent 원칙",
    headline="판정은 규칙, 서술은 LLM, 고위험은 사람이 승인한다",
    cards=[
        {"title": "규칙 우선", "items": ["초과·위험도 판정은 코드", "장비 제어도 규칙이 즉시 실행", "LLM은 판정을 재수행하지 않음"]},
        {"title": "근거 기반", "items": ["RAG 검색 결과 없이 답하지 않음", "도구 호출 여부는 LLM 자율판단", "할루시네이션 방지 핵심 장치"]},
        {"title": "승인 체계", "items": ["저위험 장비는 자동 실행", "고위험은 사람 승인 대기", "Human-in-the-loop 전 도메인 공통"]},
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

deck.add_body_table(
    headerSub="2. 도메인 확장",
    headline="설비·코퍼스만 교체하면 같은 플랫폼을 재사용한다",
    rows=[
        ["사례", "목적", "구현 상태", "핵심 특징", "재사용 요소"],
        ["산업안전 (원형)", "현장 안전 오프라인 어시스턴트", "완전 구축", "규칙판정+RAG+Agent+엣지에뮬", "-"],
        ["스마트팩토리", "화장품 AX 패턴의 일반화", "개념검증(화장품)", "설비·SOP만 교체, 나머지 재사용", "Agent+RAG 구조 전체"],
        ["AX 에듀테크", "AI 융합교육 플랫폼", "완전 구축(AI튜터·토론)", "ROUGE-L 0.239→0.373 실측", "파인튜닝+RAG 파이프라인"],
        ["AI 추천", "추천 이유 설명 가능한 AI", "미구현(가설 단계)", "판정은 규칙, 설명은 LLM 원칙 적용", "규칙+LLM 역할 분리 원칙"],
    ],
)

deck.add_body_2col(
    headerSub="3. 관통 원칙",
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
        "신규 사업 기회 3건 실증 검토",
    ],
)

deck.set_closing(
    message="MLOps-Edge-Lab — 규칙 기반 판정과 LLM을 결합한 온프레미스 AI 플랫폼",
)

out_path = r"D:\Code-CLI\MLOps-Edge-Lab\docs\교육자료\MLOps-Edge-Lab_교육자료_v3.pptx"
deck.save(out_path)
print("saved:", out_path)
