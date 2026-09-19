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
        {"level": 1, "text": "① 인프라: uv(venv+lock) · MLflow 3.16(SQLite) · tmux"},
        {"level": 1, "text": "② 데이터 추출: 문서 738건 → 텍스트, 형식별 폴백 사슬"},
        {"level": 2, "text": "pymupdf(PDF) · pyhwp(HWP) · LibreOffice(레거시) · Tesseract(OCR)"},
        {"level": 1, "text": "③ 파인튜닝: Qwen3-4B-Instruct-2507 + LoRA(r=16, alpha=32)"},
        {"level": 0, "text": "4~6단계 — 경량화에서 서빙까지"},
        {"level": 1, "text": "④ 양자화: llama.cpp GGUF Q4_K_M (7.5GB→2.3GB, 3.2배)"},
        {"level": 2, "text": "⑤ RAG: e5-small 임베딩 + FAISS(IndexFlatIP), 청크 12,964개"},
        {"level": 1, "text": "⑥ Agent: 규칙(판정) · LLM(문구·도구호출) 역할 분리"},
        {"level": 0, "text": "7~8단계 — 엣지 검증(cgroup 2코어 7.32 tok/s) + 표준 MLOps 4요소(레지스트리·모니터링·DVC·CI)"},
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
# 구조(2026-09-19 재편): 공통 인프라(두 도메인이 같이 씀) → 산업안전 전용 →
# AI튜터 전용 순으로 정리. 챕터 간지는 마스터 템플릿상 Ⅰ/Ⅱ/Ⅲ 셋으로 고정돼
# 있어(4번째 챕터 추가 불가) 챕터 Ⅱ 내부를 headerSub 접두어로 3그룹 구분한다.
deck.add_chapter("Ⅱ", "개발자 레퍼런스 아키텍처")

s = deck.add_body_2col(
    headerSub="공통 1. 인프라 스택 선택",
    headline="새 도구 도입보다 재현성에 필요한 최소한만 추가",
    left_title="검토했지만 보류한 대안",
    left_items=[
        "venv — 가장 단순하지만 lock 파일 없어 재현성 약함",
        "conda — 이 서버 용도엔 무겁고 불필요",
        "파일 기반 실험 기록 — 조건 검색·비교가 불편",
        "systemd user service — 재부팅 생존엔 유리하나 초기 설정 부담",
    ],
    right_title="채택 + 근거",
    right_items=[
        "uv — lock 파일로 버전 고정, MLOps 핵심가치인 재현성 확보",
        "MLflow Tracking(SQLite mlflow.db) — 조건 검색 가능하면서 백업은 파일 하나로 간단",
        "tmux 세션 — SSH 끊김에도 프로세스 생존, 재부팅 대응은 필요해지면 systemd로 승격 예정",
        "FastAPI — 신규 채택 아님, MLflow가 이미 써서 venv에 있던 걸 재사용",
    ],
)
fix_chapter_label(s, "Ⅱ")

s = deck.add_body_3level(
    headerSub="2. 데이터 추출 설계",
    headline="포맷별 폴백 사슬 — 실패해도 조용히 넘어가지 않고 이유를 남긴다",
    items=[
        {"level": 0, "text": "PDF — pymupdf(fitz) 채택"},
        {"level": 1, "text": "BidRadar의 pypdf 대신 선택 — 텍스트 추출 품질 우수"},
        {"level": 1, "text": "HWP — pyhwp(hwp5txt) 1차, LibreOffice 2차 안전망"},
        {"level": 2, "text": "OCR용 페이지→이미지 렌더링이 같은 라이브러리에 있어 poppler 등 외부 도구 불필요(AGPL이지만 사내 학습용이라 허용)"},
        {"level": 2, "text": "BidRadar가 pyhwp만으로 이미 완전 구현·검증해둔 걸 확인 후 재사용 — 더 가볍고 안전"},
        {"level": 0, "text": "레거시(DOC/XLS) — LibreOffice headless로 PDF 변환 후 PDF 추출기 재사용"},
        {"level": 1, "text": "포맷별 전용 파서를 안 만들기 위함 — 'OCR 폴백' 로직을 PDF 경로 한 곳에만 두면 포맷이 늘어도 변환 단계만 추가하면 됨"},
        {"level": 2, "text": "안정성 보강 — 확장자 오표기(매직바이트 PK=zip/CFBF=OLE 재확인)·한글 zip 파일명(CP437→CP949 복원)"},
        {"level": 1, "text": "⚠ 실측: 738/738건 처리 — 성공 판정 기준은 비어있지 않음+글자수 임계값뿐"},
        {"level": 0, "text": "성공 ≠ 정확 — 품질 샘플 검수(사람 대조)는 향후 과제로 명시적으로 남김"},
    ],
)
fix_chapter_label(s, "Ⅱ")

table1_rows = [
    ["모델", "라이선스", "한국어", "크기", "비고"],
    ["Qwen3", "Apache 2.0", "최상위권", "0.6B~32B", "채택 — 엣지~서버 한 계열"],
    ["Gemma 3", "Gemma 라이선스", "양호", "1B~27B", "RAM 효율 최고 (4B=4.2GB)"],
    ["DeepSeek V4", "MIT", "검증 부족", "다양", "한국어 품질 불확실"],
    ["EXAONE 4.0", "NC (비상업)", "최상", "1.2B~32B", "한국어 최상, 상업 불가"],
]
s = deck.add_body_table(
    headerSub="3. 베이스 모델 선택",
    headline="선정 기준(상업 라이선스·한국어 품질·엣지~서버 크기 계열) — Qwen3 채택",
)
fill_table(s, table1_rows)
fix_chapter_label(s, "Ⅱ")

s = deck.add_body_2col(
    headerSub="4. 파인튜닝 방법론",
    headline="LoRA(r=16) — 3B~4B급이 20GB VRAM에 여유 있어 QLoRA는 보류",
    left_title="검토한 대안",
    left_items=[
        "Full Fine-tuning — 전체 파라미터 학습, GPU 메모리·저장공간 부담 큼",
        "QLoRA(4bit) — 더 좁은 VRAM에서 더 큰 모델 학습 시 유리하나 지금은 불필요",
        "Qwen2.5-3B-Instruct — 최초 배관 검증에 사용, 이후 세대 교체",
        "EXAONE·Upstage Solar — 한국어 품질은 좋으나 각각 라이선스(비상업)·크기(엣지 부적합)로 제외",
    ],
    right_title="채택 + 근거",
    right_items=[
        "LoRA r=16, alpha=32, target=q/k/v/o_proj, bf16 — transformers+peft+trl(SFTTrainer)+accelerate",
        "Qwen3-4B-Instruct-2507로 교체 — 같은 Apache-2.0 계열 세대업, 코드 변경은 모델명 한 줄",
        "NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 — RTX 4000(NVLink 없음)에서 GPU간 P2P 통신 시도시 죽는 문제 우회",
        "평가는 ROUGE-L 자동 채점 후 MLflow 기록 — 재학습 자동화(auto_retrain.py)의 판정 기준으로 재사용",
    ],
)
fix_chapter_label(s, "Ⅱ")

s = deck.add_body_2col(
    headerSub="5. 양자화 방식 선택",
    headline="K-quant 계열의 Q4_K_M — '범용 기본값'으로 크기 절반 이하, 품질 92~95% 유지",
    left_title="검토한 대안",
    left_items=[
        "Legacy(Q4_0/Q8_0) — 32개 가중치당 스케일 1개뿐, 같은 크기 기준 K-quant보다 손실 큼",
        "I-quant(IQ4 등) — imatrix 보정 필요, 4비트대에선 이득 작아 보류(2~3비트 갈 때 재고)",
        "Q5_K_M/Q6_K — 품질은 좋지만 절감 폭 작음(2.7~3.3GB)",
        "ONNX/TensorRT — 별도 변환 스택 필요, GPU 특화라 CPU 엣지 대응에 불리",
    ],
    right_title="채택 + 근거",
    right_items=[
        "K-quant 계열 — 레이어별 민감도 반영해 섞어 저장, 같은 크기 기준 legacy보다 우수",
        "Q4_K_M 채택 — 7.5GB→2.3GB(3.2배 실측), 범용 기본값",
        "런타임은 llama.cpp(GGUF) — CPU 전용 1차 빌드 후 CUDA 빌드 병행(build/ vs build-cuda/, 같은 소스)",
        "GGUF 변환 의존성이 torch를 CPU빌드로 강제 설치하려 해서 gguf·sentencepiece만 개별 설치해 회피",
    ],
)
fix_chapter_label(s, "Ⅱ")

s = deck.add_body_stats(
    headerSub="6. 성능 실측치",
    headline="같은 GGUF 파일 하나로 CPU·GPU 모두 대응 (llama-bench, Q4_K_M)",
    stats=[
        {"num": "23.6tok/s", "label": "CPU(Xeon 8스레드) 생성 tg128"},
        {"num": "87.0tok/s", "label": "GPU(RTX4000 Ada) 생성 tg128"},
        {"num": "3.7배", "label": "GPU 가속 배율 (프롬프트 처리는 18.7배)"},
        {"num": "7.3tok/s", "label": "엣지 에뮬레이션(2코어) 실측"},
    ],
)
fix_chapter_label(s, "Ⅱ")

s = deck.add_body_2col(
    headerSub="7. RAG 설계",
    headline="'근거 없으면 모른다' — 프롬프트에 명시해 사실 아닌 답변을 막는다",
    left_title="검토한 대안",
    left_items=[
        "BAAI/bge-m3 — 더 크고 정확하지만 무거움, 검색 품질 부족 시 교체 후보로 남김",
        "Milvus/Chroma/pgvector — 별도 서버 프로세스 필요",
        "문장 단위·고정 길이 청킹 — 문맥이 잘리기 쉬움",
        "재순위화(rerank) 모델 추가 — 정확도는 오르지만 지연시간 증가, 엣지 실시간성과 상충되어 보류",
    ],
    right_title="채택 + 근거",
    right_items=[
        "intfloat/multilingual-e5-small(~470MB) — 엣지에서 실시간 질의해야 해서 가벼운 쪽 우선",
        "FAISS IndexFlatIP — 프로세스 내 인메모리, llama-cpp-python과 같은 '서버 프로세스 없이' 철학",
        "문단 슬라이딩 윈도우(700자, 겹침 100자) — 문맥 보존, 실제 738건→청크 12,964개",
        "근거 문서 없으면 '모른다' + 출처 표시를 시스템 프롬프트에 명시",
    ],
)
fix_chapter_label(s, "Ⅱ")

s = deck.add_body_3level(
    headerSub="8. Agent 설계 원칙",
    headline="위험 판정은 항상 규칙, LLM은 언어(문구·검색)만 담당",
    items=[
        {"level": 0, "text": "원칙 — 판정은 규칙, 서술은 LLM"},
        {"level": 1, "text": "위험도(정상·주의·위험, 임계값 1.5배 placeholder)는 rules.py가 판정, LLM 재판정 금지(시스템 프롬프트 명시)"},
        {"level": 1, "text": "LLM은 search_guidelines 도구 호출 여부·검색어만 스스로 판단"},
        {"level": 2, "text": "파인튜닝 평가서 발견한 'LLM 과대판정' 문제(정상을 초과로 오판)를 이 구조로 해결"},
        {"level": 2, "text": "장비 제어(가스차단기 등)도 규칙이 결정 — LLM에게 안 맡김"},
        {"level": 0, "text": "실전에서 겪은 버그"},
        {"level": 1, "text": "llama-cpp-python이 Qwen3 GGUF의 <tool_call> 태그를 구조화 API로 안 넘겨줌 — 정규식(_TOOL_CALL_RE)으로 직접 파싱"},
        {"level": 2, "text": "위험 사례 근거문서가 상황과 다소 겉도는 RAG 품질 이슈 관찰 — 미해결로 기록만"},
        {"level": 1, "text": "검증: 정상=규칙만 즉시 처리(LLM 호출 없음)·주의=도구호출로 검색+조치문·위험=+에스컬레이션·사고리포트 초안 자동생성"},
        {"level": 0, "text": "3단계 시나리오(정상·주의·위험) 전부 통과 확인"},
    ],
)
fix_chapter_label(s, "Ⅱ")

s = deck.add_body_2col(
    headerSub="9. 장비 제어 안전설계",
    headline="Decision(authority, action, detail) — 모든 결정에 규칙/LLM 권한 태그를 남긴다",
    left_title="저위험 — 규칙이 즉시 자동 실행",
    left_items=[
        "환풍기 가동",
        "소화설비 대기 전환",
        "오작동 비용이 낮아 자동화 허용",
        "LLM 응답을 기다리지 않고 즉시 실행 (/api/judge 실측 ~14ms)",
    ],
    right_title="고위험 — 자동 실행 금지, 승인 대기만",
    right_items=[
        "가스차단기 작동",
        "소화설비 방출",
        "오작동 시 공정 중단·피해가 커서 사람 승인 전제",
        "장비 조치는 LLM이 서술하기 전에 규칙이 먼저 실행 → 결과를 LLM 프롬프트에 '사실'로 주입, LLM이 상태를 지어내지 못하게 함",
    ],
)
fix_chapter_label(s, "Ⅱ")

s = deck.add_body_3level(
    headerSub="10. 엣지 에뮬레이션",
    headline="실 하드웨어 없이 cgroup으로 하한선 추정 — 정확한 예측이 아니라는 점을 명시",
    items=[
        {"level": 0, "text": "할 수 있는 것 vs 못 하는 것"},
        {"level": 1, "text": "cgroup으로 CPU 코어 수·메모리 상한 제한은 실제로 가능 — 하한선 추정에 유효"},
        {"level": 1, "text": "명령어셋(AVX-512 유무)·디스크 I/O·발열 스로틀링은 재현 불가"},
        {"level": 2, "text": "즉 '이 정도는 최소한 된다'는 하한선이지 정확한 예측은 아님"},
        {"level": 2, "text": "방법: systemd-run --user --scope -p CPUQuota=N*100% -p MemoryMax=XG -p MemorySwapMax=0 -- taskset -c 코어들 명령"},
        {"level": 0, "text": "sudo 불필요 — 유저 세션 cgroup으로 동작 확인"},
        {"level": 1, "text": "실측: 2코어/32GB — tg128 7.32 tok/s (8스레드 23.63 대비 약 3.2배 느림)"},
        {"level": 2, "text": "알림 문장 1건 생성에 4~5초 — 실용적인 속도로 판단"},
        {"level": 1, "text": "함정: 모델(2.32GB)보다 작은 메모리 상한(2GB)+스왑 켜짐 → 조용히 무한 대기(30초 타임아웃까지 무출력)"},
        {"level": 0, "text": "MemorySwapMax=0으로 스왑 차단 → 즉시 OOM kill(exit 137, 20초 내)로 깔끔한 실패 확보"},
    ],
)
fix_chapter_label(s, "Ⅱ")

table2_rows = [
    ["요소", "도구", "상태", "핵심 결정", "근거"],
    ["모델 레지스트리", "MLflow Model Registry", "완료", "create_model_version() 직접 호출로 우회 — register_model()이 최신 MLflow에서 Logged Model 요구해 실패", "46번"],
    ["운영 모니터링", "Prometheus+Grafana", "완료", "MLflow와 동일 원칙 — SSH 터널 전용, 8081은 서비스 전용 포트로 아낌", "47번"],
    ["데이터 버저닝", "DVC + 로컬 원격", "완료", ".dvc 포인터만 git 추적, 실 데이터는 dvc push로 ~/dvc-storage에", "48번"],
    ["CI/CD", "GitHub Actions", "테스트 완료·배포 대기", "자체 호스팅 러너 등록 토큰 발급이 보안 정책상 자동 차단 — 사용자 수동 등록 필요", "49번"],
]
s = deck.add_body_table(
    headerSub="11. 표준 MLOps 4요소",
    headline="새 SaaS 도입 대신, 이미 쓰던 도구(MLflow·GitHub)를 확장하는 쪽을 우선",
)
fill_table(s, table2_rows)
fix_chapter_label(s, "Ⅱ")

s = deck.add_body_3level(
    headerSub="12. 재현 가이드",
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

table3_rows = [
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
fill_table(s, table3_rows)
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
