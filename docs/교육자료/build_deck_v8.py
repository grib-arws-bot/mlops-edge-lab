# -*- coding: utf-8 -*-
"""
MLOps-Edge-Lab 교육자료 v8 빌드 스크립트.

v7 대비 변경 핵심 (Claude Docs를 다시 읽지 않고 v7 스크립트의 기존 문구를 그대로 베이스로 삼음 —
이번 세션에서 실제로 구현·실측된 두 가지만 새로 추가/수정한다. 지어낸 수치 없음):
1. 챕터Ⅱ "8. 운영 인프라" 다음에 "8-1. 데이터 드리프트 감지" 신규 서브섹션 슬라이드 추가
   (add_body_3level, 3-1/3-2·5-1·6-1과 같은 서브섹션 넘버링 패턴 재사용).
   화장품 제조AX 제안서에 명시된 "Evidently AI 기반 Drift 감지" 기술 요소를 참고해, 이 프로젝트
   자체의 산업안전 RAG 파이프라인에 실제로 적용해 학습(사업 계약·수주 여부는 이 자료의 범위 밖이라
   언급하지 않는다 — 순수 기술 맥락만 다룸). Evidently AI(PSI)로 reference vs current RAG 검색
   유사도 분포를 비교, 판정은 코드 규칙, Rollback은 사람이 /quality에서 버튼+확인 팝업을 통과해야
   실행. 가장 중요한 실측 발견은 "업계 관례 PSI 임계치(0.1/0.25)가 이 지표엔 안 맞았다" —
   reference 7건일 땐 PSI 계산 자체가 고장(같은 분포를 넣어도 13~17), 116건으로 늘려도 완전히
   같은 분포 두 표본을 비교하는 정상 상황에서 PSI가 0.06~1.69까지 나옴을 40회 반복 실측으로 확인
   → 안정<2/중간2~8/드리프트>8로 임계치를 직접 재보정(기존 0.1/0.25 대체), current 최소 표본도
   10→30으로 올렸다. 이 슬라이드가 32장 기준 22번("8. 운영 인프라") 바로 다음에 들어가며, 이후
   Chapter Ⅲ 전체 슬라이드 인덱스가 한 칸씩 밀린다(챕터Ⅲ 내부 소제목 번호 1~8 자체는 안 바뀜).
2. Chapter Ⅲ "3. 스마트팩토리" 슬라이드 갱신 — 화장품 PoC Agent에 COA(시험성적서) 자동생성 도구를
   추가한 사실만 반영한다(제안서가 "AI 에이전트 핵심가치"로 명시한 기능이라는 배경 설명만 포함,
   계약금액·수주 여부 등 사업/영업 정보는 기술 교육 자료 범위 밖이라 일절 넣지 않음 — 사용자 피드백
   반영). 저장된 값을 문서로 조립만 하고 판정은 SOP 규칙이 계산(LLM이 판정을 바꾸지 못하게 시스템
   프롬프트로 강제), 자동 시뮬레이션이 정상 종료됐을 때도 자동으로 COA 초안이 나오게 확장. "구현
   상태" 문구는 v7 그대로("개념검증(화장품 AX PoC) 단계") 유지하고 건드리지 않는다. EQUIPMENT_MAP·
   SOP_RANGES·Config 기반 확장성 등 기존 내용도 그대로 두되, 10슬롯 고정 구조라 "Human-in-the-loop
   원칙 동일 적용" 한 줄만 "1·4·7·8단계는 산업안전과 동일 구조" 줄에 합쳐 COA 자리를 확보했다.
"""
import sys

sys.path.insert(
    0,
    r"C:\Users\jyahn\.claude\skills\synced\e1336d5b-00ae-4578-bcb3-20eed59b9c78_3719b39e-e72c-4ba0-bdde-9b8c5c919051\grib-ppt\scripts",
)
from copy import deepcopy

from grib_ppt_base import GribPpt, SLIDE_BODY_TABLE
from pptx.util import Inches, Pt, Emu
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import qn

deck = GribPpt()

# ── 표지 ──────────────────────────────────────
deck.set_cover(main_title="MLOps-Edge-Lab", date="2026년 09월")
GribPpt._replace_text_in_slide(deck.prs.slides[0], {"[MLOps-Edge-Lab]": "MLOps-Edge-Lab"})
# 제목이 기본 박스 폭보다 길어 두 줄로 꺾이는 문제가 v4에서도 있었다 — 박스를 넓혀 한 줄로.
for shape in deck.prs.slides[0].shapes:
    if shape.has_text_frame and "MLOps-Edge-Lab" in shape.text_frame.text:
        shape.width = Inches(6.0)

deck.set_toc(["시스템 개요", "기술 구성과 선택 근거", "서비스 적용 사례"])


# ── 공용 헬퍼 ──────────────────────────────────
def add_flow_diagram_slide(headerSub, headline, stages, note):
    """마스터에 흐름도 레이아웃이 없어서 '본문—표' 슬라이드를 복제 후 표를 지우고
    도형+화살표로 직접 그린다(v4에서 만든 패턴 재사용, 문구는 새로 작성)."""
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
    ntf.word_wrap = True
    ntf.text = note
    ntf.paragraphs[0].alignment = PP_ALIGN.CENTER
    for run in ntf.paragraphs[0].runs:
        run.font.size = Pt(13)
        run.font.italic = True
        run.font.color.rgb = RGBColor(0x59, 0x59, 0x59)
    return slide


def add_note_box(slide, text, top_in=6.35, size_pt=11):
    """표/카드 슬라이드 하단에 '정직하게 밝혀둔다'류 캡션을 붙이는 작은 이탤릭 박스."""
    box = slide.shapes.add_textbox(Inches(0.5), Inches(top_in), Inches(12.33), Inches(0.55))
    tf = box.text_frame
    tf.word_wrap = True
    tf.text = text
    for para in tf.paragraphs:
        for run in para.runs:
            run.font.size = Pt(size_pt)
            run.font.italic = True
            run.font.color.rgb = RGBColor(0x59, 0x59, 0x59)
    return box


def _set_cell_text(cell, text, size_pt):
    tf = cell.text_frame
    p = tf.paragraphs[0]
    # 여분 문단 제거
    for extra_p in list(tf.paragraphs[1:]):
        extra_p._p.getparent().remove(extra_p._p)
    if not p.runs:
        p.add_run()
    run0 = p.runs[0]
    for extra_r in list(p.runs[1:]):
        extra_r._r.getparent().remove(extra_r._r)
    run0.text = text
    run0.font.size = Pt(size_pt)


def add_wide_table(headerSub, headline, header_row, data_rows, note=None,
                    header_font=11, data_font=10):
    """9-5 종합비교표 전용 — 마스터 표(헤더+4행)로는 부족해서(헤더+8행 필요),
    표 슬라이드를 복제한 뒤 마지막 데이터 행을 python-pptx로 복제해 행을 늘린다.
    (SKILL.md에 없는 커스텀 확장 — 행 추가 / 폰트축소 / 슬라이드분할 세 방법을 실측 비교한 뒤
    렌더링이 깨지지 않는 '행 추가+소폭 폰트축소' 조합을 채택했다.)"""
    slide = deck._copy_slide(deck.prs, SLIDE_BODY_TABLE)
    deck._apply_chapter_context(slide, headerSub)
    deck._apply_headline(slide, headline)

    table_shape = None
    for shape in slide.shapes:
        if shape.has_table:
            table_shape = shape
            break
    table = table_shape.table
    tbl = table._tbl

    n_data = len(data_rows)
    n_cols = len(header_row)
    existing_data_rows = len(table.rows) - 1  # 마스터 기본 4행
    extra_needed = n_data - existing_data_rows
    if extra_needed > 0:
        template_tr = tbl.tr_lst[-1]
        for _ in range(extra_needed):
            new_tr = deepcopy(template_tr)
            tbl.append(new_tr)
    elif extra_needed < 0:
        for tr in tbl.tr_lst[len(table.rows) + extra_needed:]:
            tbl.remove(tr)

    total_rows = 1 + n_data
    top_in = 2.20
    bottom_limit_in = 6.55 if note else 6.90  # 노트가 있으면 표를 살짝 줄여 캡션과 겹치지 않게
    row_h_in = (bottom_limit_in - top_in) / total_rows
    row_h_emu = Emu(int(Inches(row_h_in)))
    for tr in tbl.tr_lst:
        tr.set("h", str(int(row_h_emu)))
    table_shape.top = Inches(top_in)
    table_shape.height = Emu(int(row_h_emu) * total_rows)

    for c, val in enumerate(header_row):
        _set_cell_text(table.cell(0, c), val, header_font)
    for r, row in enumerate(data_rows, 1):
        for c, val in enumerate(row):
            _set_cell_text(table.cell(r, c), val, data_font)

    if note:
        add_note_box(slide, note, top_in=bottom_limit_in + 0.05, size_pt=10)
    return slide


def add_narrow_table(headerSub, headline, header_row, data_rows, col_widths_in, note=None,
                      header_font=11, data_font=10, top_in=2.20):
    """v7 신규 — AI튜터 8단계 비교표(3열×8행) 전용.
    마스터 표는 5열 고정이라, add_wide_table처럼 행만 늘리는 걸로는 부족하다 — 열도
    5개→3개로 줄여야 "산업안전 원형 vs AI튜터 변경" 두 칸이 충분히 넓어져 긴 문장이 든다.
    OOXML 표는 <a:tblGrid>의 <a:gridCol> 개수와 각 <a:tr>의 <a:tc> 개수가 일치해야 하므로,
    둘 다 동시에 뒤에서부터 잘라내고 남은 gridCol의 폭(w)을 재분배한다."""
    slide = deck._copy_slide(deck.prs, SLIDE_BODY_TABLE)
    deck._apply_chapter_context(slide, headerSub)
    deck._apply_headline(slide, headline)

    table_shape = None
    for shape in slide.shapes:
        if shape.has_table:
            table_shape = shape
            break
    table = table_shape.table
    tbl = table._tbl

    # 1) 열 개수를 목표(3)에 맞춰 줄이고 폭 재분배
    target_n = len(header_row)
    grid = tbl.find(qn("a:tblGrid"))
    gridCols = grid.findall(qn("a:gridCol"))
    if len(gridCols) > target_n:
        for gc in gridCols[target_n:]:
            grid.remove(gc)
        for tr in tbl.findall(qn("a:tr")):
            tcs = tr.findall(qn("a:tc"))
            for tc in tcs[target_n:]:
                tr.remove(tc)
    gridCols = grid.findall(qn("a:gridCol"))
    for gc, w_in in zip(gridCols, col_widths_in):
        gc.set("w", str(int(Inches(w_in))))

    # 2) 행 개수를 데이터 행 수에 맞춰 늘림 (add_wide_table과 동일 패턴)
    n_data = len(data_rows)
    existing_data_rows = len(table.rows) - 1
    extra_needed = n_data - existing_data_rows
    if extra_needed > 0:
        template_tr = tbl.tr_lst[-1]
        for _ in range(extra_needed):
            new_tr = deepcopy(template_tr)
            tbl.append(new_tr)
    elif extra_needed < 0:
        for tr in tbl.tr_lst[len(table.rows) + extra_needed:]:
            tbl.remove(tr)

    total_rows = 1 + n_data
    bottom_limit_in = 6.55 if note else 6.90
    row_h_in = (bottom_limit_in - top_in) / total_rows
    row_h_emu = Emu(int(Inches(row_h_in)))
    for tr in tbl.tr_lst:
        tr.set("h", str(int(row_h_emu)))
    table_shape.top = Inches(top_in)
    table_shape.height = Emu(int(row_h_emu) * total_rows)
    table_shape.width = Inches(sum(col_widths_in))

    for c, val in enumerate(header_row):
        _set_cell_text(table.cell(0, c), val, header_font)
    for r, row in enumerate(data_rows, 1):
        for c, val in enumerate(row):
            _set_cell_text(table.cell(r, c), val, data_font)

    if note:
        add_note_box(slide, note, top_in=bottom_limit_in + 0.05, size_pt=10)
    return slide


# ══════════════════════════════════════════════
# 챕터 Ⅰ. 시스템 개요
# ══════════════════════════════════════════════
deck.add_chapter("Ⅰ", "시스템 개요")

deck.add_body_stats(
    headerSub="1. 핵심 지표",
    headline="온프레미스·엣지 환경에서 규칙 기반 판정과 LLM을 결합한 sLLM 플랫폼",
    stats=[
        {"num": "738건", "label": "안전문서 추출 (99.9%)"},
        {"num": "273,595개", "label": "RAG 청크 (MSDS 통합 후)"},
        {"num": "3.2배", "label": "양자화 압축 (7.5GB→2.3GB)"},
        {"num": "3.7배", "label": "GPU 추론 가속(tg128)"},
    ],
)

deck.add_body_2col(
    headerSub="2. 사업 배경",
    headline="정부과제·잠재고객의 온프레미스 sLLM 수요와 완전 폐쇄망 목표가 만나는 지점",
    left_title="왜 시작했나",
    left_items=[
        "정부과제·잠재고객 sLLM 수요 지속 증가",
        "AI 성능·데이터 보안 관점 온프레미스·엣지 요구 확대",
        "수자원공사·강원AX화장품·클랙스 — 실제 수요 확인",
        "1인 개발, 온프레미스 GPU 서버 1대(2026-09-17 시작)",
    ],
    right_title="왜 온프레미스·오프라인 우선인가",
    right_items=[
        "완전 폐쇄망에서도 동작하는 오프라인 안전 어시스턴트가 목표",
        "sLLM은 중심이 아니라 부가 기능으로 요청되는 경우가 더 많음",
        "AIoT 플랫폼 Thing-X와는 목적·기술이 근본적으로 다름",
        "두 솔루션은 역할이 달라 오히려 융합이 용이",
    ],
)

add_flow_diagram_slide(
    headerSub="3. 전체 흐름",
    headline="데이터 추출부터 운영까지 8단계, 오판정·성능저하는 재학습·재색인으로 되돌린다",
    stages=["1.개발환경", "2.데이터추출", "3.모델+파인튜닝", "4.양자화", "5.RAG", "6.Agent", "7.엣지에뮬", "8.운영인프라"],
    note="↺ 6·8단계의 오판정·성능저하 피드백 → 3단계 재학습 / 5단계 재색인으로 순환",
)

deck.add_body_3level(
    headerSub="4. 목표와 원칙",
    headline="두 가지 목표를 관통하는 설계 원칙",
    items=[
        {"text": "목표 ① MLOps 파이프라인 구축"},
        {"text": "실험관리(MLflow)→학습→평가 순환"},
        {"text": "배포(모델 레지스트리)→모니터링(Prometheus/Grafana)"},
        {"text": "평가 기준 미달 시 자동 재학습으로 복귀"},
        {"text": "CI/CD로 테스트~배포 27초 자동화"},
        {"text": "목표 ② 엣지 sLLM 구축"},
        {"text": "양자화(Q4_K_M)로 경량화, GPU 없이도 동작"},
        {"text": "cgroup 에뮬레이션으로 저사양 환경 하한선 추정"},
        {"text": "규칙+LLM 역할 분리로 안전한 자동 판단"},
        {"text": "관통 원칙 — 근거 없으면 모른다고 답하고, 실패도 숨기지 않고 기록한다"},
    ],
)

# ══════════════════════════════════════════════
# 챕터 Ⅱ. 기술 구성과 선택 근거
# ══════════════════════════════════════════════
deck.add_chapter("Ⅱ", "기술 구성과 선택 근거")

deck.add_body_table(
    headerSub="1. 개발 환경",
    headline="실험 재현성과 배포 안정성을 우선한 개발 환경",
    rows=[
        ["결정", "선택", "이유", "비고", "상태"],
        ["실행 위치", "AI 서버(grib-ai-server) 하나로 통일", "학습·추론·서빙을 로컬에서 돌리지 않는다는 원칙", "로컬 PC는 편집 전용", "완료"],
        ["환경관리자", "uv (venv/conda 대신)", "lock 파일로 버전 고정 — 재현성 우선", "MLOps 핵심가치=재현성", "완료"],
        ["MLflow backend", "SQLite(mlflow.db)", "조건 검색(실험 비교)에 유리하면서 백업은 파일 하나로 간단", "파일 기반보다 비교 유리", "완료"],
        ["상시 실행", "tmux → systemd --user로 승격", "재부팅에도 살아남게", "Restart=always", "완료"],
    ],
)

deck.add_body_table(
    headerSub="2. 데이터 추출",  # v6→v7: "2. 데이터 추출 파이프라인"(15자)이 배지에서 2줄로 꺾여 축약
    headline="포맷별 폴백 사슬 + '조용한 빈 결과 금지' 원칙",
    rows=[
        ["결정", "선택", "이유", "비고", "실측"],
        ["PDF 라이브러리", "pymupdf(fitz)", "OCR용 페이지→이미지 렌더링이 같은 라이브러리에 있어 외부도구 불필요", "poppler 불필요", "-"],
        ["레거시 포맷(HWP/DOC/XLS)", "LibreOffice headless→PDF 변환 후 기존 추출기 재사용", "포맷별 전용 파서를 따로 안 만들어도 됨", "-", "-"],
        ["HWP", "pyhwp 1차, LibreOffice 2차 안전망", "더 가볍고 변환 품질이 안정적임을 확인", "-", "-"],
        ["확장자 오표기 대응", "매직바이트(PK=zip, CFBF=OLE)로 실제 구조 재확인", "확장자만으로는 실제 구조를 신뢰할 수 없는 사례 발견", "예: .hwpx인데 실제 구버전 OLE", "738/738건 성공(99.9%)"],
    ],
)
add_note_box(
    deck.prs.slides[-1],
    "정직하게 남겨둔 한계 — \"738건 100% 성공\"은 품질 검증이 아니다. 판정 기준이 '빈 문자열이 아니고 글자수 임계값을 넘음'뿐이라 "
    "OCR 오탈자·머리말만 추출돼도 통과할 수 있다. 샘플 검수는 아직 미착수.",
)

deck.add_body_3level(
    headerSub="3-1. 모델 선정 기준",
    headline="라이선스·한국어 품질·엣지 배포 가능 크기 — 세 기준으로 Qwen3 채택",
    items=[
        {"text": "채택 기준 3가지"},
        {"text": "① 상업적 이용 가능한 라이선스"},
        {"text": "② 한국어 품질"},
        {"text": "③ 엣지 배포 가능한 크기(1~3B급)"},
        {"text": "국적·회사에 따라 한국어 실력과 상업조건이 크게 갈림"},
        {"text": "제외된 후보"},
        {"text": "EXAONE 4.0 — 한국어 최상, 비상업 전용이라 제외"},
        {"text": "Upstage Solar — 최소 10.7B부터, 엣지에 부적합"},
        {"text": "DeepSeek V4 — 라이선스는 깨끗하나 한국어 신뢰도 불확실"},
        {"text": "채택 — Qwen3(Apache 2.0, 0.6B~32B+MoE로 선택지 넓음)"},
    ],
)

deck.add_body_table(
    headerSub="3-1. 모델 비교표",
    headline="4개 후보의 라이선스·한국어 품질 정량 비교",
    rows=[
        ["모델", "기업/국가", "라이선스", "한국어 품질", "결과"],
        ["Qwen3", "Alibaba·중국", "Apache 2.0(제약 없음)", "다국어 최상위권", "채택"],
        ["EXAONE 4.0", "LG AI연구원·한국", "비상업 전용", "한국어 최상", "제외(라이선스)"],
        ["Upstage Solar", "Upstage·한국", "조건부 상업 허용", "우수", "제외(최소 10.7B)"],
        ["Gemma 3/4", "Google·미국", "조건부 상업 허용", "양호", "차선책(4B 메모리효율 최고)"],
    ],
)

deck.add_body_2col(
    headerSub="3-2. 파인튜닝 스택과 실측",
    headline="현장 알림체를 소량 예시로 가르치는 LoRA",
    left_title="왜 필요했나",
    left_items=[
        "기본 모델은 일반 문장은 잘 만들지만 원하는 형식은 예시로 가르쳐야 함",
        "목표: \"3동 2층 화기작업구역, 미착용 안전모 3건 감지\"류 알림체",
        "LoRA — 전체가 아닌 보정 레이어만 학습, r=16 alpha=32, bf16",
        "RTX 4000은 NVLink 없어 NCCL_P2P_DISABLE=1 상시 필요(멀티GPU)",
    ],
    right_title="효과 — 정직하게 두 가지로",
    right_items=[
        "배관검증용 — 합성 32건 5에포크 ROUGE-L 0.986, 품질지표 아님",
        "안전 도메인 실제 개선폭 — ROUGE-L 0.525→0.719(+0.194)",
        "에듀 도메인 실제 개선폭 — ROUGE-L 0.239→0.373",
        "한계 — 템플릿 합성 데이터라 실사용 품질보증은 아님",
    ],
)

deck.add_body_table(
    headerSub="4. 양자화 방식",
    headline="K-quant 계열에서 크기·품질 균형점을 찾는다",
    rows=[
        ["방식", "크기(4B 기준)", "품질 손실", "비고", "선택"],
        ["F16(무양자화)", "~8.0GB", "없음(기준)", "GPU 서버 학습용", "-"],
        ["Q5_K_M", "~2.7GB", "적음", "정확도 민감 태스크 대안", "-"],
        ["Q4_K_M", "~2.4GB(실측)", "적당", "Legacy보다 동일크기서 품질 우위", "채택 · 범용 기본값"],
        ["Q3_K_M", "~1.9GB", "눈에 띔", "RAM 빠듯한 엣지용", "-"],
    ],
)

deck.add_body_2col(
    headerSub="4. 양자화 실측",
    headline="같은 GGUF 파일 하나로 CPU·GPU 둘 다 정상 동작",
    left_title="압축 전후",
    left_items=[
        "f16 7.5GB → Q4_K_M 2.3GB(약 3.2배 압축)",
        "LoRA는 merge_and_unload()로 먼저 병합 후 GGUF 변환",
        "llama.cpp CPU 전용 빌드(nvcc 없음, 엣지 목적에도 부합)",
        "CPU에서 정확한 문장 생성 확인",
    ],
    right_title="CPU vs GPU 벤치마크(llama-bench)",
    right_items=[
        "CPU(Xeon Silver 4510, 8스레드): tg128 23.63 t/s",
        "GPU(RTX 4000 SFF Ada): tg128 87.01 t/s",
        "생성 속도 약 3.7배, 프롬프트 처리는 약 18.7배",
        "디바이스마다 모델을 따로 만들 필요가 없어짐",
    ],
)

deck.add_body_table(
    headerSub="5. RAG 파이프라인",
    headline="근거 없이는 답하지 않는 검색 결합 생성",
    rows=[
        ["구성요소", "기술", "역할", "실측", "비고"],
        ["청킹", "문단 슬라이딩윈도우", "문서를 검색 단위로 분할", "12,964개 청크(738건 기준)", "MSDS 통합 전 원본"],
        ["임베딩", "multilingual-e5-small", "의미 기반 벡터화", "-", "다국어 지원"],
        ["검색", "FAISS", "최근접 벡터 검색", "hit_rate@4 = 0.857", "골든셋 평가(2026-09-20)"],
        ["생성", "sLLM(Qwen3)", "검색 결과 근거로만 답변 생성", "근거 없으면 모른다고 답함", "할루시네이션 방지 핵심"],
    ],
)
add_note_box(
    deck.prs.slides[-1],
    "곁가지 버그 — PDF 텍스트의 U+2028을 splitlines()가 줄바꿈으로 오인해 JSONL이 반쪽으로 잘리는 문제를 발견, "
    "쓰기·읽기 모두 \"\\n\" 명시 분할로 고정했다.",
)

deck.add_body_3level(
    headerSub="5-1. MSDS 통합",
    headline="산업안전 RAG를 화학물질 안전정보까지 확장한다 (신규 섹션)",
    items=[
        {"text": "목적 — 산업안전 RAG를 화학물질 안전정보(MSDS)까지 확장"},
        {"text": "KOSHA가 물질별 16개 표준 항목으로 관리하는 공식 문서"},
        {"text": "데이터 확보 — data.go.kr 공식 API는 검색 전용, 전체 목록 나열 불가 확인"},
        {"text": "대안 — 동일 API 기반 공개 데이터셋(HF, 48,966개 물질) 사용, SHA256 무결성 검증"},
        {"text": "라이선스 — 데이터셋 CC BY 4.0, 원본은 공공누리 제1유형(자유이용)"},
        {"text": "결과 — 273,595개 청크로 재구축(기존 12,964개 대비 약 21배)"},
        {"text": "서비스 페이지(/msds) — ① 물질 선택 즉시조회 ② 전체 소스 RAG 검색"},
        {"text": "근거 없으면 '찾지 못했다'는 원칙 동일 적용"},
        {"text": "자동 갱신 — HF 최신 커밋을 주기 조회해 새 버전만 감지"},
        {"text": "재수집은 사람이 버튼으로 직접 시작(인덱스 재구축 GPU 약 20분)"},
    ],
)

deck.add_body_3col(
    headerSub="6. Agent 원칙",
    headline="판정은 규칙, 서술은 LLM, 고위험은 사람이 승인한다",
    cards=[
        {"title": "규칙 우선", "items": ["초과·위험도 판정은 코드(rules.py)", "장비 제어도 규칙이 즉시 실행", "LLM은 판정을 재수행하지 않음"]},
        {"title": "근거 기반", "items": ["search_guidelines 호출은 LLM 자율판단", "RAG(5단계) 재사용, 근거 없이 답하지 않음", "위험 등급에도 강제 호출 안 해도 재량 설계 확인"]},
        {"title": "승인 체계", "items": ["저위험 장비(환풍기 등) 자동 실행", "고위험(가스차단기·소화설비) 사람 승인 대기", "운영 피드백은 사람이 2단계로 직접 실행"]},
    ],
)

deck.add_body_table(
    headerSub="6. Agent 역할 분리",
    headline="같은 상황도 담당 주체에 따라 처리 방식이 다르다",
    rows=[
        ["역할", "담당", "설명", "비고", "권한"],
        ["초과·위험도 판정", "규칙(rules.py)", "정상/주의/위험 3단계 계산", "LLM이 다시 판정 안 함", "코드"],
        ["도구 호출 여부·검색어", "LLM", "search_guidelines 재사용", "자율판단에 맡김", "LLM"],
        ["알림/에스칼레이션/보고서", "코드", "위험도에 따라 직접 호출", "LLM이 스스로 결정 안 함", "코드"],
        ["장비 제어(환풍기·가스차단기 등)", "규칙+사람", "저위험 자동, 고위험 승인대기", "임의 실행 방지", "코드+사람"],
    ],
)

deck.add_body_table(
    headerSub="6-1. 센서 데이터",
    headline="문서 추출이 아니라 여기 속한다 — 이 값을 소비하는 건 Agent의 규칙 판정이기 때문",
    rows=[
        ["카테고리", "측정 물질", "단위 예", "임계값 근거", "비고"],
        ["공기질", "온도·습도·PM10·PM2.5·CO2·TVOC", "℃/%/㎍/m³/ppm", "KOSHA 사무실 공기관리지침 등", "6개 물질"],
        ["가스", "산소(O2)·수소", "%/%LEL", "산업안전보건기준 제618조", "산소는 낮을수록 위험(판정 방향 반대)"],
        ["조리흄", "포름알데히드 등 9종", "ppm/ng·mg/m³", "고용노동부 노출기준 고시", "4종은 공식기준 없어 근사값"],
        ["구조 변경", "헬륨·질소·아르곤 → 산소(O2) 통합", "%", "단순 질식제는 개별기준 없음", "산소 18% 미만=산소결핍"],
    ],
)
add_note_box(
    deck.prs.slides[-1],
    "아직 실제 센서 장비는 없다 — 카테고리·임계값은 실제 법령 기준으로 교체했지만, 값 자체는 여전히 무작위 생성(웹 데모) 또는 합성 템플릿(학습 데이터)이다.",
)

deck.add_body_3level(
    headerSub="7. 엣지 에뮬레이션",
    headline="실제 하드웨어 없이 성능 하한선을 추정한다",
    items=[
        {"text": "실제 하드웨어 없이 cgroup(systemd-run)으로 제약을 흉내"},
        {"text": "재현 가능 — CPU 코어 수, 메모리 상한"},
        {"text": "재현 불가 — 명령어셋(AVX-512), 디스크 I/O, 발열 스로틀링"},
        {"text": "그래서 정확한 예측이 아니라 '최소한 이만큼은 된다'는 하한선"},
        {"text": "실측 — 2코어/32GB 제약에서 tg 7.32 t/s"},
        {"text": "발견된 버그"},
        {"text": "메모리 상한이 모델보다 작으면 스왑 때문에 조용히 멈춤(OOM 아님)"},
        {"text": "MemorySwapMax=0으로 해결"},
        {"text": "이 설정 없인 '멈춤'을 OOM 크래시로 오인하기 쉬움"},
        {"text": "다음 단계 — 실제 엣지 하드웨어 확보 후 재검증"},
    ],
)

deck.add_body_table(
    headerSub="8. 운영 인프라",
    headline="표준 MLOps 4대 갭을 순차적으로 메웠다",
    rows=[
        ["갭", "도구", "역할(실제로 하는 일)", "비고", "상태"],
        ["모델 레지스트리", "MLflow Model Registry", "기준 통과 모델만 Production으로 승격", "train/auto_retrain.py", "완료"],
        ["모니터링", "Prometheus + Grafana", "상태·요청수·응답속도 수집 후 대시보드화", "/metrics 엔드포인트", "완료"],
        ["데이터 버저닝", "DVC", "대용량 학습데이터 버전 관리, git엔 포인터만", "로컬 원격(~/dvc-storage)", "완료"],
        ["CI/CD", "GitHub Actions", "push마다 테스트→통과 시 자동 배포", "테스트~배포 27초", "완료"],
    ],
)

deck.add_body_3level(
    headerSub="8-1. 데이터 드리프트 감지",
    headline="업계 관례 임계치를 그대로 믿지 않고, 이 지표의 '정상 분포'를 직접 재봤다 (신규)",
    items=[
        {"text": "배경 — 화장품 제조AX 제안서가 명시한 'Evidently AI 기반 Drift 감지'를 이 프로젝트 파이프라인에 직접 적용"},
        {"text": "화장품 PoC가 아니라 MLOps-Edge-Lab 자신의 산업안전 RAG 파이프라인에 실제 적용해 학습(핵심 포인트)"},
        {"text": "구성 — reference(기준 분포) vs current(현재 분포)의 top-1 검색 유사도를 Evidently(PSI)로 비교"},
        {"text": "판정(PSI 임계치 초과 여부)은 코드 규칙이 정하고 LLM은 관여하지 않음"},
        {"text": "Rollback(MLflow Model Registry 이전 버전 복귀)은 사람이 /quality에서 버튼+확인 팝업을 통과해야 실행 — 자동 트리거 없음('판정은 규칙, 실행은 사람 승인' 원칙 재사용)"},
        {"text": "가장 중요한 실측 발견 — 업계 관례 PSI 임계치(<0.1 안정/0.1~0.25 드리프트)는 표본 잡음만으로도 넘는 수준이라 못 썼다"},
        {"text": "reference 7건일 땐 PSI 계산 자체가 사실상 고장 — current에 뭘 넣어도(완전히 같은 분포를 넣어도) PSI가 13~17까지 튐"},
        {"text": "reference를 116건(코퍼스 기반 LLM 초안 질문, 점수 분포 전용)으로 늘려도, 완전히 같은 분포에서 뽑은 두 표본을 비교하는 정상 상황에서 PSI가 0.06~1.69까지 나옴을 40회 반복 실측으로 확인 → 새 임계치로 재보정: 안정<2 / 중간 2~8 / 드리프트>8(기존 0.1/0.25 대체), current 최소 표본 수도 10→30으로 상향"},
        {"text": "실측 결과 — 배포 직후 실트래픽 적어 '데이터 부족(판단 불가)' 정직하게 표시(가짜로 '정상' 표시 안 함), 인위적 이동 데이터 주입 시 status=drift·PSI=12.39(reference 평균 0.90 vs current 평균 0.80), Rollback 시도는 되돌릴 Archived 버전이 없어 정상적으로 거부(409)됨까지 확인"},
        {"text": "교훈 — 업계 표준 임계치를 그대로 믿지 말고, 내 지표의 '정상일 때' 분포를 직접 재봐야 한다"},
    ],
)

# ══════════════════════════════════════════════
# 챕터 Ⅲ. 서비스 적용 사례
# ══════════════════════════════════════════════
deck.add_chapter("Ⅲ", "서비스 적용 사례")

deck.add_body_3col(
    headerSub="1. 사업 기회 개요",
    headline="산업안전 외 4개 사례로 플랫폼 재사용성을 확인 — 실제 구축은 산업안전·교육 AX뿐",
    cards=[
        {"title": "제조 AX — 스마트팩토리", "items": ["화장품 AX PoC 패턴의 일반화", "설비·SOP만 교체하면 재사용", "개념검증 단계"]},
        {"title": "교육 AX — AI튜터·AI토론", "items": ["클랙스 요청 기반 AI융합교육", "완전 오프라인, 실제 구축완료", "튜터·토론 2개 실사례로 분리"]},
        {"title": "BidRadar — 사내 실운영", "items": ["별개 인프라, sLLM API만 호출", "5,299건 호출, 성공률 100%", "가설이 아닌 실제 운영 데이터"]},
    ],
)

deck.add_body_3level(
    headerSub="2. MSDS 통합",
    headline="산업안전 sLLM을 화학물질 영역까지 실사용 서비스로 확장한다 (신규)",
    items=[
        {"text": "목적 — 기존 산업안전 sLLM 서비스를 화학물질(MSDS) 영역까지 실사용자용 기능으로 확장"},
        {"text": "챕터Ⅱ 5-1은 데이터 확보·갱신 로직 등 파이프라인 결정, 이 슬라이드는 실사용 서비스 관점"},
        {"text": "서비스 페이지(/msds)"},
        {"text": "① 물질 선택 — 카탈로그 즉시조회"},
        {"text": "② 전체 소스 검색 — 산업안전+MSDS 통합 RAG 검색 + sLLM 답변 생성"},
        {"text": "데이터 규모 — 48,966개 화학물질, 16개 표준항목, 총 273,595개 청크(기존 대비 약 21배)"},
        {"text": "근거 없으면 '찾지 못했다'는 이 프로젝트 원칙 동일 적용(할루시네이션 방지)"},
        {"text": "자동 갱신 — HF 최신 커밋 주기 조회, 새 버전 있을 때만 '지금 업데이트' 버튼 활성화"},
        {"text": "재수집은 사람이 시작(버튼 클릭)"},
        {"text": "구현 상태 — 실제 구축·배포 완료(/msds 실사용 가능)"},
    ],
)

deck.add_body_3level(
    headerSub="3. 스마트팩토리",
    headline="설비·SOP만 교체하면 같은 Agent 구조를 재사용한다 (9-1)",
    items=[
        {"text": "목적 — 화장품 AX PoC 패턴을 임의 제조현장으로 일반화"},
        {"text": "설비군·센서 종류·SOP만 교체하면 같은 플랫폼 재사용"},
        {"text": "EQUIPMENT_MAP·SOP_RANGES가 이미 설정 기반 구조로 설계됨"},
        {"text": "제안서에도 'Config 기반 아키텍처로 확장 반영' 명시"},
        {"text": "COA(시험성적서) 자동생성 추가(신규) — 제안서가 'AI 에이전트 핵심가치'로 명시한 기능, 저장된 값을 문서로 조립만 하고 판정은 SOP 규칙이 계산(LLM이 판정을 바꾸지 못하게 시스템 프롬프트로 강제)"},
        {"text": "단계별 변경"},
        {"text": "데이터=설비 매뉴얼·SOP, 센서=PLC/OPC-UA 시계열+MES 실적"},
        {"text": "3단계=설비 이상 서술체 재학습, 6단계=예지보전 알람+승인"},
        {"text": "1·4·7·8단계는 산업안전과 동일 구조, Human-in-the-loop 원칙도 동일 적용 — 자동 시뮬레이션이 정상 종료됐을 때도 COA 초안이 자동으로 나오게 확장"},
        {"text": "구현 상태 — 개념검증(화장품 AX PoC) 단계"},
    ],
)

add_narrow_table(
    headerSub="4. AI 튜터",
    headline="학교 네트워크 제약에 대응하는 완전 오프라인 튜터 — 8단계 중 4곳만 바뀐다 (9-2, 신규 분리)",
    header_row=["단계", "산업안전 원형", "AI튜터 변경"],
    data_rows=[
        ["1. 개발환경", "uv/MLflow", "동일"],
        ["2. 데이터추출", "안전 문서 738건",
         "교과서·교육과정 참고자료(현재 42건, data.go.kr 공공데이터로 확장 진행 중)"],
        ["3. 모델선정/파인튜닝", "동일 베이스, 안전 알림체",
         "동일 베이스(edge-social-lora), 학생 눈높이 설명체 — 실측 ROUGE-L 0.239→0.373"],
        ["4. 양자화", "동일", "동일"],
        ["5. RAG", "안전 법령/수칙", "교과서 지식(경제·지리·법 등), BM25+임베딩 하이브리드 검색(RRF)"],
        ["6. Agent", "규칙판정+장비제어+LLM서술",
         "장비제어 없음 — 대신 \"오답노트 생성+학습진도 추적\"이 그 자리를 대체"],
        ["7. 엣지에뮬", "서버 cgroup", "동일 프레임워크, 학교 PC/태블릿 사양으로 치환"],
        ["8. 운영인프라", "동일", "동일"],
    ],
    col_widths_in=[1.7, 3.9, 6.7333],
    note="한계 — 코퍼스 2소스·42건뿐이라 파인튜닝 개선폭이 문체 때문인지 내용 정확도 때문인지 아직 "
         "구분되지 않는다. 구현 상태 — 실제 구축·서비스 완료(완전 오프라인 학생용 페이지).",
)

deck.add_body_3level(
    headerSub="5. AI 토론",
    headline="쿼터 기반 팀 배정, Sankey→막대그래프로 시각화 변경 (9-3, 신규 분리)",
    items=[
        {"text": "목적 — 구상만 있던 'AI 토론 배틀'의 실제 구현"},
        {"text": "흐름 — 주제생성→의견 에뮬레이션→팀배정→RAG 토론자료→재의견→시각화"},
        {"text": "팀 배정 — 의견 임베딩 후 균형 클러스터링, LLM은 이름만 라벨링"},
        {"text": "찬성/반대/조건부 비율은 코드가 쿼터로 먼저 고정"},
        {"text": "이유 — '다양하게 만들라'는 프롬프트만으론 한쪽 쏠림이 실제로 있었음"},
        {"text": "실제 예시(2026-09-21) — 급식실 식재료 주제"},
        {"text": "찬성 5명이 '지역경제' vs '환경/품질' 2팀으로 자동 분리"},
        {"text": "같은 입장도 강조점이 다르면 다른 팀으로 갈릴 수 있음을 보여줌"},
        {"text": "시각화 변경 — Sankey 흐름도 → 팀별 막대그래프(전/후+신규유입)"},
        {"text": "구현 상태 — 실제 구축·서비스 완료(파인튜닝 없이 프롬프팅만)"},
    ],
)

deck.add_body_table(
    headerSub="6. BidRadar",
    headline="산업안전과 완전히 별개인 사내 프로젝트가 이 프로젝트의 서빙 API를 실제로 호출한다 (9-4, 신규)",
    rows=[
        ["항목", "내용", "실측", "엔드포인트", "비고"],
        ["공고 매칭", "공고 제목 ↔ 고객 관심주제 의미적 매칭", "5,299건 호출, 성공률 100%", "/v1/classify-topic", "평균 지연 1.6초"],
        ["보조 신호", "키워드 매칭을 대체하지 않음", "confidence는 사람 검토용", "-", "기존 매칭과 병행"],
        ["병렬 처리", "전용 GPU 풀(GPU 2·3)로 동시 처리", "4워커", "-", "실시간 서비스 GPU와 분리"],
        ["문서 분류", "첨부문서가 공통서식인지 판단", "185건 호출(성공 97·실패 88)", "/v1/classify-doc", "실패율 아직 높음"],
    ],
)
add_note_box(
    deck.prs.slides[-1],
    "정직하게 밝혀둔다 — 이 사례는 사고실험이 아니라 실제 운영 데이터지만 '추천'보다 '매칭 후보 제시'에 가깝다. "
    "최종 결정은 여전히 사람(키워드 매칭+검토)이 하며, classify-doc 실패율(185건 중 88건)이 아직 높다는 점도 그대로 남겨둔다.",
)

add_wide_table(
    headerSub="7. 네 사례 종합 비교",
    headline="8단계 구조가 얼마나 그대로 재사용되는가 — BidRadar는 나머지 셋과 성격이 다르다 (9-5, 신규)",
    header_row=["단계", "스마트팩토리", "AI튜터", "AI토론", "BidRadar"],
    data_rows=[
        ["1. 개발환경", "동일", "동일", "동일", "해당없음(별도 인프라)"],
        ["2. 데이터추출", "변경(설비 매뉴얼·SOP)", "변경(교과서·공공데이터)", "동일(AI튜터와 코퍼스 재사용)", "해당없음"],
        ["3. 파인튜닝", "변경(설비 이상 서술체)", "변경(학생 눈높이 설명체)", "해당없음(프롬프팅만 사용)", "해당없음(재학습 없음)"],
        ["4. 양자화", "동일", "동일", "동일", "해당없음(양자화된 모델 그대로 호출)"],
        ["5. RAG", "변경(SOP 검색)", "변경(교과서 하이브리드 검색)", "변경(토론주제 교과서 검색)", "변형(임베딩 매칭으로 대체)"],
        ["6. Agent", "동일 구조(임계값+승인)", "변경(오답노트 생성)", "장비제어 없음(팀배정·재의견)", "변형(키워드 보조, 최종은 사람)"],
        ["7. 엣지에뮬", "동일(현장PC 흉내)", "동일(학교PC·태블릿)", "해당없음(서버 웹서비스)", "해당없음(엣지 배포 없음)"],
        ["8. 운영인프라", "동일", "동일", "동일", "별도(분리 인프라, API만 호출)"],
    ],
    note="BidRadar는 '같은 플랫폼에 다른 도메인을 넣은 확장'이 아니라, 이미 만든 5·6단계 결과물을 완전히 별개 시스템이 "
         "외부에서 호출하는 구조다 — 그래서 1·2·3·4·7·8단계는 '변경'이 아니라 '해당없음'이 정확하다.",
)

deck.add_body_2col(
    headerSub="8. 관통 원칙",
    headline="네 실사례를 관통하는 원칙과 다음 확장 방향",
    left_title="관통한 설계 원칙",
    left_items=[
        "규칙 우선, LLM은 보조",
        "근거 없으면 모른다고 답함",
        "주장보다 실측 수치",
        "실패도 숨기지 않고 기록",
    ],
    right_title="앞으로의 확장 방향",
    right_items=[
        "실제 엣지 하드웨어 확보 후 7단계 재검증",
        "센서 임계값을 placeholder에서 정식 기준으로 완전 교체",
        "6→2/3단계 피드백 루프를 자동 재학습까지 확장",
        "아직 가설 단계인 부분의 실증 확대",
    ],
)

deck.set_closing(
    message="MLOps-Edge-Lab — 규칙 기반 판정과 LLM을 결합해 4개 도메인으로 검증된 온프레미스 AI 플랫폼",
)

out_path = r"D:\Code-CLI\MLOps-Edge-Lab\docs\교육자료\MLOps-Edge-Lab_교육자료_v8.pptx"
deck.save(out_path)
print("saved:", out_path)
print("total slides:", len(deck.prs.slides))
