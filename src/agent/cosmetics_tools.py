"""화장품 제조 AX PoC(강원정보문화산업진흥원 제안서 Layer 4 검증용) — Agent가 호출하는
도구 4종과, 그 아래 Layer 1~3을 흉내 내는 mock 데이터.

**이 파일 전체가 concept 검증용이다** — 나중에 실제 프로젝트를 시작하면 완전히 새로운
서버에 다시 구축할 예정이라, 여기서는 프로덕션 수준의 견고함을 추구하지 않는다
(사용자 결정, 2026-09-20).

**Layer 1~3을 흉내 내는 방식**:
- Layer 1(장비)+Layer 2(RDB) → `LOT_DATA`(정적 JSON, cosmetics/lot_data.json)로 통째로
  대체. Agent 입장에선 "이미 수집·저장까지 끝난 데이터를 조회"하는 것뿐이라 실시간 스트림을
  흉내 낼 필요가 없다. 각 측정값에 실제 제안서(슬라이드 10 요구사항분석)에 나온 진짜
  센서명을 태그로 붙여서, 어느 장비에서 나온 값인지 추적 가능하게 한다.
- Layer 3(AI 예측모델) → 두 갈래로 나눈다.
  ① 과거 배치의 판정·원인 — 이미 계산이 끝난 결과이므로 LOT_DATA에 미리 박아둔
     `판정`/`비고` 필드를 그대로 반환(추가 추론 불필요).
  ② 아직 안 일어난 배치에 대한 추천값 — `predict_condition()`이 SOP 문서의 기준값을
     그대로 읽어와 반환하는 **규칙 기반 스텁**이다. 실제 XGBoost/GRU 모델이 아니다 —
     이 사실을 함수 반환값(`mock: True`)과 화면(도구 호출 추적)에 명시해서, 데모를 보고
     실제 학습된 모델이 있다고 오해하지 않게 한다(합성 데이터는 합성이라고 명시하는 이
     프로젝트 전체의 원칙과 동일).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

from rag.build_index import EMBED_MODEL
from rag.query import retrieve

_ROOT = Path(__file__).resolve().parents[2]
_LOT_DATA_PATH = _ROOT / "cosmetics" / "lot_data.json"
_SOP_INDEX_PATH = _ROOT / "data" / "processed" / "cosmetics_index.faiss"
_SOP_META_PATH = _ROOT / "data" / "processed" / "cosmetics_chunks.jsonl"


@dataclass
class CosmeticsToolContext:
    """agent/tools.py의 ToolContext와 같은 지연 로드 패턴 — search_sop이 실제로
    호출될 때만 임베딩 모델·인덱스를 로드한다."""

    embed_model: SentenceTransformer | None = None
    index: object = None
    meta: list[dict] | None = None

    def _ensure_loaded(self) -> None:
        if self.embed_model is None:
            self.embed_model = SentenceTransformer(EMBED_MODEL)
            self.index = faiss.read_index(str(_SOP_INDEX_PATH))
            text = _SOP_META_PATH.read_text(encoding="utf-8")
            self.meta = [json.loads(line) for line in text.split("\n") if line.strip()]


def search_sop(ctx: CosmeticsToolContext, query: str, top_k: int = 3) -> list[dict]:
    """Layer 4 지식베이스 검색 도구 — SOP 문서에서 관련 조항을 찾는다. rag/query.py의
    retrieve()를 그대로 재사용(인덱스만 화장품 도메인 것으로 교체)."""
    ctx._ensure_loaded()
    hits = retrieve(query, ctx.embed_model, ctx.index, ctx.meta, top_k=top_k)
    return [{"source": h["source"], "text": h["text"]} for h, score in hits]

# 제안서 슬라이드 10(요구사항 분석 — 필요센서 및 통신방식)에 실제로 나온 센서 목록을
# 그대로 옮겨온 것 — 지어낸 장비가 아니다.
EQUIPMENT_MAP: dict[str, dict[str, dict[str, str]]] = {
    "천연물 추출": {
        "추출온도_C": {"장비": "Pt100 RTD 온도센서", "통신": "4-20mA→PLC AI 모듈"},
        "교반RPM": {"장비": "로터리 인코더", "통신": "RS-485(Modbus RTU)"},
        "손실률_pct": {"장비": "로드셀 + 중량 인디케이터", "통신": "RS-485(Modbus RTU)"},
        "HPLC지표성분_pct": {"장비": "HPLC 분석장비(연구소)", "통신": "수기입력 연동"},
    },
    "미생물 발효": {
        "발효온도_C": {"장비": "Pt100 RTD 온도센서", "통신": "4-20mA→PLC AI 모듈"},
        "pH_최종": {"장비": "pH 전극(유리전극)", "통신": "RS-485(Modbus RTU)"},
        "DO_평균_pct": {"장비": "DO 전극(격막형/광학식)", "통신": "4-20mA"},
        "교반RPM": {"장비": "로터리 인코더", "통신": "RS-485(Modbus RTU)"},
    },
    "초고압 나노분산": {
        "인가압력_bar": {"장비": "고압용 압력 트랜스미터", "통신": "설비 MES Data 활용"},
        "D50_nm": {"장비": "DLS 분석장비(입도크기)", "통신": "별도 계측기 연계"},
        "PDI": {"장비": "DLS 분석장비(입도크기)", "통신": "별도 계측기 연계"},
        "제타전위_mV": {"장비": "DLS 분석장비(입도크기)", "통신": "별도 계측기 연계"},
    },
}

# predict_condition()이 참조하는 SOP 기준값 — cosmetics/sop/*.txt에 적힌 값을 그대로
# 코드로 옮긴 것(원문 대조 가능). RAG 검색과 별개로, 구조화된 "추천값 조회"를 흉내 내려면
# 숫자 자체가 코드에도 있어야 한다.
_SOP_RANGES = {
    "천연물 추출": {
        "추출온도_C": (65, 70), "승온시간_분": (30, 40), "가열유지시간_분": (90, 120),
        "교반RPM": (40, 60), "출처": "SOP-EXT-01",
    },
    "미생물 발효": {
        "발효온도_C": (28, 32), "pH_초기": (5.5, 6.0), "pH_목표": (4.0, 4.5),
        "DO_최소_pct": (20, None), "교반RPM": (100, 150), "출처": "SOP-FMT-01",
    },
    "초고압 나노분산": {
        "인가압력_저점도_bar": (1000, 1200), "인가압력_고점도_bar": (1200, 1500),
        "Pass횟수": (2, 4), "냉각수온도_C": (15, 20), "출처": "SOP-NDP-01",
    },
}


def _load_lots() -> list[dict]:
    data = json.loads(_LOT_DATA_PATH.read_text(encoding="utf-8"))
    return data["lots"]


def query_lot(lot_id: str | None = None, process: str | None = None, 판정: str | None = None) -> dict:
    """Layer 2(RDB) 조회 도구 — REST 계약은 GET /api/cosmetics/lots와 동일하다(app.py
    라우트가 이 함수를 그대로 감싼다). lot_id를 주면 단건, 아니면 process/판정으로 필터링한
    목록을 반환한다. 각 필드에 Layer 1 장비 출처(EQUIPMENT_MAP)를 같이 실어서, 프론트가
    "이 값이 어느 센서에서 왔는지"를 보여줄 수 있게 한다.

    **2026-09-20 실측으로 발견**: LLM이 lot_id를 정확히 넘기면서도 process/판정을 추측해서
    같이 넘기는 경우가 실제로 있었다(예: "EXT-..." 로트인데 process="초고압 나노분산"으로
    잘못 추측) — 세 필터를 AND로 묶으면 lot_id는 맞았는데도 0건이 나온다. lot_id는 그
    자체로 고유키이므로, lot_id가 있으면 다른 필터는 무시한다."""
    lots = _load_lots()
    if lot_id:
        lots = [l for l in lots if l["lot_id"] == lot_id]
    else:
        if process:
            lots = [l for l in lots if l["process"] == process]
        if 판정:
            lots = [l for l in lots if l["판정"] == 판정]

    equipment_trace = []
    for lot in lots:
        eqmap = EQUIPMENT_MAP.get(lot["process"], {})
        for field, spec in eqmap.items():
            if field in lot:
                equipment_trace.append({
                    "lot_id": lot["lot_id"], "필드": field, "값": lot[field],
                    "장비": spec["장비"], "통신": spec["통신"],
                })

    return {"lots": lots, "count": len(lots), "equipment_trace": equipment_trace}


def generate_coa_draft(lot_id: str, live_ticks: list[dict] | None = None) -> dict:
    """제안서가 "AI 에이전트 핵심가치"로 명시한 COA(시험성적서) 초안 자동생성(2026-09-23
    추가). LOT_DATA에 이미 있는 측정값·판정을 문서 양식으로 조립만 할 뿐, 합격/불합격을
    새로 판단하지 않는다 — 판정은 이 프로젝트 전체 원칙대로 이미 정해진 데이터(규칙)이고
    Agent는 그걸 문서로 정리하는 역할만 한다. 정식 발급 문서가 아니라 초안이라는 점을
    반환값에 명시해서(mock과 같은 이유) 실제 서명 없이 유통되지 않게 한다.

    **2026-09-24 확장**: 시나리오 데모(자동 시뮬레이션)가 만드는 lot_id는 LOT_DATA에 없는
    라이브 값(예: FMT-20260920-01)이라, lot_id를 못 찾으면 live_ticks(query_recent_ticks와
    같은 형태)의 최신 시점 값으로 COA를 조립하는 대체 경로를 탄다. 이때도 판정은 프론트가
    "이상 없이 종료됐다"고 주장하는 걸 그대로 믿지 않고, 이미 SOP 판정에 쓰고 있는
    `_SOP_RANGES`의 DO 최소 기준을 여기서도 다시 적용해 직접 판정한다 — 자유 질의로
    호출됐을 때도(예: 이상 상태에서 COA를 물어봐도) 같은 규칙으로 정직하게 답하기 위함."""
    lots = _load_lots()
    lot = next((l for l in lots if l["lot_id"] == lot_id), None)
    if lot is not None:
        eqmap = EQUIPMENT_MAP.get(lot["process"], {})
        측정항목 = [
            {"항목": field, "측정값": lot[field], "측정장비": eqmap.get(field, {}).get("장비", "-")}
            for field in lot
            if field in eqmap
        ]
        return {
            "lot_id": lot["lot_id"],
            "공정": lot["process"],
            "원물_또는_제형": lot.get("원물", "-"),
            "측정항목": 측정항목,
            "판정": lot["판정"],
            "비고": lot.get("비고", ""),
            "문서상태": "초안 — 정식 발급 아님",
            "안내": "QC 담당자 검토·서명 후에만 정식 COA로 발급할 수 있습니다.",
        }

    if not live_ticks:
        return {"error": f"Lot ID를 찾을 수 없습니다: {lot_id}"}

    latest = live_ticks[0]
    process = "미생물 발효"
    eqmap = EQUIPMENT_MAP.get(process, {})
    측정항목 = [
        {"항목": field, "측정값": latest[field], "측정장비": eqmap.get(field, {}).get("장비", "-")}
        for field in eqmap
        if field in latest
    ]
    do_min, _ = _SOP_RANGES[process]["DO_최소_pct"]
    do_now = latest.get("DO_평균_pct")
    if do_now is not None and do_now < do_min:
        판정, 비고 = "재작업", f"발효 종료 시점 DO {do_now}%가 SOP 최소 기준({do_min}%) 미만 — 재작업 대상."
    else:
        판정, 비고 = "합격", "실시간 시나리오 최종 시점 값 기준 자동 조립(별도 QC 재시험 전제)."

    return {
        "lot_id": lot_id,
        "공정": process,
        "원물_또는_제형": "-",
        "측정항목": 측정항목,
        "판정": 판정,
        "비고": 비고,
        "문서상태": "초안 — 정식 발급 아님",
        "안내": "QC 담당자 검토·서명 후에만 정식 COA로 발급할 수 있습니다.",
    }


def predict_condition(process: str, **hint) -> dict:
    """Layer 3(AI 분석층) mock 호출 도구 — REST 계약은 POST /api/cosmetics/predict와
    동일하다. **실제 학습된 모델이 아니다** — SOP 기준값(_SOP_RANGES)의 중간값을 그대로
    반환하는 규칙 기반 스텁이다. 제안서의 4종 모델(XGBoost/GRU/물리모델+RF/Isolation
    Forest)이 실제로 자리할 위치를 인터페이스만 미리 잡아둔 것."""
    ranges = _SOP_RANGES.get(process)
    if not ranges:
        return {"error": f"알 수 없는 공정: {process}", "mock": True}

    recommended = {}
    for key, val in ranges.items():
        if key == "출처":
            continue
        if isinstance(val, tuple):
            lo, hi = val
            if hi is None:
                recommended[key] = f"{lo} 이상"
            else:
                recommended[key] = round((lo + hi) / 2, 1)

    return {
        "process": process,
        "recommended_condition": recommended,
        "근거_SOP": ranges.get("출처"),
        "mock": True,
        "mock_설명": "실제 XGBoost/GRU 등 학습된 모델이 아니라, SOP 기준값 범위의 중간값을 반환하는 규칙 기반 스텁입니다.",
    }
