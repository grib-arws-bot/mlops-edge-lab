"""Agent가 호출하는 도구들. `search_guidelines`만 LLM이 직접 호출 여부·검색어를 판단하고,
`notify`/`escalate`/`draft_incident_report`는 규칙(`rules.judge`)의 위험도에 따라 코드가
직접 호출한다 — "누구를 부를지는 LLM이 유연하게, 위험도는 규칙이 고정적으로" 원칙.

지금은 실제 메신저·이메일 연동이 없어서 `notify`/`escalate`는 시뮬레이션(로그만 남김)이다.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

from rag.build_index import EMBED_MODEL
from rag.query import load_meta, retrieve

_ROOT = Path(__file__).resolve().parents[2]
_INDEX_PATH = _ROOT / "data" / "processed" / "rag_index.faiss"


@dataclass
class ToolContext:
    """RAG 인덱스·임베딩 모델을 지연 로드한다 — LLM이 실제로 search_guidelines를 호출할
    때만 로드하고, 로드 후에는 재사용한다.

    **2026-09-19 변경**: 원래는 load()가 무조건 즉시 로드했는데, 실측해보니 이게
    7초 넘게 걸렸다(e5-small 임베딩 모델 + FAISS 인덱스). 엣지 에뮬레이션(/simulate,
    agent/run_cli.py)은 요청마다 새 프로세스를 띄우는데, LLM이 검색을 아예 안 부르는
    경우(실제로 흔함 — 판정·장비조치는 이미 다 정해져 있어서 LLM이 굳이 안 찾아봐도
    될 때가 많음)에도 이 로드 비용을 매번 냈다. "복합 센서 시뮬레이션이 GPU를 써도
    느리다"는 사용자 지적을 파보다가 실제 병목이 여기였다는 걸 확인함 — LLM 추론
    자체는 1~5초인데 이 초기화가 7초를 더 얹고 있었음."""

    embed_model: SentenceTransformer | None = None
    index: object = None
    meta: list[dict] | None = None
    notify_log: list[dict] = field(default_factory=list)

    @classmethod
    def load(cls) -> "ToolContext":
        return cls()

    def _ensure_rag_loaded(self) -> None:
        if self.embed_model is None:
            self.embed_model = SentenceTransformer(EMBED_MODEL)
            self.index = faiss.read_index(str(_INDEX_PATH))
            self.meta = load_meta()


def search_guidelines(ctx: ToolContext, query: str, top_k: int = 3) -> list[dict]:
    """로드맵 5번 RAG 검색을 도구로 노출. LLM이 검색 여부·검색어를 스스로 판단해 호출한다."""
    ctx._ensure_rag_loaded()
    hits = retrieve(query, ctx.embed_model, ctx.index, ctx.meta, top_k=top_k)
    return [{"source": h["source"], "text": h["text"][:300]} for h, score in hits]


def notify(ctx: ToolContext, channel: str, message: str) -> dict:
    """1차 알림 발송(시뮬레이션)."""
    record = {
        "type": "notify",
        "channel": channel,
        "message": message,
        "sent_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    ctx.notify_log.append(record)
    return {"status": "sent", "channel": channel}


def actuate_equipment(ctx: ToolContext, action) -> dict:
    """저위험 장비 제어 — 규칙이 이미 결정한 것을 그대로 실행(시뮬레이션). LLM 관여 없음."""
    record = {
        "type": "actuate",
        "equipment": action.name,
        "description": action.description,
        "status": "실행됨",
        "at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    ctx.notify_log.append(record)
    return record


def request_equipment_approval(ctx: ToolContext, action) -> dict:
    """고위험 장비 제어 — 자동 실행하지 않고 사람 승인을 요청만 한다(시뮬레이션). LLM 관여 없음."""
    record = {
        "type": "actuate",
        "equipment": action.name,
        "description": action.description,
        "status": "승인 대기",
        "at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    ctx.notify_log.append(record)
    return record


def escalate(ctx: ToolContext, reason: str) -> dict:
    """에스컬레이션(시뮬레이션). 위험도가 '위험'일 때 코드가 직접 호출한다(LLM 판단 아님)."""
    record = {
        "type": "escalate",
        "reason": reason,
        "sent_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    ctx.notify_log.append(record)
    return {"status": "escalated"}


def draft_incident_report(ctx: ToolContext, event: dict, judgement, narrative: str, guideline_sources: list[str]) -> dict:
    """사고 리포트 초안. 이미 만든 알림 문장(narrative)을 재사용 — 추가 LLM 호출 없이 조립만 한다."""
    return {
        "type": "incident_report_draft",
        "location": event["location"],
        "category": event["category"],
        "substance": event["substance"],
        "value": event["value"],
        "threshold": event["threshold"],
        "severity": judgement.severity.value,
        "ratio": round(judgement.ratio, 2),
        "narrative": narrative,
        "guideline_sources": guideline_sources,
        "drafted_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
