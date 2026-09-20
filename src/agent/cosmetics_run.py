"""화장품 제조 AX PoC — Layer 4(AI 에이전트) 오케스트레이션.

agent/run.py(산업안전)의 tool-calling 루프와 같은 패턴 — `<tool_call>` 태그를
llama-cpp-python이 구조화 필드로 안 주는 문제도 동일하게 정규식으로 우회한다
(의사결정_로그 30번에서 이미 확인된 동작).

**이 파일이 검증하려는 것**: 자연어 질문 → 도구 호출(SOP 검색 / Lot 조회 / AI 추론mock)
→ 근거 기반 답변 종합, 그리고 설비제어·품질판정·공식문서 확정처럼 사람 승인이 필요한
사안은 LLM이 스스로 판단하지 않고 SOP-QC-01 정책을 인용해 승인 필요라고 답하는지 — 이
네 가지만 확인하면 된다(사용자와 합의한 PoC 범위).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from llama_cpp import Llama

from agent import cosmetics_tools as tools
from agent.cosmetics_tools import CosmeticsToolContext

_ROOT = Path(__file__).resolve().parents[2]

_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_sop",
            "description": "화장품 제조 SOP·작업표준서·정책 문서에서 관련 조항을 검색한다(운전조건 기준, 이상 시 조치, 승인 정책 등).",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "검색어"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_lot",
            "description": "생산 Lot 이력 데이터를 조회한다(Lot ID, 공정, 판정 결과, 측정값). 과거 배치의 판정·원인을 물어볼 때 사용.",
            "parameters": {
                "type": "object",
                "properties": {
                    "lot_id": {"type": "string", "description": "특정 Lot 번호(예: EXT-20260903-02)"},
                    "process": {"type": "string", "description": "공정명(천연물 추출 / 미생물 발효 / 초고압 나노분산)"},
                    "판정": {"type": "string", "description": "합격/불합격/재작업 등으로 필터링"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict_condition",
            "description": "아직 진행하지 않은 새 배치에 대한 권장 운전조건을 추천받는다(과거 이력이 아니라 사전 추천이 필요할 때 사용).",
            "parameters": {
                "type": "object",
                "properties": {
                    "process": {"type": "string", "description": "공정명(천연물 추출 / 미생물 발효 / 초고압 나노분산)"},
                },
                "required": ["process"],
            },
        },
    },
]

_SYSTEM_PROMPT = (
    "당신은 화장품 제조 현장의 AI 업무지원 에이전트입니다. 작업자의 질문에 답하기 위해 "
    "필요하면 search_sop(SOP 문서 검색), query_lot(생산 Lot 이력 조회), predict_condition"
    "(신규 배치 추천 조건, 과거 이력이 아닌 사전 추천일 때만) 도구를 사용하세요.\n"
    "도구가 필요하면 판단 과정을 문장으로 먼저 설명하지 말고 곧바로 호출하세요 — "
    "설명은 도구 결과를 받은 뒤 최종 답변에서만 하세요.\n"
    "반드시 지킬 규칙(이 순서로 먼저 판단하세요):\n"
    "1) 먼저 확인: 이 요청이 설비 제어(운전조건 실제 변경), 배치 합격/불합격 등 품질의 "
    "'최종 판정' 확정, 또는 COA 등 공식 문서 '확정'을 요청하는 것입니까? 그렇다면 당신은 "
    "직접 판정·확정·실행하지 말고, 곧바로 search_sop으로 'SOP-QC-01 승인 정책'을 검색한 "
    "뒤 그 내용을 인용하면서 '이 사안은 담당자 승인이 필요합니다'라고 답하세요. 이 규칙은 "
    "다른 모든 규칙보다 우선합니다.\n"
    "2) 위 경우가 아니라면, 도구로 확인한 근거에 없는 내용은 답하지 말고 '자료에서 근거를 "
    "찾지 못했습니다'라고 답하세요.\n"
    "3) predict_condition의 결과는 학습된 AI 모델이 아니라 SOP 기준값 기반 참고용 추천이라는 "
    "점을 답변에 명시하세요."
)

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
# 도구 호출 전에 장황한 설명을 늘어놓다가 max_tokens에 걸려 닫는 태그(</tool_call>) 전에
# 잘리는 경우가 실측에서 나왔다(2026-09-20) — 프롬프트로 "바로 호출하라"고 유도했지만
# 완전히 막힌다는 보장은 없어서, 닫히지 않은 tool_call도 마지막 수단으로 복구를 시도한다.
_TOOL_CALL_UNCLOSED_RE = re.compile(r"<tool_call>\s*(\{.*\})\s*$", re.DOTALL)


def _extract_tool_calls(content: str) -> list[dict]:
    closed = _TOOL_CALL_RE.findall(content or "")
    if closed:
        return [json.loads(m) for m in closed]
    m = _TOOL_CALL_UNCLOSED_RE.search(content or "")
    if not m:
        return []
    try:
        return [json.loads(m.group(1))]
    except json.JSONDecodeError:
        return []  # 중간에 잘린 JSON까지는 복구 안 함 — 억지로 짜맞추면 잘못된 도구 인자를 실행할 위험


@dataclass
class ToolStep:
    """Layer 파이프라인 시각화용 — 도구 호출 하나가 Layer 몇에 해당하는지, 어떤
    장비·데이터를 거쳤는지 프론트에 그대로 넘긴다."""

    tool: str
    layer: int
    layer_label: str
    args: dict
    result_summary: str
    detail: dict = field(default_factory=dict)


_LAYER_LABEL = {1: "Layer 1 · 장비", 2: "Layer 2 · 데이터(RDB)", 3: "Layer 3 · AI 분석", 4: "Layer 4 · 지식베이스(SOP)"}


def _run_query_lot(args: dict) -> tuple[dict, ToolStep]:
    result = tools.query_lot(lot_id=args.get("lot_id"), process=args.get("process"), 판정=args.get("판정"))
    lots = result["lots"]
    summary = f"{result['count']}건 조회됨" + (f" (첫 건: {lots[0]['lot_id']} {lots[0]['판정']})" if lots else "")
    step = ToolStep(
        tool="query_lot", layer=2, layer_label=_LAYER_LABEL[2], args=args, result_summary=summary,
        detail={"lots": lots, "equipment_trace": result["equipment_trace"]},
    )
    return result, step


def _run_predict_condition(args: dict) -> tuple[dict, ToolStep]:
    result = tools.predict_condition(process=args.get("process", ""))
    summary = f"{args.get('process', '')} 추천조건 생성(mock, 근거: {result.get('근거_SOP', '-')})"
    step = ToolStep(tool="predict_condition", layer=3, layer_label=_LAYER_LABEL[3], args=args, result_summary=summary, detail=result)
    return result, step


def _run_search_sop(ctx: CosmeticsToolContext, args: dict) -> tuple[list[dict], ToolStep]:
    hits = tools.search_sop(ctx, args.get("query", ""))
    summary = f"{len(hits)}개 조항 검색됨" + (f" (출처: {', '.join(dict.fromkeys(h['source'] for h in hits))})" if hits else "")
    step = ToolStep(tool="search_sop", layer=4, layer_label=_LAYER_LABEL[4], args=args, result_summary=summary, detail={"hits": hits})
    return hits, step


def run_cosmetics_agent(question: str, ctx: CosmeticsToolContext, llm: Llama, max_tool_turns: int = 4) -> dict:
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    steps: list[ToolStep] = []
    narrative = ""

    for _ in range(max_tool_turns):
        result = llm.create_chat_completion(messages=messages, tools=_TOOLS_SCHEMA, temperature=0.0, max_tokens=700)
        msg = result["choices"][0]["message"]
        content = msg.get("content") or ""
        tool_calls = msg.get("tool_calls") or None
        manual_calls = None if tool_calls else _extract_tool_calls(content)

        if not tool_calls and not manual_calls:
            narrative = content
            break

        messages.append({"role": "assistant", "content": content})
        calls = tool_calls or [{"function": {"name": c.get("name"), "arguments": json.dumps(c.get("arguments", {}))}} for c in manual_calls]

        for call in calls:
            fn_name = call["function"]["name"]
            try:
                args = json.loads(call["function"]["arguments"])
            except json.JSONDecodeError:
                args = {}

            if fn_name == "query_lot":
                tool_result, step = _run_query_lot(args)
            elif fn_name == "predict_condition":
                tool_result, step = _run_predict_condition(args)
            elif fn_name == "search_sop":
                tool_result, step = _run_search_sop(ctx, args)
            else:
                tool_result, step = {"error": f"알 수 없는 도구: {fn_name}"}, ToolStep(fn_name, 0, "?", args, "오류")

            steps.append(step)
            messages.append({"role": "tool", "content": json.dumps(tool_result, ensure_ascii=False, default=str)})
    else:
        narrative = narrative or "(도구 호출 반복 한도 초과 — 최종 답변을 얻지 못함)"

    return {
        "question": question,
        "narrative": narrative,
        "steps": [
            {"tool": s.tool, "layer": s.layer, "layer_label": s.layer_label, "args": s.args, "result_summary": s.result_summary, "detail": s.detail}
            for s in steps
        ],
    }
