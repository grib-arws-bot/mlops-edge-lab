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
            "name": "query_recent_ticks",
            "description": "화면에 보이는 미생물 발효기의 최근 실시간 시계열(온도·pH·DO·RPM)과 이상 감지 메모를 조회한다. '시점 N부터 M까지 무슨 문제가 있었냐'처럼 방금 진행된 시나리오 자체에 대한 질문일 때 사용 — 과거에 이미 종료·저장된 배치 이력(query_lot)과는 다르다.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_coa_draft",
            "description": "특정 Lot의 QC 측정값·판정을 COA(시험성적서) 양식으로 정리한 초안을 생성한다. 저장된 과거 Lot이면 이미 저장된 값을 그대로 조립하고, 지금 화면에 보이는 실시간 시나리오의 lot_id면 최근 실시간 이력(query_recent_ticks와 같은 데이터)을 근거로 SOP 기준에 따라 판정까지 함께 조립한다 — 정식 발급이 아닌 초안이며 QC 담당자 검토·서명이 필요하다.",
            "parameters": {
                "type": "object",
                "properties": {"lot_id": {"type": "string", "description": "COA를 생성할 Lot 번호(예: EXT-20260901-01)"}},
                "required": ["lot_id"],
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

_NO_EVIDENCE_MSG = "자료에서 근거를 찾지 못했습니다."

_SYSTEM_PROMPT = (
    "당신은 화장품 제조 현장의 AI 업무지원 에이전트입니다. 작업자의 질문에 답하기 위해 "
    "필요하면 search_sop(SOP 문서 검색), query_lot(과거에 종료·저장된 생산 Lot 이력 조회), "
    "query_recent_ticks(지금 화면에 보이는 시나리오의 최근 실시간 시점 이력 — '시점 N', "
    "'방금', '지금까지' 같은 질문에 사용), generate_coa_draft(특정 Lot의 COA 시험성적서 "
    "초안 생성 — 이미 저장된 값을 문서로 조립만 함, 새로 판정하지 않음), "
    "predict_condition(신규 배치 추천 조건, 과거 이력이 "
    "아닌 사전 추천일 때만) 도구를 사용하세요.\n"
    "query_recent_ticks가 돌려준 시점 범위 밖을 물어보면(예: 표시된 것보다 훨씬 이전 시점), "
    "모른다고 하지 말고 '표시된 범위(가장 오래된 시점~최신 시점) 밖이라 알 수 없습니다'라고 "
    "정확히 답하세요.\n"
    "도구가 필요하면 판단 과정을 문장으로 먼저 설명하지 말고 곧바로 호출하세요 — "
    "설명은 도구 결과를 받은 뒤 최종 답변에서만 하세요.\n"
    "반드시 지킬 규칙(이 순서로 먼저 판단하세요):\n"
    "0) 인사말처럼 SOP·Lot·공정과 무관한 질문이 아니라면, 아는 내용이라고 생각되더라도 "
    "먼저 도구를 최소 1회 호출해 근거를 확인한 뒤에만 답하세요. 도구를 하나도 "
    "호출하지 않고 기술적인 내용을 답하는 것은 금지됩니다.\n"
    "1) 먼저 확인: 이 요청이 설비 제어(운전조건 실제 변경), 배치 합격/불합격 등 품질의 "
    "'최종 판정' 확정, 또는 COA를 정식 발급·확정하는 것입니까?(COA '초안'을 만들어달라는 "
    "요청은 여기 해당하지 않습니다 — 그건 generate_coa_draft를 바로 호출하세요.) 그렇다면 "
    "당신은 직접 판정·확정·실행하지 말고, 곧바로 search_sop으로 'SOP-QC-01 승인 정책'을 "
    "검색한 뒤 그 내용을 인용하면서 '이 사안은 담당자 승인이 필요합니다'라고 답하세요. 이 "
    "규칙은 다른 모든 규칙보다 우선합니다.\n"
    "2) 위 경우가 아니라면, 도구로 확인한 근거에 없는 내용은 답하지 말고 '자료에서 근거를 "
    "찾지 못했습니다'라고 답하세요.\n"
    "3) predict_condition의 결과는 학습된 AI 모델이 아니라 SOP 기준값 기반 참고용 추천이라는 "
    "점을 답변에 명시하세요.\n"
    "4) generate_coa_draft의 결과를 보여줄 때는 반드시 '초안이며 QC 담당자 검토·서명이 "
    "필요하다'는 점을 답변에 명시하세요."
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


_LAYER_LABEL = {
    1: "Layer 1 · 장비(실시간 시계열)", 2: "Layer 2 · 데이터(RDB)", 3: "Layer 3 · AI 분석", 4: "Layer 4 · 지식베이스(SOP)",
}
# Layer 1(장비) 뱃지는 지금까지 정의만 돼 있고 실제로 쓰는 도구가 없었다(query_lot은
# 이미 저장된 Layer 2 RDB, predict_condition은 Layer 3) — query_recent_ticks가 처음으로
# "아직 DB에 쌓이기 전, 화면에 보이는 실시간 장비 값" 자리를 채운다.


def _run_query_lot(args: dict) -> tuple[dict, ToolStep]:
    result = tools.query_lot(lot_id=args.get("lot_id"), process=args.get("process"), 판정=args.get("판정"))
    lots = result["lots"]
    summary = f"{result['count']}건 조회됨" + (f" (첫 건: {lots[0]['lot_id']} {lots[0]['판정']})" if lots else "")
    step = ToolStep(
        tool="query_lot", layer=2, layer_label=_LAYER_LABEL[2], args=args, result_summary=summary,
        detail={"lots": lots, "equipment_trace": result["equipment_trace"]},
    )
    return result, step


def _run_query_recent_ticks(live_ticks: list[dict]) -> tuple[dict, ToolStep]:
    """query_lot(과거에 종료·저장된 배치)과 달리, 이건 브라우저 화면에 그려지고 있는
    "지금 이 시나리오"의 최근 시점 값이다 — 서버는 이 값을 따로 갖고 있지 않고,
    프론트가 매 질문마다 화면에 보이는 이력을 같이 보내준다(2026-09-23, 사용자 지적:
    "Layer 4에 직접 질문하기"가 실시간 데이터에 대한 질문엔 답을 못 하고 있었음).
    화면에 그려진 범위(logRows, 최근 최대 50개) 밖의 시점은 이 도구로도 알 수 없다
    — 그 경우는 정직하게 '표시된 범위 밖'이라고 답해야 한다(시스템 프롬프트에 명시)."""
    if not live_ticks:
        result = {"ticks": [], "note": "화면에 표시된 실시간 이력이 없습니다(아직 시나리오를 시작하지 않았을 수 있음)."}
        summary = "실시간 이력 없음"
    else:
        result = {"ticks": live_ticks, "count": len(live_ticks)}
        summary = f"최근 {len(live_ticks)}개 시점 조회됨(최신: {live_ticks[0].get('t', '?')})"
    step = ToolStep(tool="query_recent_ticks", layer=1, layer_label=_LAYER_LABEL[1], args={}, result_summary=summary, detail=result)
    return result, step


def _run_generate_coa_draft(args: dict, live_ticks: list[dict] | None) -> tuple[dict, ToolStep]:
    result = tools.generate_coa_draft(lot_id=args.get("lot_id", ""), live_ticks=live_ticks)
    if "error" in result:
        summary = result["error"]
    else:
        summary = f"{result['lot_id']} COA 초안 생성됨 (판정: {result['판정']}, {result['문서상태']})"
    step = ToolStep(tool="generate_coa_draft", layer=2, layer_label=_LAYER_LABEL[2], args=args, result_summary=summary, detail=result)
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


def run_cosmetics_agent(
    question: str, ctx: CosmeticsToolContext, llm: Llama, max_tool_turns: int = 4,
    live_ticks: list[dict] | None = None,
) -> dict:
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
            elif fn_name == "query_recent_ticks":
                tool_result, step = _run_query_recent_ticks(live_ticks)
            elif fn_name == "generate_coa_draft":
                tool_result, step = _run_generate_coa_draft(args, live_ticks)
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
