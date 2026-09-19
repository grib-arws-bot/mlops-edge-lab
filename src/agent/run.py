"""Agent 오케스트레이션 — 판정(규칙) → (필요시) LLM 도구 호출 루프 → 알림/에스컬레이션/리포트.

역할 분담(모듈 docstring들 참고):
- 초과 여부·위험도: rules.judge()  (규칙, 결정론)
- 검색 여부·검색어: LLM이 tool_call로 스스로 판단
- 알림·에스컬레이션·리포트 실행 여부: 이 파일의 코드가 위험도로 직접 결정 (LLM에게 안 맡김)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from llama_cpp import Llama

from agent import rules, tools
from agent.tools import ToolContext


@dataclass
class Decision:
    """모든 결정에 "누가 결정했는가"를 명시적으로 남긴다 — 사용자 지적: LLM 자율판단이
    들어간 케이스와 아닌 케이스를 분기·추적할 수 있어야 한다."""

    authority: str  # "rule" | "llm"
    action: str
    detail: dict

_ROOT = Path(__file__).resolve().parents[2]
_GGUF_PATH = _ROOT / "experiments" / "toy-sensor-lora" / "model-Q4_K_M.gguf"

_TOOLS_SCHEMA = [{
    "type": "function",
    "function": {
        "name": "search_guidelines",
        "description": "산업안전 가이드라인 문서에서 관련 조치 방법을 검색한다.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "검색어"}},
            "required": ["query"],
        },
    },
}]

_SYSTEM_PROMPT = (
    "당신은 산업 현장 안전관리 보조 에이전트입니다. 센서 이상 이벤트가 발생하면, "
    "필요하다면 search_guidelines 도구로 관련 안전수칙을 찾아본 뒤, "
    "현장 담당자에게 보낼 간결한 한국어 알림 문장을 작성하세요. "
    "판정(초과 여부·위험도)과 장비 조치 실행 여부는 이미 시스템이 정했으니 당신은 다시 "
    "판정하거나 새로운 장비 조치를 지시하지 말고, 전달받은 사실을 안내문에 반영해서 "
    "설명만 하세요."
)


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def _extract_tool_calls(content: str) -> list[dict]:
    """llama-cpp-python이 Qwen3의 <tool_call> 태그를 구조화된 tool_calls 필드로 파싱해
    주지 않아서(실제로 확인된 동작 — docs/의사결정_로그.md 30번), 모델이 낸 원문 텍스트에서
    직접 정규식으로 뽑아낸다."""
    return [json.loads(m) for m in _TOOL_CALL_RE.findall(content or "")]


def _event_prompt(event: dict, judgement: rules.Judgement) -> str:
    # 산소농도처럼 "낮을수록 위험"한 물질은 LLM이 방향을 헷갈려 "초과했다"고 잘못 서술할 수
    # 있어서(수치가 내려갔는데 "초과"라고 하면 반대로 들림), 방향을 프롬프트에 명시해준다.
    direction = "낮을수록 위험한 지표" if event.get("lower_is_worse") else "높을수록 위험한 지표"
    return (
        f"{event['location']} {event['category']}센서({event['substance']}, {direction}), "
        f"측정값 {event['value']}{event['unit']}, 임계값 {event['threshold']}{event['unit']}, "
        f"위험도 판정: {judgement.severity.value}"
    )


def preview(event: dict) -> dict:
    """LLM·도구 호출 없이 규칙만으로 즉시 판정한다 — 웹 UI가 "장비 상태는 즉시 반응"을
    보여줄 때 쓴다. 장비 상태는 위험도에서 결정론적으로 정해지므로(risk tier만 보면 됨)
    실제로 로그에 남기는 tools.actuate_*를 부르지 않아도 결과가 똑같다 — 그래서 ctx가
    필요 없는 순수 함수다(부작용 없음, 두 번 불러도 안전)."""
    judgement = rules.judge(event["value"], event["threshold"], event.get("lower_is_worse", False))
    equipment = []
    if judgement.exceeded:
        for action in rules.required_equipment_actions(event["category"], judgement.severity):
            status = "실행됨" if action.risk == rules.Risk.LOW else "승인 대기"
            equipment.append({"equipment": action.name, "status": status})
    return {
        "judgement": {
            "severity": judgement.severity.value,
            "ratio": round(judgement.ratio, 2),
            "exceeded": judgement.exceeded,
        },
        "equipment_status": equipment,
    }


def _run_llm_tool_loop(
    messages: list[dict], ctx: ToolContext, llm: Llama, fallback_query: str, max_tool_turns: int,
) -> tuple[str, list[dict], list[str], list[Decision]]:
    """LLM에게 도구 호출 기회를 주고 최종 텍스트가 나올 때까지 반복한다. 단일 이벤트
    (run_agent)와 복합 이벤트(run_agent_composite)가 완전히 같은 루프를 쓴다 — 도구 호출
    처리 로직이 둘 다 동일해서 따로 둘 이유가 없다."""
    tool_trace: list[dict] = []
    guideline_sources: list[str] = []
    decisions: list[Decision] = []
    narrative = ""

    for _ in range(max_tool_turns):
        result = llm.create_chat_completion(
            messages=messages, tools=_TOOLS_SCHEMA, temperature=0.0, max_tokens=400
        )
        msg = result["choices"][0]["message"]
        content = msg.get("content") or ""
        # 구조화된 tool_calls 필드를 먼저 보고, 없으면 텍스트에서 직접 파싱한다(위 설명 참고)
        tool_calls = msg.get("tool_calls") or None
        manual_calls = None if tool_calls else _extract_tool_calls(content)

        if not tool_calls and not manual_calls:
            narrative = content
            break

        messages.append({"role": "assistant", "content": content})

        calls = tool_calls or [{"function": {"arguments": json.dumps(c["arguments"])}} for c in manual_calls]
        for call in calls:
            args = json.loads(call["function"]["arguments"])
            query = args.get("query", fallback_query)
            hits = tools.search_guidelines(ctx, query)
            guideline_sources.extend(h["source"] for h in hits)
            trace_entry = {"tool": "search_guidelines", "args": args, "result_count": len(hits)}
            tool_trace.append(trace_entry)
            decisions.append(Decision("llm", "tool_call", trace_entry))
            messages.append({"role": "tool", "content": json.dumps(hits, ensure_ascii=False)})
    else:
        narrative = narrative or "(도구 호출 반복 한도 초과 — 최종 답변을 얻지 못함)"

    decisions.append(Decision("llm", "compose_narrative", {"narrative": narrative}))
    return narrative, tool_trace, guideline_sources, decisions


def run_agent(event: dict, ctx: ToolContext, llm: Llama, max_tool_turns: int = 3) -> dict:
    decisions: list[Decision] = []

    judgement = rules.judge(event["value"], event["threshold"], event.get("lower_is_worse", False))
    decisions.append(Decision("rule", "judge", {
        "severity": judgement.severity.value, "ratio": round(judgement.ratio, 2),
    }))

    if not judgement.exceeded:
        if event.get("lower_is_worse"):
            narrative = (
                f"{event['location']}의 {event['substance']} 농도는 {event['value']}{event['unit']}로 "
                f"안전 기준({event['threshold']}{event['unit']} 이상)을 충족합니다. 정상 범위입니다."
            )
        else:
            narrative = (
                f"{event['location']}의 {event['substance']} 농도는 {event['value']}{event['unit']}로 "
                f"임계값({event['threshold']}{event['unit']}) 이내입니다. 정상 범위입니다."
            )
        return {
            "judgement": judgement, "narrative": narrative, "tool_trace": [],
            "notify_log": [], "report": None, "decisions": decisions,
        }

    # 장비 제어는 규칙이 즉시 결정·실행한다 — LLM이 서술을 시작하기 전에 먼저 처리한다.
    # (실제 현장이라면 문구가 완성되길 기다렸다가 환기를 켜는 건 말이 안 됨)
    equipment_actions = rules.required_equipment_actions(event["category"], judgement.severity)
    equipment_status: list[dict] = []
    for action in equipment_actions:
        if action.risk == rules.Risk.LOW:
            record = tools.actuate_equipment(ctx, action)
        else:
            record = tools.request_equipment_approval(ctx, action)
        equipment_status.append(record)
        decisions.append(Decision("rule", "equipment", record))

    equipment_summary = (
        "; ".join(f"{r['equipment']}({r['status']})" for r in equipment_status)
        if equipment_status else "해당 없음"
    )

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"{_event_prompt(event, judgement)}\n"
            f"이미 자동으로 처리된 장비 조치(사실, 그대로 인용할 것): {equipment_summary}"
        )},
    ]

    narrative, tool_trace, guideline_sources, tool_decisions = _run_llm_tool_loop(
        messages, ctx, llm, event["substance"], max_tool_turns,
    )
    decisions.extend(tool_decisions)

    # 알림/에스컬레이션/리포트 실행 여부는 코드가 위험도로 직접 결정한다 (LLM에게 안 맡김)
    tools.notify(ctx, channel="현장관리자-알림방", message=narrative)
    decisions.append(Decision("rule", "notify", {"channel": "현장관리자-알림방"}))
    report = None
    if judgement.severity == rules.Severity.DANGER:
        tools.escalate(ctx, reason=f"{event['substance']} 위험 수준 감지 (임계값의 {judgement.ratio:.1f}배)")
        decisions.append(Decision("rule", "escalate", {"reason": event["substance"]}))
        report = tools.draft_incident_report(
            ctx, event, judgement, narrative, list(dict.fromkeys(guideline_sources))
        )
        decisions.append(Decision("rule", "draft_incident_report", {}))

    return {
        "judgement": judgement,
        "narrative": narrative,
        "tool_trace": tool_trace,
        "equipment_status": equipment_status,
        "notify_log": list(ctx.notify_log),
        "report": report,
        "decisions": decisions,
    }


_COMPOSITE_SYSTEM_PROMPT = (
    "당신은 산업 현장 안전관리 보조 에이전트입니다. 한 공간에 설치된 여러 센서의 측정값을 "
    "한꺼번에 받습니다. 센서마다 따로 코멘트하지 말고, 전체 상황을 종합해서 이 공간이 지금 "
    "얼마나 안전한지 하나의 의견으로 정리하세요. 가장 위험도가 높은 항목을 중심으로 설명하되 "
    "다른 항목도 함께 언급하세요. 판정(초과 여부·위험도)과 장비 조치 실행 여부는 이미 시스템이 "
    "정했으니 당신은 다시 판정하거나 새로운 장비 조치를 지시하지 말고, 전달받은 사실을 "
    "종합 안내문에 반영해서 설명만 하세요."
)


def _composite_event_lines(judged: list[tuple[dict, rules.Judgement]]) -> str:
    lines = []
    for event, judgement in judged:
        direction = "낮을수록 위험한 지표" if event.get("lower_is_worse") else "높을수록 위험한 지표"
        lines.append(
            f"- {event['category']}센서({event['substance']}, {direction}): "
            f"측정값 {event['value']}{event['unit']}, 임계값 {event['threshold']}{event['unit']}, "
            f"판정: {judgement.severity.value}"
        )
    return "\n".join(lines)


def run_agent_composite(events: list[dict], ctx: ToolContext, llm: Llama, max_tool_turns: int = 3) -> dict:
    """여러 센서가 같은 공간에 있다고 가정하고, LLM이 센서마다 따로 코멘트하는 대신 전체를
    종합한 의견 하나를 낸다(사용자 요청, 2026-09-19 — "/simulate"의 여러 센서를 한 공간으로
    간주). 판정·장비 제어는 기존 run_agent와 동일하게 센서(물질)별로 독립 처리한다 — 물질마다
    임계값·위험 기준이 달라서 그 판정 자체를 하나로 뭉개면 틀린 결과가 나온다. 종합되는 건
    "LLM이 마지막에 내놓는 설명" 한 곳뿐이다."""
    if not events:
        raise ValueError("events가 비어있음")

    decisions: list[Decision] = []
    judged: list[tuple[dict, rules.Judgement]] = []
    for event in events:
        judgement = rules.judge(event["value"], event["threshold"], event.get("lower_is_worse", False))
        judged.append((event, judgement))
        decisions.append(Decision("rule", "judge", {
            "substance": event["substance"], "severity": judgement.severity.value, "ratio": round(judgement.ratio, 2),
        }))

    location = events[0]["location"]
    exceeded = [(e, j) for e, j in judged if j.exceeded]

    if not exceeded:
        return {
            "location": location, "judged": judged,
            "narrative": f"{location}의 모든 센서가 정상 범위입니다.",
            "tool_trace": [], "equipment_status": [], "notify_log": [], "report": None,
            "decisions": decisions,
        }

    # 장비 제어는 초과한 센서마다 독립적으로 즉시 실행(기존과 동일 원칙, LLM 서술 이전에 처리)
    equipment_status: list[dict] = []
    for event, judgement in exceeded:
        for action in rules.required_equipment_actions(event["category"], judgement.severity):
            record = (
                tools.actuate_equipment(ctx, action) if action.risk == rules.Risk.LOW
                else tools.request_equipment_approval(ctx, action)
            )
            equipment_status.append(record)
            decisions.append(Decision("rule", "equipment", record))

    equipment_summary = (
        "; ".join(f"{r['equipment']}({r['status']})" for r in equipment_status)
        if equipment_status else "해당 없음"
    )
    messages = [
        {"role": "system", "content": _COMPOSITE_SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"{location}에 설치된 센서들의 측정 결과:\n{_composite_event_lines(judged)}\n\n"
            f"이미 자동으로 처리된 장비 조치(사실, 그대로 인용할 것): {equipment_summary}"
        )},
    ]

    narrative, tool_trace, guideline_sources, tool_decisions = _run_llm_tool_loop(
        messages, ctx, llm, exceeded[0][0]["substance"], max_tool_turns,
    )
    decisions.extend(tool_decisions)

    tools.notify(ctx, channel="현장관리자-알림방", message=narrative)
    decisions.append(Decision("rule", "notify", {"channel": "현장관리자-알림방"}))

    report = None
    danger = [(e, j) for e, j in exceeded if j.severity == rules.Severity.DANGER]
    if danger:
        reasons = ", ".join(e["substance"] for e, _ in danger)
        tools.escalate(ctx, reason=f"{reasons} 위험 수준 감지")
        decisions.append(Decision("rule", "escalate", {"reason": reasons}))
        # 리포트 양식은 이벤트 하나를 기준으로 하므로, 여럿 중 가장 심한(ratio가 가장 큰)
        # 것을 대표로 삼는다 — narrative 자체는 이미 종합 서술이라 리포트 본문엔 그대로 실림.
        worst_event, worst_judgement = max(danger, key=lambda pair: pair[1].ratio)
        report = tools.draft_incident_report(
            ctx, worst_event, worst_judgement, narrative, list(dict.fromkeys(guideline_sources))
        )
        decisions.append(Decision("rule", "draft_incident_report", {}))

    return {
        "location": location,
        "judged": judged,
        "narrative": narrative,
        "tool_trace": tool_trace,
        "equipment_status": equipment_status,
        "notify_log": list(ctx.notify_log),
        "report": report,
        "decisions": decisions,
    }


def to_dict(result: dict) -> dict:
    """run_agent() 결과를 JSON 직렬화 가능한 순수 dict로 바꾼다 — 데이터클래스/Enum이
    섞여 있어서 그대로 json.dumps 하면 실패한다. 웹앱(in-process)과 run_cli.py(서브프로세스,
    엣지 스펙 에뮬레이션용) 둘 다 이 함수로 같은 모양의 결과를 만든다."""
    return {
        "judgement": {
            "severity": result["judgement"].severity.value,
            "ratio": round(result["judgement"].ratio, 2),
            "exceeded": result["judgement"].exceeded,
        },
        "narrative": result["narrative"],
        "tool_trace": result["tool_trace"],
        "equipment_status": result.get("equipment_status", []),
        "report": result["report"],
        "decisions": [{"authority": d.authority, "action": d.action, "detail": d.detail} for d in result["decisions"]],
    }


def to_dict_composite(result: dict) -> dict:
    """run_agent_composite() 결과를 JSON 직렬화 가능한 dict로 바꾼다 — to_dict()와 같은
    이유, 다만 이벤트가 여러 개라 judged를 통째로 events 리스트로 펼친다."""
    return {
        "location": result["location"],
        "events": [
            {
                "event": e,
                "judgement": {"severity": j.severity.value, "ratio": round(j.ratio, 2), "exceeded": j.exceeded},
            }
            for e, j in result["judged"]
        ],
        "narrative": result["narrative"],
        "tool_trace": result["tool_trace"],
        "equipment_status": result.get("equipment_status", []),
        "report": result["report"],
        "decisions": [{"authority": d.authority, "action": d.action, "detail": d.detail} for d in result["decisions"]],
    }


_DEMO_EVENTS = [
    {"category": "가스", "substance": "헬륨", "value": 3.0, "unit": "%", "threshold": 5.0, "location": "실험동 가스저장실"},
    {"category": "조리흄", "substance": "일산화탄소", "value": 36, "unit": "ppm", "threshold": 30, "location": "본관 급식실 조리구역"},
    {"category": "가스", "substance": "수소", "value": 22, "unit": "%LEL", "threshold": 10, "location": "실험동 가스저장실"},
    {"category": "조리흄", "substance": "오일미스트", "value": 12, "unit": "mg/m³", "threshold": 5, "location": "본관 급식실 조리구역"},
]


def main() -> None:
    ctx = ToolContext.load()
    llm = Llama(model_path=str(_GGUF_PATH), n_ctx=4096, n_threads=8, verbose=False)

    for event in _DEMO_EVENTS:
        print("=" * 70)
        print(f"이벤트: {event}")
        result = run_agent(event, ctx, llm)
        print(f"판정: {result['judgement'].severity.value} (비율 {result['judgement'].ratio:.2f})")
        print("결정 추적(권한 표시):")
        for d in result["decisions"]:
            print(f"  [{d.authority:4s}] {d.action}: {d.detail}")
        print(f"알림 문장: {result['narrative']}")
        if result["report"]:
            print(f"사고 리포트 초안 출처: {result['report']['guideline_sources']}")
        ctx.notify_log.clear()


if __name__ == "__main__":
    main()
