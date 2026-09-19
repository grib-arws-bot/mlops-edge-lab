"""명령줄 진입점 — 이벤트 목록(JSON 파일)을 받아 run_agent()를 실행하고 결과를 JSON
파일로 출력한다. 엣지 스펙 에뮬레이션(systemd-run+taskset로 코어/메모리 제한)이 이
프로세스 전체를 감싸서 실행할 때 쓴다(web/app.py 참고, docs/의사결정_로그.md 32번의
방법을 재사용).

결과를 stdout이 아니라 파일로 쓰는 이유: 모델 로딩 중 라이브러리들이 stdout에 경고를
찍을 수 있어서, JSON과 섞이면 파싱이 깨진다. 파일로 분리하면 그럴 걱정이 없다.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from llama_cpp import Llama

from agent.run import run_agent, run_agent_composite, to_dict, to_dict_composite
from agent.tools import ToolContext

_ROOT = Path(__file__).resolve().parents[2]
_GGUF_PATH = _ROOT / "experiments" / "toy-sensor-lora" / "model-Q4_K_M.gguf"


def main() -> None:
    events_path, output_path = sys.argv[1], sys.argv[2]
    events = json.loads(Path(events_path).read_text(encoding="utf-8"))
    n_threads = int(os.environ.get("AGENT_THREADS", "2"))
    # 엣지 프로파일의 GPU 유무 축(web/app.py의 EDGE_PROFILES, 의사결정_로그 62번) —
    # taskset의 CPU 코어 제한과는 무관하게 GPU/PCIe 접근은 그대로 가능해서 같이 걸어도 됨.
    n_gpu_layers = -1 if os.environ.get("AGENT_GPU") == "1" else 0

    ctx = ToolContext.load()
    llm = Llama(model_path=str(_GGUF_PATH), n_ctx=4096, n_threads=n_threads, n_gpu_layers=n_gpu_layers, verbose=False)

    # AGENT_COMPOSITE=1이면 여러 센서를 한 공간으로 간주해 LLM이 종합 의견 하나만 낸다
    # (web/app.py의 /simulate 복합 모드, 2026-09-19). 안 켜져 있으면(control-room 등
    # 기존 호출부는 이 값을 아예 안 넘김) 기존처럼 이벤트별로 독립 처리한다 — 하위 호환.
    if os.environ.get("AGENT_COMPOSITE") == "1":
        r = run_agent_composite(events, ctx, llm)
        output = to_dict_composite(r)
    else:
        results = []
        for event in events:
            r = run_agent(event, ctx, llm)
            results.append({"event": event, **to_dict(r)})
            ctx.notify_log.clear()
        output = results

    Path(output_path).write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
