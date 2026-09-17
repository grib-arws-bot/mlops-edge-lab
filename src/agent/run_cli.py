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

from agent.run import run_agent, to_dict
from agent.tools import ToolContext

_ROOT = Path(__file__).resolve().parents[2]
_GGUF_PATH = _ROOT / "experiments" / "toy-sensor-lora" / "model-Q4_K_M.gguf"


def main() -> None:
    events_path, output_path = sys.argv[1], sys.argv[2]
    events = json.loads(Path(events_path).read_text(encoding="utf-8"))
    n_threads = int(os.environ.get("AGENT_THREADS", "2"))

    ctx = ToolContext.load()
    llm = Llama(model_path=str(_GGUF_PATH), n_ctx=4096, n_threads=n_threads, verbose=False)

    results = []
    for event in events:
        r = run_agent(event, ctx, llm)
        results.append({"event": event, **to_dict(r)})
        ctx.notify_log.clear()

    Path(output_path).write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
