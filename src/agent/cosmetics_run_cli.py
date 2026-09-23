"""명령줄 진입점 — 화장품 PoC 질문 하나를 받아 run_cosmetics_agent()를 실행하고 결과를
JSON 파일로 출력한다. 엣지 스펙 에뮬레이션(systemd-run+taskset로 코어/메모리 제한)이 이
프로세스 전체를 감싸서 실행할 때 쓴다(agent/run_cli.py·web/app.py의 EDGE_PROFILES와
동일한 방법 재사용 — 로드맵 7번, 의사결정_로그 32번).

결과를 stdout이 아니라 파일로 쓰는 이유는 run_cli.py와 같다: 모델 로딩 중 라이브러리
경고가 stdout에 섞이면 JSON 파싱이 깨진다.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from llama_cpp import Llama

from agent.cosmetics_run import run_cosmetics_agent
from agent.cosmetics_tools import CosmeticsToolContext

_ROOT = Path(__file__).resolve().parents[2]
_GGUF_PATH = _ROOT / "experiments" / "toy-sensor-lora" / "model-Q4_K_M.gguf"


def main() -> None:
    input_path, output_path = sys.argv[1], sys.argv[2]
    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    question = payload["question"]
    live_ticks = payload.get("live_ticks")
    n_threads = int(os.environ.get("AGENT_THREADS", "2"))
    n_gpu_layers = -1 if os.environ.get("AGENT_GPU") == "1" else 0

    ctx = CosmeticsToolContext()
    llm = Llama(model_path=str(_GGUF_PATH), n_ctx=4096, n_threads=n_threads, n_gpu_layers=n_gpu_layers, verbose=False)

    result = run_cosmetics_agent(question, ctx, llm, live_ticks=live_ticks)
    Path(output_path).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
