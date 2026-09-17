"""MLOps-Edge-Lab 교육용 웹 — 로드맵 1~6번 결과물을 한 화면에서 보여준다.

FastAPI를 새로 고른 게 아니라 MLflow가 이미 내부적으로 쓰고 있어서 venv에 있던 걸
그대로 재사용한다(docs/의사결정_로그.md 33번). 빌드 도구 없이 Jinja2 템플릿 + 순수
HTML/JS로만 프론트를 구성 — 개발자 교육 트랙에서 "군더더기 없는 레퍼런스"로 보여주기 위함.

모델(LLM, 임베딩, FAISS)은 시작할 때 한 번만 로드해서 전역으로 재사용한다 — 요청마다
다시 불러오면 수 초씩 걸려서 데모가 느려진다.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from llama_cpp import Llama

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent import rules
from agent.run import preview, run_agent, to_dict
from agent.tools import ToolContext
from sensors import CATEGORIES, LOCS, SEVERITY_RATIO, substance_lookup

_ROOT = Path(__file__).resolve().parents[2]
_GGUF_PATH = _ROOT / "experiments" / "toy-sensor-lora" / "model-Q4_K_M.gguf"
_F16_PATH = _ROOT / "experiments" / "toy-sensor-lora" / "model-f16.gguf"
_HERE = Path(__file__).resolve().parent

_state: dict = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _state["ctx"] = ToolContext.load()
    _state["llm"] = Llama(model_path=str(_GGUF_PATH), n_ctx=4096, n_threads=8, verbose=False)
    yield
    _state.clear()


app = FastAPI(title="MLOps-Edge-Lab 데모", lifespan=lifespan)
templates = Jinja2Templates(directory=str(_HERE / "templates"))


def _file_mb(path: Path) -> float | None:
    return round(path.stat().st_size / (1024 * 1024), 0) if path.exists() else None


def _count_lines(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _mlflow_finetune_metrics() -> dict | None:
    """MLflow에 기록된 실제 학습·평가 지표를 읽어온다. MLflow가 안 떠 있으면 조용히
    None을 반환하고, 화면에서는 '측정값 없음'으로 대체한다(조용한 실패 위장 금지 원칙과
    같은 맥락 — 못 가져왔다는 사실 자체는 숨기지 않는다)."""
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        client = MlflowClient()
        exp = client.get_experiment_by_name("sllm-finetune")
        if not exp:
            return None
        runs = client.search_runs([exp.experiment_id], order_by=["start_time DESC"], max_results=20)
        train_run = next((r for r in runs if r.info.run_name == "toy-sensor-lora"), None)
        eval_run = next((r for r in runs if r.info.run_name == "toy-sensor-lora-eval"), None)
        return {
            "train_loss": train_run.data.metrics.get("train_loss") if train_run else None,
            "eval_loss": train_run.data.metrics.get("eval_loss") if train_run else None,
            "token_acc": train_run.data.metrics.get("eval_mean_token_accuracy") if train_run else None,
            "rouge_l": eval_run.data.metrics.get("avg_rougeL") if eval_run else None,
        }
    except Exception:
        return None


def _extraction_success_rate() -> str:
    path = _ROOT / "data" / "processed" / "extraction_results.jsonl"
    if not path.exists():
        return "미실행"
    total = ok = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            if json.loads(line).get("ok"):
                ok += 1
    return f"{ok}/{total}건 ({ok / total:.0%})" if total else "0건"


def _pipeline_stages() -> list[dict]:
    f16_mb = _file_mb(_F16_PATH)
    q4_mb = _file_mb(_GGUF_PATH)
    chunk_count = _count_lines(_ROOT / "data" / "processed" / "rag_chunks.jsonl")
    ft = _mlflow_finetune_metrics()

    if ft and ft["train_loss"] is not None:
        ft_stats = [
            f"MLflow 실측 — train_loss {ft['train_loss']:.3f} · eval_loss {ft['eval_loss']:.3f} · 토큰정확도 {ft['token_acc']:.1%}",
            f"평가 ROUGE-L {ft['rouge_l']:.3f} (합성 데이터 5에폭 과적합 수치 — 품질 지표 아닌 배관 동작 증거)" if ft["rouge_l"] else "평가 run 없음",
        ]
    else:
        ft_stats = ["MLflow 연결 안 됨 — 서버에서 mlflow 프로세스 확인 필요"]

    return [
        {
            "no": 1, "title": "MLOps 인프라 구축",
            "definition": "Python 환경·실험 추적·프로젝트 구조 등 재현 가능한 개발 기반 마련",
            "role": "이후 모든 실험을 추적·재현 가능하게 만드는 기반. 여기서부터 '왜 이렇게 했는가'를 기록하는 습관이 시작됨",
            "library": "uv(패키지·venv 관리) · MLflow(Tracking, SQLite backend) · tmux(상시 실행)",
            "stats": ["AI 서버: GPU 4장(RTX 4000 SFF Ada, 20GB/장), 48코어 CPU", "MLflow·웹서비스 전부 tmux 세션으로 상시 구동"],
        },
        {
            "no": 2, "title": "데이터 추출",
            "definition": "HWP/HWPX/PDF/DOCX/XLSX/이미지/ZIP 등 다형식 문서를 텍스트로 변환",
            "role": "RAG 코퍼스와 향후 학습 데이터의 원천 확보",
            "library": "pymupdf · pyhwp(hwp5txt) · python-docx · openpyxl · Tesseract OCR · LibreOffice(안전망)",
            "stats": [f"실제 안전문서 처리: {_extraction_success_rate()}", "폴백 사슬 — 실패해도 조용히 넘어가지 않고 이유를 남김"],
        },
        {
            "no": 3, "title": "sLLM 파인튜닝",
            "definition": "LoRA로 베이스 모델에 도메인 문체(안전 알림 톤)를 가볍게 학습",
            "role": "센서 이벤트 → 간결한 한국어 알림 문장으로 바꾸는 능력을 익힘. 위험 판정 자체는 여기서도 LLM에게 안 맡김",
            "library": "Qwen3-4B-Instruct-2507(Apache-2.0) · peft(LoRA r=16) · trl(SFTTrainer) · MLflow",
            "stats": ft_stats,
            "loop": "평가 기준 미달 시 자동 재학습 (구현됨 — train/auto_retrain.py)",
            "mlflow_hint": True,
        },
        {
            "no": 4, "title": "양자화 + 패키징",
            "definition": "GGUF 변환 후 Q4_K_M 양자화(4비트 압축)",
            "role": "GPU 있는/없는 디바이스 모두 하나의 파일로 대응 가능하게 경량화",
            "library": "llama.cpp(convert_hf_to_gguf, llama-quantize) · GGUF Q4_K_M",
            "stats": (
                [f"크기: f16 {f16_mb:.0f}MB → Q4_K_M {q4_mb:.0f}MB (약 {f16_mb / q4_mb:.1f}배 압축)" if f16_mb and q4_mb else "크기: (파일 없음)"]
                + ["실측 속도: CPU(8스레드) 23.6 tok/s · GPU(RTX4000 Ada) 87.0 tok/s (약 3.7배)"]
            ),
        },
        {
            "no": 5, "title": "RAG",
            "definition": "질문과 관련된 문서 조각을 검색해 근거로 붙여서 답변 생성",
            "role": "모델을 다시 학습시키지 않고 방대한 문서 지식을 활용. 근거 없으면 '모른다'고 답함",
            "library": "intfloat/multilingual-e5-small(임베딩) · FAISS(IndexFlatIP)",
            "stats": [f"인덱싱된 문서 조각: {chunk_count}개" if chunk_count else "인덱스 없음", "청킹: 문단 슬라이딩윈도우(700자, 겹침 100자)"],
        },
        {
            "no": 6, "title": "Agent",
            "definition": "판정(규칙) → 필요시 도구 호출(LLM) → 장비 제어·알림·에스컬레이션",
            "role": "위험도·장비 제어는 규칙이, 문서 검색·문구 작성은 LLM이 — 역할을 명시적으로 나눠 추적 가능하게 함",
            "library": "llama-cpp-python(tool-calling) · 자체 규칙 엔진(rules.py)",
            "stats": ["고위험 장비(가스차단·소화방출)는 자동 실행 안 함 — 승인 대기만", "모든 결정에 rule/llm 권한 태그 기록"],
            "loop": "운영 피드백 루프 → 3번으로 순환 (구현됨 — 시뮬레이션 페이지 👍/👎 → incorporate_feedback.py)",
        },
        {
            "no": 7, "title": "엣지 배포·검증",
            "definition": "실제 엣지 하드웨어 없이 cgroup(코어·메모리 제한)으로 에뮬레이션해 하한선 추정",
            "role": "GPU 없는/약한 디바이스에서도 실용적 속도가 나오는지 실제 배포 전에 검증",
            "library": "systemd-run(cgroup) · taskset",
            "stats": ["실측: 2코어 tg 7.32 tok/s(8코어 대비 약 3.2배 느림)", "진짜 하드웨어 검증은 아직 — 코어/메모리만 흉내낸 하한선"],
        },
    ]


_REMAINING_WORK = [
    "모델 레지스트리 — MLflow Tracking만 쓰는 중, 버전 승격(Staging→Production) 관리는 안 함",
    "운영 모니터링 대시보드 — 지연시간·오류율·정확도 추이를 실시간으로 재는 도구(Prometheus/Grafana 등) 없음",
    "데이터 버저닝 — 추출된 문서·학습 데이터에 버전 태그 없음(DVC 등 미사용)",
    "CI/CD — git 저장소 자체가 아직 없어서 자동화된 빌드·테스트·배포 파이프라인 없음. 전부 SSH로 수동 실행 중",
]


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        request, "index.html", {"stages": _pipeline_stages(), "remaining_work": _REMAINING_WORK}
    )


_MAX_SENSORS = 5
_ALL_EQUIPMENT = ["환풍기", "가스차단기", "소화설비(대기)", "소화설비(방출)"]

# 서버 기본은 이미 로드된 인프로세스 모델을 그대로 써서 빠르다(코어 제한 없음 = 지금
# 서버 그대로). 나머지는 로드맵 7번에서 검증한 cgroup 에뮬레이션(의사결정_로그 32번)을
# 요청 단위로 재사용 — 매번 모델을 새로 불러오는 대신, 실제로 그 코어/메모리 조건에서
# 돌린 진짜 결과를 보여주기 위해서다(가짜로 숫자만 줄이는 게 아니라).
EDGE_PROFILES = {
    "server": {"label": "서버 기본 (제한 없음)", "cores": None, "mem_gb": None},
    "2c32g": {"label": "2코어 / 32GB", "cores": 2, "mem_gb": 32},
    "2c4g": {"label": "2코어 / 4GB", "cores": 2, "mem_gb": 4},
    "4c8g": {"label": "4코어 / 8GB", "cores": 4, "mem_gb": 8},
}

_CATEGORY_SUBSTANCES = {cat: [row[0] for row in table] for cat, (table, _action) in CATEGORIES.items()}
_CATEGORY_SUBSTANCES_JSON = json.dumps(_CATEGORY_SUBSTANCES, ensure_ascii=False)


_FEEDBACK_PATH = _ROOT / "data" / "processed" / "feedback.jsonl"


def _default_location(idx: int) -> str:
    """위치 입력을 없앤 대신, 센서 슬롯 번호로 결정론적으로 위치를 배정한다 — 데모
    편의를 위한 placeholder이지 실제 위치 정보가 아니다."""
    return LOCS[(idx - 1) % len(LOCS)]


def _severity_to_event(category: str, substance: str, severity: str, idx: int) -> dict | None:
    lookup = substance_lookup(category, substance)
    if not lookup:
        return None
    unit, threshold = lookup
    ratio = SEVERITY_RATIO.get(severity, 0.6)
    value = round(threshold * ratio, 2)
    return {
        "category": category, "substance": substance, "value": value, "threshold": threshold,
        "unit": unit, "location": _default_location(idx),
    }


def _events_from_payload(sensors: list[dict]) -> list[dict]:
    events = []
    for idx, s in enumerate(sensors, start=1):
        event = _severity_to_event(s.get("category", ""), s.get("substance", ""), s.get("severity", "정상"), idx)
        if event:
            events.append(event)
    return events


def _aggregate_equipment(results: list[dict]) -> list[dict]:
    status = {name: "꺼짐" for name in _ALL_EQUIPMENT}
    for r in results:
        for e in r.get("equipment_status", []):
            status[e["equipment"]] = e["status"]
    return [{"equipment": name, "status": status[name]} for name in _ALL_EQUIPMENT]


def _run_events_inprocess(events: list[dict]) -> list[dict]:
    results = []
    for event in events:
        r = run_agent(event, _state["ctx"], _state["llm"])
        results.append({"event": event, **to_dict(r)})
        _state["ctx"].notify_log.clear()
    return results


def _run_events_emulated(events: list[dict], cores: int, mem_gb: int) -> list[dict]:
    """엣지 스펙 에뮬레이션 — 별도 프로세스를 systemd-run(cgroup)+taskset으로 감싸서
    실제로 그 코어 수·메모리로 제한된 조건에서 돌린다(로드맵 7번, 의사결정_로그 32번과
    동일한 방법). 매번 모델을 새로 불러와서 인프로세스보다 느리지만, 숫자가 진짜다."""
    core_list = ",".join(str(i) for i in range(cores))
    python_bin = _ROOT / ".venv" / "bin" / "python"

    with tempfile.TemporaryDirectory() as tmp:
        events_path = Path(tmp) / "events.json"
        output_path = Path(tmp) / "output.json"
        events_path.write_text(json.dumps(events, ensure_ascii=False), encoding="utf-8")

        cmd = [
            "systemd-run", "--user", "--scope", "--quiet",
            "-p", f"CPUQuota={cores * 100}%",
            "-p", f"MemoryMax={mem_gb}G",
            "-p", "MemorySwapMax=0",
            "--", "taskset", "-c", core_list,
            str(python_bin), "-m", "agent.run_cli", str(events_path), str(output_path),
        ]
        proc = subprocess.run(
            cmd, cwd=str(_ROOT), timeout=180, capture_output=True, text=True,
            env={**os.environ, "PYTHONPATH": str(_ROOT / "src"), "AGENT_THREADS": str(cores)},
        )
        if proc.returncode != 0 or not output_path.exists():
            raise RuntimeError(f"엣지 에뮬레이션 실행 실패(exit {proc.returncode}): {proc.stderr[-1500:]}")
        return json.loads(output_path.read_text(encoding="utf-8"))


@app.get("/simulate", response_class=HTMLResponse)
def simulate_form(request: Request):
    return templates.TemplateResponse(
        request, "simulate.html",
        {
            "sensors_range": range(1, _MAX_SENSORS + 1), "categories": _CATEGORY_SUBSTANCES,
            "categories_json": _CATEGORY_SUBSTANCES_JSON, "edge_profiles": EDGE_PROFILES,
        },
    )


@app.post("/api/judge")
async def api_judge(request: Request):
    """① 즉시 반응 단계 — 규칙만으로 판정·장비 상태를 계산한다. LLM 호출이 전혀 없어서
    수 밀리초 안에 끝난다. 사용자가 '장비 제어는 즉시 동작해야 한다'고 요청한 부분."""
    payload = await request.json()
    events = _events_from_payload(payload.get("sensors", []))
    judged = []
    equipment_map = {name: "꺼짐" for name in _ALL_EQUIPMENT}
    for event in events:
        p = preview(event)
        judged.append({"event": event, "judgement": p["judgement"]})
        for e in p["equipment_status"]:
            equipment_map[e["equipment"]] = e["status"]
    equipment = [{"equipment": n, "status": equipment_map[n]} for n in _ALL_EQUIPMENT]
    return JSONResponse({"events": judged, "equipment": equipment})


@app.post("/api/narrate")
async def api_narrate(request: Request):
    """② LLM 처리 단계 — 실제 run_agent()(또는 엣지 에뮬레이션)를 돌려서 알림 문구·결정
    추적까지 완성한다. 시간이 걸리는 부분이라 프론트가 이 호출 동안 스톱워치를 보여준다."""
    payload = await request.json()
    events = _events_from_payload(payload.get("sensors", []))
    profile = EDGE_PROFILES.get(payload.get("edge_profile", "server"), EDGE_PROFILES["server"])

    if not events:
        return JSONResponse({"error": "최소 1개 센서를 활성화해주세요."}, status_code=400)

    start = time.perf_counter()
    try:
        if profile["cores"] is None:
            results = _run_events_inprocess(events)
        else:
            results = _run_events_emulated(events, profile["cores"], profile["mem_gb"])
    except Exception as exc:  # noqa: BLE001 — 데모 화면에 원인을 그대로 보여주기 위함
        return JSONResponse({"error": str(exc)}, status_code=500)
    elapsed = round(time.perf_counter() - start, 2)

    return JSONResponse({
        "results": results,
        "equipment": _aggregate_equipment(results),
        "elapsed": elapsed,
        "edge_label": profile["label"],
    })


@app.post("/api/feedback")
async def api_feedback(request: Request):
    """운영 피드백 루프(6→1/3번) — 사람이 "이 알림 문구가 적절했는지"를 여기서 남기면
    data/processed/feedback.jsonl에 쌓인다. train/incorporate_feedback.py가 이 파일을
    학습 데이터로 변환해 다음 재학습(auto_retrain.py)에 반영한다. 매 피드백마다 즉시
    재학습하지 않는 이유: 노이즈 하나에 모델이 흔들리는 걸 막기 위해 배치로 처리한다."""
    payload = await request.json()
    _FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "event": payload.get("event"), "narrative": payload.get("narrative"),
        "rating": payload.get("rating"), "correction": payload.get("correction"),
        "logged_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with _FEEDBACK_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return JSONResponse({"status": "saved"})
