"""MLOps-Edge-Lab 교육용 웹 — 로드맵 1~6번 결과물을 한 화면에서 보여준다.

FastAPI를 새로 고른 게 아니라 MLflow가 이미 내부적으로 쓰고 있어서 venv에 있던 걸
그대로 재사용한다(docs/의사결정_로그.md 33번). 빌드 도구 없이 Jinja2 템플릿 + 순수
HTML/JS로만 프론트를 구성 — 개발자 교육 트랙에서 "군더더기 없는 레퍼런스"로 보여주기 위함.

모델(LLM, 임베딩, FAISS)은 시작할 때 한 번만 로드해서 전역으로 재사용한다 — 요청마다
다시 불러오면 수 초씩 걸려서 데모가 느려진다.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from llama_cpp import Llama
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent import rules
from agent.run import preview, run_agent, to_dict
from agent.tools import ToolContext
from sensors import CATEGORIES, LOCS, SEVERITY_RATIO, substance_lookup

_ROOT = Path(__file__).resolve().parents[2]
_GGUF_PATH = _ROOT / "experiments" / "toy-sensor-lora" / "model-Q4_K_M.gguf"
_F16_PATH = _ROOT / "experiments" / "toy-sensor-lora" / "model-f16.gguf"
_HERE = Path(__file__).resolve().parent
_DECK_PPTX = _ROOT / "docs" / "교육자료" / "MLOps-Edge-Lab_교육자료.pptx"
_STATIC_DIR = _HERE / "static"
_DECK_PDF = _STATIC_DIR / "deck.pdf"
_STATIC_DIR.mkdir(parents=True, exist_ok=True)

_state: dict = {}


def _ensure_deck_pdf() -> None:
    """교육자료 PPT를 PDF로 변환해 웹에 그대로 박아 넣는다.

    파일을 미리 변환해서 커밋해두지 않는 이유는 파이프라인 페이지 수치와 같은 원칙 —
    "화면에 보이는 것은 항상 실제 산출물에서 나온다"를 지키기 위함이다. pptx가 갱신되면
    (mtime 비교) 다음 서버 기동 때 자동으로 다시 변환된다. LibreOffice(soffice)는 PPT
    검증 단계에서 이미 서버에 설치돼 있던 걸 재사용.
    """
    if not _DECK_PPTX.exists():
        return
    if _DECK_PDF.exists() and _DECK_PDF.stat().st_mtime >= _DECK_PPTX.stat().st_mtime:
        return
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            ["soffice", "--headless", "--convert-to", "pdf", "--outdir", tmp, str(_DECK_PPTX)],
            check=True, timeout=120,
        )
        produced = next(Path(tmp).glob("*.pdf"))
        produced.replace(_DECK_PDF)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _state["ctx"] = ToolContext.load()
    _state["llm"] = Llama(model_path=str(_GGUF_PATH), n_ctx=4096, n_threads=8, verbose=False)
    _ensure_deck_pdf()
    _init_control_room()
    control_room_task = asyncio.create_task(_control_room_loop())
    yield
    control_room_task.cancel()
    _state.clear()


app = FastAPI(title="MLOps-Edge-Lab 데모", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(_HERE / "templates"))

# 표준 MLOps 갭 중 "운영 모니터링" — Prometheus 클라이언트로 지표를 노출만 하고,
# Prometheus/Grafana는 별도 docker-compose(infra/monitoring/)로 스크레이핑한다.
# Grafana·Prometheus UI는 MLflow와 같은 이유로 SSH 터널 전용 — 8081은 서비스용으로 아낀다.
METRIC_JUDGE_REQUESTS = Counter("agent_judge_requests_total", "Total /api/judge calls")
METRIC_NARRATE_REQUESTS = Counter("agent_narrate_requests_total", "Total /api/narrate calls", ["edge_profile"])
METRIC_NARRATE_ERRORS = Counter("agent_narrate_errors_total", "Total /api/narrate failures")
METRIC_SEVERITY = Counter("agent_severity_total", "Judged severity counts", ["severity"])
METRIC_EQUIPMENT_ACTUATED = Counter("agent_equipment_actuated_total", "Equipment actuation counts", ["equipment", "status"])
METRIC_FEEDBACK = Counter("agent_feedback_total", "Feedback submissions", ["rating"])
METRIC_CONTROL_ROOM_EVENTS = Counter("agent_control_room_events_total", "Autonomous control-room demo events", ["site", "severity"])
METRIC_NARRATE_LATENCY = Histogram("agent_narrate_latency_seconds", "Time to complete /api/narrate", ["edge_profile"])

# 인프로세스 llm 인스턴스는 하나뿐인데 /api/narrate(사용자 요청)와 통합관제 백그라운드
# 루프가 둘 다 이걸 쓴다. llama.cpp의 Llama 객체는 동시 추론을 지원하지 않아서(같은 KV
# 캐시를 두 스레드가 동시에 건드리면 꼬임) 락으로 직렬화한다 — 서로 겹치면 그냥 순서대로
# 기다렸다가 처리된다(둘 다 실패시키는 것보단 나음).
_llm_lock = threading.Lock()


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


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
            "loop": "평가 기준 미달 시 자동 재학습",
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
            "loop": "운영 피드백 루프 → 3번으로 순환",
        },
        {
            "no": 7, "title": "엣지 배포·검증",
            "definition": "실제 엣지 하드웨어 없이 cgroup(코어·메모리 제한)으로 에뮬레이션해 하한선 추정",
            "role": "GPU 없는/약한 디바이스에서도 실용적 속도가 나오는지 실제 배포 전에 검증",
            "library": "systemd-run(cgroup) · taskset",
            "stats": ["실측: 2코어 tg 7.32 tok/s(8코어 대비 약 3.2배 느림)", "진짜 하드웨어 검증은 아직 — 코어/메모리만 흉내낸 하한선"],
        },
    ]


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"stages": _pipeline_stages()})


@app.get("/education", response_class=HTMLResponse)
def education(request: Request):
    return templates.TemplateResponse(request, "education.html", {"deck_available": _DECK_PDF.exists()})


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
    METRIC_JUDGE_REQUESTS.inc()
    payload = await request.json()
    events = _events_from_payload(payload.get("sensors", []))
    judged = []
    equipment_map = {name: "꺼짐" for name in _ALL_EQUIPMENT}
    for event in events:
        p = preview(event)
        judged.append({"event": event, "judgement": p["judgement"]})
        METRIC_SEVERITY.labels(severity=p["judgement"]["severity"]).inc()
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
    profile_key = payload.get("edge_profile", "server")
    profile = EDGE_PROFILES.get(profile_key, EDGE_PROFILES["server"])
    METRIC_NARRATE_REQUESTS.labels(edge_profile=profile_key).inc()

    if not events:
        return JSONResponse({"error": "최소 1개 센서를 활성화해주세요."}, status_code=400)

    start = time.perf_counter()
    try:
        if profile["cores"] is None:
            with _llm_lock:
                results = _run_events_inprocess(events)
        else:
            results = _run_events_emulated(events, profile["cores"], profile["mem_gb"])
    except Exception as exc:  # noqa: BLE001 — 데모 화면에 원인을 그대로 보여주기 위함
        METRIC_NARRATE_ERRORS.inc()
        return JSONResponse({"error": str(exc)}, status_code=500)
    elapsed = round(time.perf_counter() - start, 2)
    METRIC_NARRATE_LATENCY.labels(edge_profile=profile_key).observe(elapsed)
    for r in results:
        for e in r.get("equipment_status", []):
            METRIC_EQUIPMENT_ACTUATED.labels(equipment=e["equipment"], status=e["status"]).inc()

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
    METRIC_FEEDBACK.labels(rating=payload.get("rating", "unknown")).inc()
    _FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "event": payload.get("event"), "narrative": payload.get("narrative"),
        "rating": payload.get("rating"), "correction": payload.get("correction"),
        "logged_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with _FEEDBACK_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return JSONResponse({"status": "saved"})


# ── 산업안전 통합관제 (시연용) ──────────────────────────────────────────
# 영업/기획 트랙 시연을 위한 페이지 — 사람이 값을 입력하는 /simulate와 달리, 여러 현장의
# 센서 이벤트가 스스로 발생하는 것처럼 백그라운드에서 주기적으로 실제 파이프라인을 돌린다.
# "그럴듯하게 보이는 가짜 숫자"가 아니라 매번 preview()/run_agent()로 실제 계산한 결과다
# — 이 프로젝트 전체를 관통하는 원칙(하드코딩 대신 실제 산출물)과 동일하게 맞췄다.
#
# 카드 하나 = 현장 하나. 카드 안에 멀티 센서(2~3개)를 두고, 그중 하나가 무작위로 바뀌면서
# 이상 감지 시 장비 조치 + 자연어 안내(알림 박스)까지 같은 카드 안에서 전부 보여준다.
# 그 센서가 다시 정상으로 돌아오면 알림 박스는 사라진다(사용자 요청) — "이상이 지금도
# 진행 중인가"만 카드 하나만 보고 알 수 있게 하기 위함.
_CONTROL_ROOM_SITES = {
    "본관 급식실 조리구역": {"category": "조리흄", "substances": ["일산화탄소", "오일미스트", "포름알데히드"]},
    "본관 급식실 배식구역": {"category": "공기질", "substances": ["CO2", "미세먼지(PM10)", "온도"]},
    "지하 기계실": {"category": "가스", "substances": ["수소", "헬륨"]},
    "실험동 가스저장실": {"category": "가스", "substances": ["수소", "아르곤", "질소"]},
    "2층 사무실": {"category": "공기질", "substances": ["CO2", "TVOC"]},
    "체육관": {"category": "공기질", "substances": ["CO2", "습도"]},
}
_CONTROL_ROOM_INTERVAL_SECONDS = 12

_control_room_state: dict[str, dict] = {}
# /simulate와 달리 통합관제는 여러 사람이 같이 보는 "관제실 화면 하나"라는 컨셉이라,
# 요청마다 프로파일을 넘기는 게 아니라 서버 쪽 전역 설정 하나로 둔다(이 화면을 보는
# 모두가 같은 조건을 본다) — EDGE_PROFILES는 /simulate와 동일한 것을 재사용.
_control_room_edge_profile = "server"


def _now_hms() -> str:
    return time.strftime("%H:%M:%S")


def _random_event(site: str, substance: str) -> dict:
    category = _CONTROL_ROOM_SITES[site]["category"]
    unit, threshold = substance_lookup(category, substance)
    # 정상이 더 자주 나오게 가중치를 둠 — 알림이 쉴 새 없이 뜨면 "위험 신호"의 의미가
    # 옅어져서 오히려 데모로서 설득력이 떨어진다.
    severity = random.choices(["정상", "주의", "심각"], weights=[55, 30, 15])[0]
    ratio = SEVERITY_RATIO.get(severity, 0.6)
    value = round(threshold * ratio, 2)
    return {"category": category, "substance": substance, "value": value, "threshold": threshold, "unit": unit, "location": site}


def _init_control_room() -> None:
    """서버 기동 시 각 현장의 모든 센서를 정상 상태 기본값으로 채워둔다 — 첫 폴링 전에도
    화면이 비어있지 않게 하기 위함. 알림(alert)은 처음엔 당연히 없음(None)."""
    for site, spec in _CONTROL_ROOM_SITES.items():
        sensors = {}
        for substance in spec["substances"]:
            unit, threshold = substance_lookup(spec["category"], substance)
            sensors[substance] = {
                "value": round(threshold * SEVERITY_RATIO["정상"], 2), "threshold": threshold,
                "unit": unit, "severity": "정상", "updated_at": _now_hms(),
            }
        _control_room_state[site] = {"sensors": sensors, "alert": None}


def _update_sensor(site: str, substance: str, event: dict, preview_result: dict) -> None:
    _control_room_state[site]["sensors"][substance] = {
        "value": event["value"], "threshold": event["threshold"], "unit": event["unit"],
        "severity": preview_result["judgement"]["severity"], "updated_at": _now_hms(),
    }


def _run_control_room_narrative(event: dict) -> dict:
    """백그라운드 스레드(run_in_executor)에서 호출됨 — asyncio 이벤트 루프를 LLM 추론
    시간(수 초) 동안 막지 않기 위해 별도 스레드로 뺐다. 선택된 엣지 프로파일이 서버
    기본이면 인프로세스 모델을(_llm_lock으로 /api/narrate와 직렬화), 아니면 /simulate와
    동일한 cgroup 에뮬레이션(_run_events_emulated)을 그대로 재사용한다."""
    profile = EDGE_PROFILES.get(_control_room_edge_profile, EDGE_PROFILES["server"])
    if profile["cores"] is None:
        with _llm_lock:
            result = run_agent(event, _state["ctx"], _state["llm"])
            _state["ctx"].notify_log.clear()
            return to_dict(result)
    return _run_events_emulated([event], profile["cores"], profile["mem_gb"])[0]


def _set_site_alert(site: str, substance: str, severity: str, result: dict, elapsed: float) -> None:
    _control_room_state[site]["alert"] = {
        "substance": substance, "severity": severity, "narrative": result["narrative"],
        "equipment": result.get("equipment_status", []), "logged_at": _now_hms(),
        "elapsed": elapsed, "edge_label": EDGE_PROFILES.get(_control_room_edge_profile, EDGE_PROFILES["server"])["label"],
    }


def _clear_site_alert_if_owner(site: str, substance: str) -> None:
    """지금 켜진 알림이 '이 센서' 때문에 켜진 게 맞을 때만 끈다 — 다른 센서가 원인인
    알림까지 같이 꺼버리는 걸 방지."""
    alert = _control_room_state[site]["alert"]
    if alert and alert["substance"] == substance:
        _control_room_state[site]["alert"] = None


async def _control_room_loop() -> None:
    """12초마다 무작위 현장의 무작위 센서 하나를 골라 실제 규칙 판정을 다시 계산하고,
    주의/위험이면 LLM까지 돌려 그 현장 카드의 알림으로 반영한다. 정상으로 돌아오면(그
    알림을 유발한 센서일 때만) 알림을 지운다. 백그라운드 태스크가 예외로 죽으면 그 뒤로
    통합관제 페이지가 영원히 멈춰버리므로, 매 틱을 try/except로 감싸 하나 실패해도 다음
    틱은 계속되게 한다."""
    loop = asyncio.get_event_loop()
    while True:
        await asyncio.sleep(_CONTROL_ROOM_INTERVAL_SECONDS)
        try:
            site = random.choice(list(_CONTROL_ROOM_SITES))
            substance = random.choice(_CONTROL_ROOM_SITES[site]["substances"])
            event = _random_event(site, substance)
            preview_result = preview(event)
            severity = preview_result["judgement"]["severity"]
            _update_sensor(site, substance, event, preview_result)
            METRIC_CONTROL_ROOM_EVENTS.labels(site=site, severity=severity).inc()

            if severity != "정상":
                start = time.perf_counter()
                result = await loop.run_in_executor(None, _run_control_room_narrative, event)
                elapsed = round(time.perf_counter() - start, 2)
                _set_site_alert(site, substance, severity, result, elapsed)
            else:
                _clear_site_alert_if_owner(site, substance)
        except Exception as exc:  # noqa: BLE001 — 백그라운드 루프는 절대 죽으면 안 됨
            print(f"[control-room] tick 실패: {exc}")


@app.get("/control-room", response_class=HTMLResponse)
def control_room(request: Request):
    return templates.TemplateResponse(
        request, "control_room.html",
        {"sites": list(_CONTROL_ROOM_SITES.keys()), "edge_profiles": EDGE_PROFILES},
    )


@app.get("/api/control-room/status")
def control_room_status():
    sites = [{"name": name, **_control_room_state.get(name, {})} for name in _CONTROL_ROOM_SITES]
    return JSONResponse({"sites": sites, "edge_profile": _control_room_edge_profile})


@app.post("/api/control-room/edge-profile")
async def set_control_room_edge_profile(request: Request):
    """관제실 화면은 여러 사람이 같이 보는 하나의 화면이라, 여기서 바꾼 엣지 프로파일은
    다음 백그라운드 틱부터 전역으로 적용된다(요청 보낸 사람만 바뀌는 게 아님)."""
    global _control_room_edge_profile
    payload = await request.json()
    key = payload.get("edge_profile", "server")
    if key not in EDGE_PROFILES:
        return JSONResponse({"error": "알 수 없는 엣지 프로파일"}, status_code=400)
    _control_room_edge_profile = key
    return JSONResponse({"status": "ok", "edge_profile": key, "label": EDGE_PROFILES[key]["label"]})
