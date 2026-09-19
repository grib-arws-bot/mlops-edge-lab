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
import re
import urllib.error
import urllib.request
import random
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from llama_cpp import Llama
import numpy as np
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent import rules
from agent.run import preview, run_agent, run_agent_composite, to_dict, to_dict_composite
from agent.tools import ToolContext
from collect import registry as collect_registry
from collect.storage import collection_summary
from rag import query_edu
from rag.chunk import chunk_text
from sensors import CATEGORIES, LOCS, SEVERITY_RATIO, VALUE_CEILING, is_lower_is_worse, substance_lookup

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


def _load_gpu_llm_or_disable_gpu_profiles() -> None:
    """GPU 프로파일(EDGE_PROFILES의 gpu=True 항목)이 실제로 쓸모 있으려면 GPU로
    오프로드된 모델이 정말 로드돼 있어야 한다. llama-cpp-python이 CUDA 없이 빌드된
    환경이면 여기서 실패하는데, 그럴 땐 "GPU"라고 표시된 채 몰래 CPU로 도는 걸 막기
    위해 그 항목들을 EDGE_PROFILES에서 아예 지워버린다 — 선택 목록에 안 뜨면 거짓말할
    일도 없다."""
    try:
        _state["llm_gpu"] = Llama(
            model_path=str(_GGUF_PATH), n_ctx=4096, n_threads=8, n_gpu_layers=-1, verbose=False,
        )
    except Exception as exc:  # noqa: BLE001 — 기동 로그에 원인을 남기고 GPU 프로파일만 제거
        print(f"[startup] GPU 모델 로드 실패, GPU 프로파일 비활성화: {exc}")
        _state["llm_gpu"] = None
        for key in [k for k, p in EDGE_PROFILES.items() if p.get("gpu")]:
            del EDGE_PROFILES[key]


def _cleanup_orphaned_edge_scopes() -> None:
    """서버가 막 기동했다는 건, 이전 프로세스가 추적하던 진행 중 시뮬레이션은 개념적으로
    전부 끝났어야 한다는 뜻이다 — 그런데 엣지 에뮬레이션은 systemd 스코프로 독립적인
    생명주기를 가져서(83번 항목, 타임아웃 좀비 문제), 이전 인스턴스가 재시작 직전에
    막 시작시킨 스코프는 새 서버가 전혀 모른 채로 계속 CPU를 붙잡고 있을 수 있다
    (2026-09-19, 동시성 테스트 중 실제로 겪음 — 재시작 타이밍과 겹쳐 좀비 2개 발생).
    기동 시점에 이름 패턴(mlops-edge-emu-*)으로 남아있는 걸 전부 정리해서, 새 인스턴스는
    항상 깨끗한 상태로 시작하게 한다."""
    try:
        result = subprocess.run(
            ["systemctl", "--user", "list-units", "mlops-edge-emu-*.scope", "--no-legend", "--all"],
            capture_output=True, text=True, timeout=10,
        )
        units = [line.split()[0] for line in result.stdout.splitlines() if line.strip()]
        for unit in units:
            subprocess.run(["systemctl", "--user", "stop", unit], capture_output=True, timeout=10)
        if units:
            print(f"[startup] 이전 인스턴스의 좀비 엣지 에뮬레이션 스코프 {len(units)}개 정리: {units}")
    except Exception as exc:  # noqa: BLE001 — 정리 실패로 서버 기동 자체가 막히면 안 됨
        print(f"[startup] 좀비 스코프 정리 중 오류(무시하고 계속): {exc}")


def _load_edu_rag_or_disable() -> None:
    """교육 RAG 인덱스가 아직 안 만들어졌을 수도 있다(운영 콘솔에서 소스를 새로 고를
    때마다 다시 빌드하는 게 아니라 수동 빌드 스크립트라서) — 없으면 조용히 숨기지 않고
    /edu-admin 화면에서 "인덱스 없음"이라고 명시한다."""
    if not query_edu.index_available():
        _state["edu_embed_model"] = None
        _state["edu_index"] = None
        _state["edu_meta"] = None
        _state["edu_bm25"] = None
        return
    try:
        _state["edu_embed_model"] = SentenceTransformer(query_edu.EMBED_MODEL)
        _state["edu_index"] = query_edu.load_index()
        _state["edu_meta"] = query_edu.load_meta()
        _state["edu_bm25"] = query_edu.load_bm25()  # 하이브리드 검색(2026-09-19, 88번)
    except Exception as exc:  # noqa: BLE001 — 로드 실패해도 나머지 페이지는 정상 동작해야 함
        print(f"[startup] 교육 RAG 인덱스 로드 실패: {exc}")
        _state["edu_embed_model"] = None
        _state["edu_index"] = None
        _state["edu_meta"] = None
        _state["edu_bm25"] = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _cleanup_orphaned_edge_scopes()
    _state["ctx"] = ToolContext.load()
    _state["llm"] = Llama(model_path=str(_GGUF_PATH), n_ctx=4096, n_threads=8, verbose=False)
    _load_gpu_llm_or_disable_gpu_profiles()
    _load_edu_rag_or_disable()
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
# GPU 모델은 별개의 Llama 인스턴스(별도 KV 캐시)라 CPU 모델과는 서로 안 겹쳐도 되지만,
# GPU 모델 자기 자신끼리는(예: /simulate와 통합관제가 동시에 GPU 프로파일을 고르는 경우)
# 여전히 직렬화가 필요해서 별도 락을 둔다.
_llm_gpu_lock = threading.Lock()


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
        # MLflow가 죽어있으면 기본 클라이언트는 몇 분씩 재시도하며 요청 자체를 블로킹한다
        # (실제로 겪음 — 68번 포트 이전 작업 중 MLflow가 내려간 상태로 이 페이지가 응답
        # 없이 멈춰버림). 짧은 타임아웃을 강제해서 "MLflow 연결 안 됨"으로 빨리 넘어가게 한다.
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "3")
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri("http://127.0.0.1:8082")
        client = MlflowClient()
        exp = client.get_experiment_by_name("sllm-finetune")
        if not exp:
            return None
        runs = client.search_runs([exp.experiment_id], order_by=["start_time DESC"], max_results=20)
        # auto_retrain.py(38번 항목) 도입 이후 run 이름이 "toy-sensor-lora-auto-attemptN"
        # 식으로 바뀌었는데, 여기는 옛날 고정 이름만 찾고 있어서 최근 20개 run이 전부
        # auto-attempt 계열이면 항상 None이 되는(= "MLflow 연결 안 됨"으로 잘못 표시되는)
        # 버그가 있었다 — 실제로는 연결이 멀쩡한데 메시지가 오해를 불렀다. 두 이름 패턴을
        # 다 인식하도록 수정.
        def _is_train_run(name: str) -> bool:
            return name == "toy-sensor-lora" or (name.startswith("toy-sensor-lora-auto-attempt") and not name.endswith("-eval"))

        def _is_eval_run(name: str) -> bool:
            return name == "toy-sensor-lora-eval" or (name.startswith("toy-sensor-lora-auto-attempt") and name.endswith("-eval"))

        train_run = next((r for r in runs if _is_train_run(r.info.run_name)), None)
        eval_run = next((r for r in runs if _is_eval_run(r.info.run_name)), None)
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


def _extraction_quality_note() -> str | None:
    """"성공 ≠ 정확"(의사결정_로그 13번)을 실제로 메꾸는 부분 — src/extract/check_quality.py가
    만든 리포트가 있으면 재검토 필요 건수를 그대로 보여준다. 리포트가 없으면 조용히 숨기지
    않고 "아직 점검 안 함"이라고 명시한다."""
    path = _ROOT / "data" / "processed" / "quality_report.jsonl"
    if not path.exists():
        return "품질 점검 미실행 — src/extract/check_quality.py 실행 필요"
    total = review = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            if json.loads(line).get("needs_review"):
                review += 1
    if total == 0:
        return None
    return f"품질 재검토 필요: {review}/{total}건 ({review / total:.0%}) — 반복 줄·한글 비율·깨진 문자 기준" if review else f"품질 점검 {total}건 — 재검토 필요 없음"


_HEALTH_STATE_PATH = _ROOT / ".health_state.json"
_ALERTS_LOG_PATH = _ROOT / "logs" / "alerts.log"
_HEALTH_CHECK_LABELS = {
    "systemd:mlops-web": "웹 서비스",
    "systemd:mlops-mlflow": "MLflow",
    "systemd:mlops-actions-runner": "CI/CD 러너",
    "http:web": "웹 서비스 응답",
    "http:mlflow": "MLflow 응답",
}


def _system_status() -> dict:
    """71번 항목(Alertmanager-lite)이 남긴 두 파일을 그대로 읽어서 보여준다 — 별도
    저장소나 API 없이, scripts/healthcheck_alert.py가 2분마다 갱신하는 상태 파일과
    로그 파일을 그대로 노출한다. 사용자 결정(2026-09-19): push 채널(Slack 등)은
    상용화 단계로 미루고, 지금은 이 화면에 보여주는 정도로 충분하다."""
    checks: list[dict] = []
    if _HEALTH_STATE_PATH.exists():
        try:
            state = json.loads(_HEALTH_STATE_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            state = {}
        checks = [{"label": _HEALTH_CHECK_LABELS.get(k, k), "up": v} for k, v in state.items()]

    recent_alerts: list[str] = []
    if _ALERTS_LOG_PATH.exists():
        lines = [ln for ln in _ALERTS_LOG_PATH.read_text(encoding="utf-8").splitlines() if ln.strip()]
        recent_alerts = list(reversed(lines[-20:]))

    return {"checked": bool(checks), "checks": checks, "recent_alerts": recent_alerts}


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
    elif ft is not None:
        # 연결은 됐지만 이름이 매칭되는 run이 없는 경우 — "연결 안 됨"이라고 하면 실제
        # 원인(연결 vs run 부재)을 오해하게 되므로 구분해서 표시한다.
        ft_stats = ["MLflow 연결됨 — 이름이 일치하는 학습 run을 못 찾음(train/finetune_lora.py 또는 auto_retrain.py 실행 필요)"]
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
            "stats": [
                f"실제 안전문서 처리: {_extraction_success_rate()}", "폴백 사슬 — 실패해도 조용히 넘어가지 않고 이유를 남김",
                *([_extraction_quality_note()] if _extraction_quality_note() else []),
            ],
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
    return templates.TemplateResponse(
        request, "index.html", {"stages": _pipeline_stages(), "status": _system_status()},
    )


@app.get("/education", response_class=HTMLResponse)
def education(request: Request):
    return templates.TemplateResponse(request, "education.html", {"deck_available": _DECK_PDF.exists()})


_MAX_SENSORS = 4
_ALL_EQUIPMENT = ["환풍기", "가스차단기", "소화설비(대기)", "소화설비(방출)"]

# 서버 기본은 이미 로드된 인프로세스 모델을 그대로 써서 빠르다(코어 제한 없음 = 지금
# 서버 그대로). 나머지는 로드맵 7번에서 검증한 cgroup 에뮬레이션(의사결정_로그 32번)을
# 요청 단위로 재사용 — 매번 모델을 새로 불러오는 대신, 실제로 그 코어/메모리 조건에서
# 돌린 진짜 결과를 보여주기 위해서다(가짜로 숫자만 줄이는 게 아니라).
#
# gpu=True 항목은 llama-cpp-python을 CUDA로 재빌드한 뒤(의사결정_로그 62번) 실제
# n_gpu_layers=-1(전체 오프로드)로 돌린다 — GPU 빌드가 안 된 환경이면 lifespan에서
# 이 항목들을 통째로 지워서 애초에 선택 목록에 안 뜨게 한다(거짓으로 "GPU"라고 표시된
# 채 실제로는 CPU로 도는 걸 방지).
EDGE_PROFILES = {
    "server": {"label": "서버 기본 (CPU 전용, 제한 없음)", "cores": None, "mem_gb": None, "gpu": False},
    "server_gpu": {"label": "서버 기본 (GPU, 제한 없음)", "cores": None, "mem_gb": None, "gpu": True},
    "2c32g": {"label": "2코어 / 32GB (GPU 없음)", "cores": 2, "mem_gb": 32, "gpu": False},
    "2c32g_gpu": {"label": "2코어 / 32GB + GPU", "cores": 2, "mem_gb": 32, "gpu": True},
    "2c4g": {"label": "2코어 / 4GB (GPU 없음)", "cores": 2, "mem_gb": 4, "gpu": False},
    "2c4g_gpu": {"label": "2코어 / 4GB + GPU", "cores": 2, "mem_gb": 4, "gpu": True},
    "4c8g": {"label": "4코어 / 8GB (GPU 없음)", "cores": 4, "mem_gb": 8, "gpu": False},
    "4c8g_gpu": {"label": "4코어 / 8GB + GPU", "cores": 4, "mem_gb": 8, "gpu": True},
}
# "서버 기본(제한 없음)"은 이 프로젝트의 개발/운영 서버 자체라 실제 엣지 디바이스로서는
# 비현실적이다(사용자 지적) — 실제 엣지 배포에서 가장 흔한 스펙으로 꼽히는 4코어/8GB급
# (산업용 미니PC·NUC급)을 기본 선택지로 삼는다.
_DEFAULT_EDGE_PROFILE = "4c8g"

_CATEGORY_SUBSTANCES = {cat: [row[0] for row in table] for cat, (table, _action) in CATEGORIES.items()}
_CATEGORY_SUBSTANCES_JSON = json.dumps(_CATEGORY_SUBSTANCES, ensure_ascii=False)


_FEEDBACK_PATH = _ROOT / "data" / "processed" / "feedback.jsonl"


def _default_location() -> str:
    """위치 입력을 없앤 대신 고정 위치를 배정한다 — 데모 편의를 위한 placeholder이지
    실제 위치 정보가 아니다. 활성화된 센서 전부가 같은 공간에 있다고 가정하므로
    (2026-09-19, 복합 센서 시뮬레이션) 슬롯마다 다른 위치를 주던 예전 방식과 달리
    한 요청 안에서는 항상 같은 값 하나를 쓴다."""
    return LOCS[0]


def _value_for_ratio(substance: str, threshold: float, ratio: float) -> float:
    """SEVERITY_RATIO(정상/주의/심각 배율)를 실제 측정값으로 바꾼다. 대부분 물질은 '높을수록
    위험'이라 곱하면 되지만, 산소농도처럼 '낮을수록 위험'한 물질(sensors.LOWER_IS_WORSE)은
    나누는 방향이어야 방향이 맞다 — 물리적으로 불가능한 값이 안 나오게 상한(VALUE_CEILING)도
    같이 적용한다(예: 대기 중 산소는 20.9%를 넘을 수 없음)."""
    if is_lower_is_worse(substance):
        value = threshold / ratio if ratio else threshold
        ceiling = VALUE_CEILING.get(substance)
        if ceiling is not None:
            value = min(value, ceiling)
    else:
        value = threshold * ratio
    return round(value, 2)


def _severity_to_event(category: str, substance: str, severity: str, location: str) -> dict | None:
    lookup = substance_lookup(category, substance)
    if not lookup:
        return None
    unit, threshold = lookup
    ratio = SEVERITY_RATIO.get(severity, 0.6)
    value = _value_for_ratio(substance, threshold, ratio)
    return {
        "category": category, "substance": substance, "value": value, "threshold": threshold,
        "unit": unit, "location": location, "lower_is_worse": is_lower_is_worse(substance),
    }


def _events_from_payload(sensors: list[dict]) -> list[dict]:
    """활성화된 센서 전부가 같은 공간(location)에 있다고 가정한다 — "복합 센서"
    시뮬레이션(2026-09-19, 사용자 요청)의 전제라 여기서 위치를 한 번만 정해서 전체
    센서에 똑같이 준다."""
    location = _default_location()
    events = []
    for s in sensors:
        event = _severity_to_event(s.get("category", ""), s.get("substance", ""), s.get("severity", "정상"), location)
        if event:
            events.append(event)
    return events


def _aggregate_equipment(results: list[dict]) -> list[dict]:
    status = {name: "꺼짐" for name in _ALL_EQUIPMENT}
    for r in results:
        for e in r.get("equipment_status", []):
            status[e["equipment"]] = e["status"]
    return [{"equipment": name, "status": status[name]} for name in _ALL_EQUIPMENT]


def _run_events_inprocess_composite(events: list[dict], llm) -> dict:
    """활성화된 센서 전부를 한 공간으로 보고 LLM이 종합 의견 하나를 내게 한다
    (run_agent_composite, 2026-09-19). 이 함수는 /api/narrate의 in-process 경로에서만
    쓰인다 — control-room은 항상 이벤트 하나짜리 run_agent를 별도로 직접 호출한다."""
    r = run_agent_composite(events, _state["ctx"], llm)
    result = to_dict_composite(r)
    _state["ctx"].notify_log.clear()
    return result


def _pick_inprocess_llm(profile: dict):
    """GPU 프로파일이 선택됐고 GPU 모델이 실제로 로드돼 있으면 그걸, 아니면 항상 있는
    CPU 모델을 쓴다. EDGE_PROFILES에서 gpu=True 항목은 로드 실패 시 아예 지워지므로
    (lifespan 참고) 이 폴백은 이론상만 필요하지만 방어적으로 남겨둔다."""
    if profile.get("gpu") and _state.get("llm_gpu") is not None:
        return _state["llm_gpu"], _llm_gpu_lock
    return _state["llm"], _llm_lock


_gpu_llm_busy = False


def _run_inprocess_composite_locked(events: list[dict], profile: dict) -> dict:
    """threading.Lock 획득 + 추론을 한 덩어리로 executor에 넘기기 위한 래퍼 — lock
    획득 자체도 블로킹이라 이벤트 루프에서 바로 하면 안 된다(/api/narrate 참고)."""
    global _gpu_llm_busy
    llm, lock = _pick_inprocess_llm(profile)
    is_gpu = lock is _llm_gpu_lock
    with lock:
        if is_gpu:
            # AI튜터 쪽 학습/파인튜닝(transformers)이 같은 물리 GPU를 쓰므로, 통합관제가
            # GPU 프로파일로 추론하는 동안엔 학습 시작을 막는다(사용자 질문 계기,
            # 2026-09-19) — 반대 방향(학습 중 GPU 추론 시작)은 edu 쪽 학습락이 이미
            # AI튜터 전체를 막고 있어 이 파일 안에서 한 방향만 추가로 지키면 충분하다.
            _gpu_llm_busy = True
        try:
            return _run_events_inprocess_composite(events, llm)
        finally:
            if is_gpu:
                _gpu_llm_busy = False


def _run_events_emulated(events: list[dict], cores: int, mem_gb: int, gpu: bool, composite: bool = False) -> list[dict] | dict:
    """엣지 스펙 에뮬레이션 — 별도 프로세스를 systemd-run(cgroup)+taskset으로 감싸서
    실제로 그 코어 수·메모리로 제한된 조건에서 돌린다(로드맵 7번, 의사결정_로그 32번과
    동일한 방법). 매번 모델을 새로 불러와서 인프로세스보다 느리지만, 숫자가 진짜다.
    gpu=True면 AGENT_GPU 환경변수로 서브프로세스(agent/run_cli.py)에 전달해서 그
    안에서 n_gpu_layers=-1로 새로 모델을 띄우게 한다 — taskset의 CPU 코어 제한은
    GPU/PCIe 접근과 무관해서 같이 걸어도 문제없다.

    composite=True면 AGENT_COMPOSITE=1을 넘겨서 agent/run_cli.py가 이벤트 전체를 하나의
    종합 결과(dict)로 처리·반환하게 한다(2026-09-19). 기본값 False는 control-room처럼
    이벤트 하나짜리 리스트를 그대로 기대하는 기존 호출부를 그대로 보존하기 위함."""
    core_list = ",".join(str(i) for i in range(cores))
    python_bin = _ROOT / ".venv" / "bin" / "python"
    # 스코프 이름을 직접 지정해둔다 — 타임아웃 시 이 이름으로 확실히 정지시키기 위해
    # 필요하다(아래 except 참고).
    scope_unit = f"mlops-edge-emu-{uuid.uuid4().hex[:8]}.scope"

    with tempfile.TemporaryDirectory() as tmp:
        events_path = Path(tmp) / "events.json"
        output_path = Path(tmp) / "output.json"
        events_path.write_text(json.dumps(events, ensure_ascii=False), encoding="utf-8")

        cmd = [
            "systemd-run", "--user", "--scope", "--quiet", f"--unit={scope_unit}",
            "-p", f"CPUQuota={cores * 100}%",
            "-p", f"MemoryMax={mem_gb}G",
            "-p", "MemorySwapMax=0",
            "--", "taskset", "-c", core_list,
            str(python_bin), "-m", "agent.run_cli", str(events_path), str(output_path),
        ]
        try:
            proc = subprocess.run(
                cmd, cwd=str(_ROOT), timeout=180, capture_output=True, text=True,
                env={
                    **os.environ, "PYTHONPATH": str(_ROOT / "src"), "AGENT_THREADS": str(cores),
                    "AGENT_GPU": "1" if gpu else "0", "AGENT_COMPOSITE": "1" if composite else "0",
                },
            )
        except subprocess.TimeoutExpired:
            # subprocess.run의 기본 타임아웃 처리는 우리가 직접 띄운 systemd-run
            # "실행기" 프로세스만 죽인다 — --scope로 만든 실제 스코프(무거운 연산이
            # 도는 곳)는 별개로 계속 돌아서, 죽이지 않으면 좀비로 남아 taskset이
            # 고정한 CPU 코어를 계속 점유한다. 2026-09-19 실제로 겪음: 타임아웃 난
            # 요청의 좀비가 남아서 다음 요청들까지 같은 코어를 나눠 쓰다 연쇄
            # 타임아웃으로 번짐. 스코프 이름을 알고 있으니 명시적으로 정지시킨다.
            subprocess.run(["systemctl", "--user", "stop", scope_unit], capture_output=True)
            raise RuntimeError(f"엣지 에뮬레이션 타임아웃(180초 초과) — 좀비 프로세스는 정리했습니다")

        if proc.returncode != 0 or not output_path.exists():
            raise RuntimeError(f"엣지 에뮬레이션 실행 실패(exit {proc.returncode}): {proc.stderr[-1500:]}")
        return json.loads(output_path.read_text(encoding="utf-8"))


@app.get("/simulate")
def simulate_form():
    """/simulate를 통합관제 페이지로 합쳤다(2026-09-19, 사용자 요청) — 예전 링크·북마크가
    깨지지 않게 리다이렉트만 남겨둔다."""
    return RedirectResponse(url="/control-room")


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


_simulation_history: list[dict] = []
_simulation_in_progress = False
_MAX_SIMULATION_HISTORY = 20
# 자동(실시간 통합관제 루프)과 수동 시뮬레이션이 서로의 실행 여부를 확인하는 상호
# 배제 플래그 — 2026-09-19 사용자 요청("자동/수동 둘 중에 하나만 동작"). 하나가 cgroup
# 코어를 점유 중일 때 다른 하나가 같이 돌면 서로 느려져 타임아웃까지 발생했다.
_auto_tick_in_progress = False


@app.get("/api/simulate/history")
def simulate_history():
    return JSONResponse({"history": _simulation_history})


@app.post("/api/narrate")
async def api_narrate(request: Request):
    """② LLM 처리 단계 — 실제 run_agent()(또는 엣지 에뮬레이션)를 돌려서 알림 문구·결정
    추적까지 완성한다. 시간이 걸리는 부분이라 프론트가 이 호출 동안 스톱워치를 보여준다.

    **동시 실행 금지(2026-09-19, 사용자 요청)**: 엣지 에뮬레이션은 taskset으로 특정
    CPU 코어를 고정해서 쓰는데, 두 시뮬레이션이 동시에 돌면 같은 코어를 나눠 쓰면서
    서로 느려진다(83번 항목에서 실제로 겪은 좀비 프로세스 문제와 같은 종류의 자원
    경합). 그래서 서버가 전역으로 "지금 하나 돌고 있으면 새 요청은 거절"한다 —
    프론트에서 버튼을 비활성화하는 것만으로는 다른 탭/사용자가 동시에 누르는 걸
    못 막아서, 최종 방어선은 서버에 둔다."""
    global _simulation_in_progress

    if _simulation_in_progress:
        return JSONResponse({"error": "이미 다른 시뮬레이션이 진행 중입니다. 완료 후 다시 시도해주세요."}, status_code=409)
    if _auto_tick_in_progress:
        # 자동(실시간) 루프가 지금 막 LLM 처리 중이면 cgroup 코어 경합을 피하려고
        # 수동 실행을 거절한다(2026-09-19 사용자 요청) — 자동 루프는 12초 간격이라
        # 잠깐 뒤 재시도하면 대부분 바로 통과한다.
        return JSONResponse({"error": "자동 시뮬레이션이 처리 중입니다. 잠시 후 다시 시도해주세요."}, status_code=409)

    payload = await request.json()
    events = _events_from_payload(payload.get("sensors", []))
    profile_key = payload.get("edge_profile", _DEFAULT_EDGE_PROFILE)
    profile = EDGE_PROFILES.get(profile_key, EDGE_PROFILES[_DEFAULT_EDGE_PROFILE])
    METRIC_NARRATE_REQUESTS.labels(edge_profile=profile_key).inc()

    if not events:
        return JSONResponse({"error": "최소 1개 센서를 활성화해주세요."}, status_code=400)

    _simulation_in_progress = True
    start = time.perf_counter()
    loop = asyncio.get_event_loop()
    try:
        if profile["cores"] is None:
            # threading.Lock도 동기 블로킹이라, lock 획득까지 통째로 executor 안에서
            # 해야 한다 — 여기서 바로 `with lock:` 하면 락 대기 자체가 이벤트 루프를
            # 막아버린다(아래 subprocess 호출과 같은 이유).
            composite_result = await loop.run_in_executor(None, _run_inprocess_composite_locked, events, profile)
        else:
            # run_in_executor로 감싸지 않으면 이 subprocess.run() 호출(최대 180초)이
            # 이벤트 루프를 통째로 막는다 — 그동안 control-room·edu-admin 등 다른
            # 페이지도 전부 응답을 못 하게 된다. 2026-09-19 동시 실행 방지 기능을
            # 테스트하다 실제로 겪음: 두 번째 요청이 409로 거절되는 게 아니라 첫 번째가
            # 끝날 때까지 그냥 먹통으로 대기하고 있었다.
            composite_result = await loop.run_in_executor(
                None, _run_events_emulated, events, profile["cores"], profile["mem_gb"], profile.get("gpu", False), True,
            )
    except Exception as exc:  # noqa: BLE001 — 데모 화면에 원인을 그대로 보여주기 위함
        METRIC_NARRATE_ERRORS.inc()
        return JSONResponse({"error": str(exc)}, status_code=500)
    finally:
        _simulation_in_progress = False
    elapsed = round(time.perf_counter() - start, 2)
    METRIC_NARRATE_LATENCY.labels(edge_profile=profile_key).observe(elapsed)
    for e in composite_result.get("equipment_status", []):
        METRIC_EQUIPMENT_ACTUATED.labels(equipment=e["equipment"], status=e["status"]).inc()

    # 실행 기록으로 남긴다(사용자 요청) — 여러 번 돌려본 결과를 화면에서 계속 비교해볼
    # 수 있어야 하는데, 예전엔 매번 결과 패널 하나를 덮어써서 직전 결과가 사라졌다.
    # 서버 재시작(배포)까지 살아남을 필요는 없다고 판단해 파일이 아니라 메모리에만
    # 쌓는다 — control-room의 상태와 같은 수준의 "런타임 동안만 유지" 성격.
    record = {
        "id": uuid.uuid4().hex[:8],
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "edge_profile": profile_key,
        "edge_label": profile["label"],
        "elapsed": elapsed,
        "composite": composite_result,
        "equipment": _aggregate_equipment([composite_result]),
    }
    _simulation_history.insert(0, record)
    del _simulation_history[_MAX_SIMULATION_HISTORY:]

    return JSONResponse(record)


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
    "지하 기계실": {"category": "가스", "substances": ["수소", "산소(O2)"]},
    "실험동 가스저장실": {"category": "가스", "substances": ["수소", "산소(O2)"]},
    "2층 사무실": {"category": "공기질", "substances": ["CO2", "TVOC"]},
    "체육관": {"category": "공기질", "substances": ["CO2", "습도"]},
}
_CONTROL_ROOM_INTERVAL_SECONDS = 12

_control_room_state: dict[str, dict] = {}
# /simulate와 달리 통합관제는 여러 사람이 같이 보는 "관제실 화면 하나"라는 컨셉이라,
# 요청마다 프로파일을 넘기는 게 아니라 서버 쪽 전역 설정 하나로 둔다(이 화면을 보는
# 모두가 같은 조건을 본다) — EDGE_PROFILES는 /simulate와 동일한 것을 재사용.
_control_room_edge_profile = _DEFAULT_EDGE_PROFILE
_control_room_auto_enabled = True


def _now_hms() -> str:
    return time.strftime("%H:%M:%S")


def _random_event(site: str, substance: str) -> dict:
    category = _CONTROL_ROOM_SITES[site]["category"]
    unit, threshold = substance_lookup(category, substance)
    # 정상이 더 자주 나오게 가중치를 둠 — 알림이 쉴 새 없이 뜨면 "위험 신호"의 의미가
    # 옅어져서 오히려 데모로서 설득력이 떨어진다.
    severity = random.choices(["정상", "주의", "심각"], weights=[55, 30, 15])[0]
    ratio = SEVERITY_RATIO.get(severity, 0.6)
    value = _value_for_ratio(substance, threshold, ratio)
    return {
        "category": category, "substance": substance, "value": value, "threshold": threshold,
        "unit": unit, "location": site, "lower_is_worse": is_lower_is_worse(substance),
    }


def _init_control_room() -> None:
    """서버 기동 시 각 현장의 모든 센서를 정상 상태 기본값으로 채워둔다 — 첫 폴링 전에도
    화면이 비어있지 않게 하기 위함. 알림(alert)은 처음엔 당연히 없음(None)."""
    for site, spec in _CONTROL_ROOM_SITES.items():
        sensors = {}
        for substance in spec["substances"]:
            unit, threshold = substance_lookup(spec["category"], substance)
            sensors[substance] = {
                "value": _value_for_ratio(substance, threshold, SEVERITY_RATIO["정상"]), "threshold": threshold,
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
    global _gpu_llm_busy
    profile = EDGE_PROFILES.get(_control_room_edge_profile, EDGE_PROFILES[_DEFAULT_EDGE_PROFILE])
    if profile["cores"] is None:
        llm, lock = _pick_inprocess_llm(profile)
        is_gpu = lock is _llm_gpu_lock
        with lock:
            if is_gpu:
                _gpu_llm_busy = True
            try:
                result = run_agent(event, _state["ctx"], llm)
                _state["ctx"].notify_log.clear()
                return to_dict(result)
            finally:
                if is_gpu:
                    _gpu_llm_busy = False
    return _run_events_emulated([event], profile["cores"], profile["mem_gb"], profile.get("gpu", False))[0]


def _set_site_alert_pending(site: str, substance: str, severity: str, equipment_status: list[dict], sensor_changed_at: str) -> None:
    """LLM 응답을 기다리는 동안 먼저 보여줄 상태(사용자 요청, 2026-09-19 — /simulate처럼
    단계별로 보이게). preview()의 equipment_status는 실제 tools.actuate_equipment를 부르지
    않는 순수 미리보기라 부작용이 없다(/simulate의 "①즉시 반응"과 같은 패턴) — 실제 실행
    기록은 이후 _set_site_alert가 run_agent 결과로 덮어쓴다."""
    _control_room_state[site]["alert"] = {
        "substance": substance, "severity": severity, "narrative": None,
        "equipment": equipment_status, "logged_at": None,
        "sensor_changed_at": sensor_changed_at, "elapsed": None,
        "edge_label": EDGE_PROFILES.get(_control_room_edge_profile, EDGE_PROFILES[_DEFAULT_EDGE_PROFILE])["label"],
        "pending": True,
    }


def _set_site_alert(site: str, substance: str, severity: str, result: dict, elapsed: float, sensor_changed_at: str) -> None:
    """단계별 타임스탬프(사용자 요청, 의사결정_로그 61번)를 전부 남긴다 — 센서 변경 시점은
    여기서 직접 넘겨받고, 장비 조치 시점은 각 equipment_status 항목이 이미 갖고 있는
    ISO 'at' 필드에서, LLM 완성 시점은 지금(logged_at)으로 기록한다."""
    _control_room_state[site]["alert"] = {
        "substance": substance, "severity": severity, "narrative": result["narrative"],
        "equipment": result.get("equipment_status", []), "logged_at": _now_hms(),
        "sensor_changed_at": sensor_changed_at,
        "elapsed": elapsed, "edge_label": EDGE_PROFILES.get(_control_room_edge_profile, EDGE_PROFILES[_DEFAULT_EDGE_PROFILE])["label"],
        "pending": False,
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
        if not _control_room_auto_enabled:
            # 사용자가 명시적으로 중지시켰다(2026-09-19 요청) — 상호배제 플래그만으로는
            # 12초 틱이 계속 재시도해서 수동 시뮬레이션이 반복적으로 거절될 수 있어,
            # 아예 이번 루프를 완전히 쉬게 하는 명시적 on/off 스위치를 추가했다.
            continue
        try:
            site = random.choice(list(_CONTROL_ROOM_SITES))
            substance = random.choice(_CONTROL_ROOM_SITES[site]["substances"])
            event = _random_event(site, substance)
            preview_result = preview(event)
            severity = preview_result["judgement"]["severity"]
            sensor_changed_at = _now_hms()
            _update_sensor(site, substance, event, preview_result)
            METRIC_CONTROL_ROOM_EVENTS.labels(site=site, severity=severity).inc()

            if severity != "정상":
                # 수동 시뮬레이션이 지금 CPU 코어를 점유 중이면 이번 틱은 건너뛴다
                # (사용자 요청, 2026-09-19) — 자동 루프와 수동 시뮬레이션이 같은 cgroup
                # 코어를 동시에 쓰면 서로 느려져 수동 쪽이 타임아웃까지 걸렸다. 12초
                # 뒤 다음 틱에서 다시 시도하면 되므로 이번 틱은 조용히 넘어간다.
                if _simulation_in_progress:
                    continue
                # 규칙(장비 조치)은 즉시 보여주고, LLM 문구는 나중에 채운다(사용자 요청,
                # /simulate의 단계별 표시와 동일한 원칙 — "판정은 즉시, LLM은 나중"이라는
                # 이 프로젝트 전체의 설계를 카드 화면에서도 실제로 보이게 함).
                _set_site_alert_pending(site, substance, severity, preview_result["equipment_status"], sensor_changed_at)
                global _auto_tick_in_progress
                _auto_tick_in_progress = True
                try:
                    start = time.perf_counter()
                    result = await loop.run_in_executor(None, _run_control_room_narrative, event)
                    elapsed = round(time.perf_counter() - start, 2)
                finally:
                    _auto_tick_in_progress = False
                _set_site_alert(site, substance, severity, result, elapsed, sensor_changed_at)
            else:
                _clear_site_alert_if_owner(site, substance)
        except Exception as exc:  # noqa: BLE001 — 백그라운드 루프는 절대 죽으면 안 됨
            print(f"[control-room] tick 실패: {exc}")


@app.get("/control-room", response_class=HTMLResponse)
def control_room(request: Request):
    return templates.TemplateResponse(
        request, "control_room.html",
        {
            "sites": list(_CONTROL_ROOM_SITES.keys()), "edge_profiles": EDGE_PROFILES,
            "default_edge_profile": _control_room_edge_profile,
            # 아래 셋은 /simulate 페이지를 여기로 통합하면서(2026-09-19, 사용자 요청)
            # 같이 필요해진 것 — 수동 시뮬레이션 섹션의 센서 입력 카드용.
            "sensors_range": range(1, _MAX_SENSORS + 1), "categories": _CATEGORY_SUBSTANCES,
            "categories_json": _CATEGORY_SUBSTANCES_JSON,
        },
    )


@app.get("/api/control-room/status")
def control_room_status():
    sites = [{"name": name, **_control_room_state.get(name, {})} for name in _CONTROL_ROOM_SITES]
    return JSONResponse({"sites": sites, "edge_profile": _control_room_edge_profile, "auto_enabled": _control_room_auto_enabled})


@app.post("/api/control-room/auto-toggle")
async def toggle_control_room_auto(request: Request):
    """자동(실시간) 시뮬레이션 중지/재개 — 사용자 요청(2026-09-19): 상호배제 플래그만
    으로는 12초 틱이 계속 재시도해서 수동 시뮬레이션이 반복 거절될 수 있어, 아예 루프를
    쉬게 하는 명시적 스위치를 추가했다. 관제실 화면과 같은 "여러 사람이 보는 하나의
    화면" 성격이라 전역으로 적용한다(edge-profile과 동일한 설계)."""
    global _control_room_auto_enabled
    payload = await request.json()
    _control_room_auto_enabled = bool(payload.get("enabled", True))
    return JSONResponse({"status": "ok", "auto_enabled": _control_room_auto_enabled})


@app.post("/api/control-room/edge-profile")
async def set_control_room_edge_profile(request: Request):
    """관제실 화면은 여러 사람이 같이 보는 하나의 화면이라, 여기서 바꾼 엣지 프로파일은
    다음 백그라운드 틱부터 전역으로 적용된다(요청 보낸 사람만 바뀌는 게 아님)."""
    global _control_room_edge_profile
    payload = await request.json()
    key = payload.get("edge_profile", _DEFAULT_EDGE_PROFILE)
    if key not in EDGE_PROFILES:
        return JSONResponse({"error": "알 수 없는 엣지 프로파일"}, status_code=400)
    _control_room_edge_profile = key
    return JSONResponse({"status": "ok", "edge_profile": key, "label": EDGE_PROFILES[key]["label"]})


# ── 스마트교육 운영 콘솔 (2026-09-19) ──────────────────────────────────────
# 수집 현황 표시 + 소스 선택 학습 트리거 + "학생 체험"(RAG vs 파인튜닝 비교).
# 학습/파인튜닝-단독 답변은 둘 다 무거운 별도 프로세스로 돌린다(위 control-room과
# 달리 GPU에 transformers 모델을 새로 올려야 해서 인프로세스 llama_cpp와 자원을
# 다툰다 — finetune_lora.py에서 실제로 겪은 GPU 충돌과 같은 이유).

_EDU_JOB_STATUS_PATH = _ROOT / "experiments" / "edu-social-lora" / "job_status.json"
_EDU_PYTHON_BIN = _ROOT / ".venv" / "bin" / "python"


def _edu_job_status() -> dict:
    if not _EDU_JOB_STATUS_PATH.exists():
        return {"status": "idle"}
    try:
        return json.loads(_EDU_JOB_STATUS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"status": "idle"}


_DEFAULT_SCHOOL_LEVEL = "중학교"
_DEFAULT_SUBJECT = "사회"


def _registry_school_levels() -> list[str]:
    levels: list[str] = []
    for cfg in collect_registry.SOURCES:
        for lv in cfg.school_levels:
            if lv not in levels:
                levels.append(lv)
    return levels


def _registry_subjects() -> list[str]:
    subjects: list[str] = []
    for cfg in collect_registry.SOURCES:
        for subj in cfg.subjects:
            if subj not in subjects:
                subjects.append(subj)
    return subjects


@app.get("/edu-admin", response_class=HTMLResponse)
def edu_admin(request: Request, school_level: str = _DEFAULT_SCHOOL_LEVEL, subject: str = _DEFAULT_SUBJECT):
    # 학교급·과목 선택(사용자 요청, 2026-09-19) — registry에 등록된 값만 옵션으로
    # 보여준다(존재하지 않는 조합을 지어내지 않기 위함). 지금은 중학교/사회만 실제
    # 수집된 데이터가 있고 나머지는 "등록은 됐지만 소스 미구현"으로 정직하게 0건 표시.
    matched = collect_registry.sources_for(school_level, subject)
    matched_ids = {cfg.source_id for cfg in matched}

    summary_by_source = {s["source_id"]: s for s in collection_summary()}
    sources = []
    for config in matched:
        collected = summary_by_source.get(config.source_id)
        sources.append({
            "source_id": config.source_id,
            "name": config.name,
            "implemented": config.implemented,
            "requires_auth": config.requires_auth,
            "notes": config.notes,
            "license_type": config.license.license_type,
            "allows_modification": collected["allows_modification"] if collected else config.license.allows_modification,
            "item_count": collected["item_count"] if collected else 0,
            "last_collected_at": collected["last_collected_at"] if collected else None,
        })

    rag_available = _state.get("edu_index") is not None and any(
        summary_by_source.get(sid, {}).get("item_count", 0) > 0 for sid in matched_ids
    )

    return templates.TemplateResponse(
        request, "edu_admin.html",
        {
            "sources": sources,
            "school_levels": _registry_school_levels(),
            "subjects": _registry_subjects(),
            "selected_school_level": school_level,
            "selected_subject": subject,
            "rag_available": rag_available,
            "finetuned_available": (_ROOT / "experiments" / "edu-social-lora" / "final").exists(),
            "claude_available": bool(os.environ.get("ANTHROPIC_API_KEY")),
        },
    )


@app.get("/api/edu-admin/status")
def edu_admin_status():
    return JSONResponse(_edu_job_status())


@app.get("/ai-debate", response_class=HTMLResponse)
def ai_debate(request: Request):
    return templates.TemplateResponse(
        request, "ai_debate.html",
        {
            "school_levels": _registry_school_levels(),
            "subjects": _registry_subjects(),
            "selected_school_level": _DEFAULT_SCHOOL_LEVEL,
            "selected_subject": _DEFAULT_SUBJECT,
            "rag_available": _state.get("edu_index") is not None,
        },
    )


@app.get("/bidradar", response_class=HTMLResponse)
def bidradar_page(request: Request):
    return templates.TemplateResponse(request, "bidradar.html", {})


@app.get("/api/bidradar/stats")
def bidradar_stats():
    """실제 BidRadar 호출 기록 집계 — 시뮬레이션이 아니라 /v1/* 엔드포인트가 실제로
    받은 요청만 반영한다(사용자 요청, 2026-09-20). 서버 재시작 전까지만 유지되는
    런타임 메모리 기록이라(control-room 패턴과 동일), 배포 후엔 초기화된다."""
    endpoints = ["classify-doc", "classify-topic", "extract-requirements"]
    per_endpoint = {}
    for ep in endpoints:
        calls = [c for c in _bidradar_call_log if c["endpoint"] == ep]
        success = [c for c in calls if c["success"]]
        per_endpoint[ep] = {
            "total": len(calls),
            "success": len(success),
            "failed": len(calls) - len(success),
            "avg_latency_ms": round(sum(c["latency_ms"] for c in success) / len(success)) if success else None,
            "total_tokens_in": sum(c["tokens_in"] for c in calls),
            "total_tokens_out": sum(c["tokens_out"] for c in calls),
        }
    return JSONResponse({
        "endpoints": per_endpoint,
        "recent": _bidradar_call_log[:50],
        "total_calls": len(_bidradar_call_log),
    })


_LICENSE_OVERRIDE_LOG = _ROOT / "logs" / "license_overrides.log"


@app.post("/api/edu-admin/train")
async def edu_admin_train(request: Request):
    payload = await request.json()
    selected = payload.get("sources", [])
    license_override_ack = bool(payload.get("license_override_ack"))
    if not selected:
        return JSONResponse({"error": "소스를 하나 이상 선택하세요"}, status_code=400)

    summary_by_source = {s["source_id"]: s for s in collection_summary()}
    restricted = [
        sid for sid in selected
        if not summary_by_source.get(sid, {}).get("allows_modification")
    ]
    if restricted and not license_override_ack:
        # 라이선스 위반 방지는 서버가 최종 책임진다 — 체크박스는 클라이언트에서
        # 얼마든지 우회 가능하기 때문에, "동의했다는 플래그"까지 서버가 재확인한다.
        return JSONResponse(
            {"error": f"파인튜닝에 쓸 수 없는 소스(라이선스상 변경 금지): {', '.join(restricted)} — 동의 없이는 진행할 수 없습니다"},
            status_code=400,
        )
    if restricted:
        # 사용자 요청으로 2026-09-19부터 명시적 동의하에 허용(의사결정_로그 참고) —
        # 라이선스 위반 소지가 있는 선택이라 별도 로그에 남겨서 나중에 추적 가능하게 한다.
        _LICENSE_OVERRIDE_LOG.parent.mkdir(parents=True, exist_ok=True)
        with _LICENSE_OVERRIDE_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "logged_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "restricted_sources": restricted, "all_sources": selected,
            }, ensure_ascii=False) + "\n")

    current = _edu_job_status()
    if current.get("status") == "running":
        return JSONResponse({"error": "이미 학습이 진행 중입니다"}, status_code=409)
    if _gpu_llm_busy:
        # 통합관제가 GPU 프로파일로 추론 중이면 학습을 시작하지 않는다(사용자 질문
        # 계기, 2026-09-19) — 둘 다 같은 물리 GPU를 쓰는 transformers/llama.cpp
        # 프로세스라 동시에 돌면 자원을 다툰다.
        return JSONResponse({"error": "통합관제가 GPU로 추론 중입니다. 잠시 후 다시 시도해주세요."}, status_code=409)

    # 그냥 subprocess.Popen만 쓰면 systemd가 mlops-web 서비스를 cgroup째로 관리하기
    # 때문에, 배포 중 `systemctl restart mlops-web`이 뜨면 이 자식 프로세스도 같이
    # 죽는다(2026-09-19 실제로 겪음 — 학습 도중 다른 기능을 배포했더니 학습이 조용히
    # 죽어있었음). systemd-run --scope로 별도 스코프에 띄우면 mlops-web과 생명주기가
    # 분리돼 배포가 학습을 방해하지 않는다(엣지 에뮬레이션에 쓰던 패턴과 동일).
    train_cmd = [
        "systemd-run", "--user", "--scope", "--quiet", "--",
        str(_EDU_PYTHON_BIN), "-m", "train.run_edu_training_job", "--sources", ",".join(selected),
    ]
    if restricted:
        # 동의(license_override_ack)를 실제 데이터 생성 단계까지 전달한다(2026-09-19) —
        # make_edu_dataset.py가 allows_modification을 항목 단위로 다시 검사하는 별도
        # 안전장치를 갖고 있어서(라이선스 강제 지점, 79번), 이 플래그 없이는 웹에서
        # 동의해도 실제로는 계속 조용히 제외되고 있었다(사용자가 실측으로 발견).
        train_cmd.append("--allow-restricted")

    subprocess.Popen(
        train_cmd,
        cwd=str(_ROOT),
        env={**os.environ, "PYTHONPATH": str(_ROOT / "src")},
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return JSONResponse({"status": "started", "sources": selected})


@app.post("/api/edu-admin/ask")
async def edu_admin_ask(request: Request):
    """학생 체험 — RAG 모드는 이미 로드된 인프로세스 리소스로 즉시 답하고, 파인튜닝
    비교 모드는 별도 프로세스(ask_edu_adapter.py)를 띄운다(느림, 수십 초)."""
    payload = await request.json()
    question = (payload.get("question") or "").strip()
    mode = payload.get("mode", "rag")
    school_level = payload.get("school_level", _DEFAULT_SCHOOL_LEVEL)
    subject = payload.get("subject", _DEFAULT_SUBJECT)
    if not question:
        return JSONResponse({"error": "질문을 입력하세요"}, status_code=400)

    # 학습 중에는 AI 튜터를 막는다(사용자 요청, 2026-09-19) — 파인튜닝 실행은 GPU를
    # 학습 프로세스와 다투고, RAG/Claude도 지금 학습 중인 데이터로 답하면 "학습 전
    # 상태인지 후 상태인지" 혼란스러워서 전체를 잠근다.
    if _edu_job_status().get("status") == "running":
        return JSONResponse({"error": "학습이 진행 중입니다. 완료 후 다시 시도해주세요."}, status_code=409)

    loop = asyncio.get_event_loop()

    if mode == "rag":
        if _state.get("edu_index") is None:
            return JSONResponse({"error": "RAG 인덱스가 없습니다 — rag/build_edu_index.py 먼저 실행 필요"}, status_code=400)
        allowed_source_ids = {cfg.source_id for cfg in collect_registry.sources_for(school_level, subject)}

        def _run_rag():
            # _state["llm"]은 산업안전 통합관제(_control_room_loop)와 공유하는 동일한
            # in-process Llama 인스턴스다 — 락 없이 부르면 백그라운드 루프의 12초 틱과
            # 동시에 같은 llama.cpp 컨텍스트에서 create_chat_completion이 겹쳐 호출될 수
            # 있고, 실제로 이게 SIGSEGV로 서버를 죽인 원인이었다(2026-09-19). 통합관제
            # 쪽은 이미 _llm_lock으로 직렬화하고 있었는데, 이 경로에만 빠져 있었다.
            with _llm_lock:
                return query_edu.answer(
                    question, _state["edu_embed_model"], _state["edu_index"],
                    _state["edu_meta"], _state["edu_bm25"], _state["llm"],
                    allowed_source_ids=allowed_source_ids,
                )

        start = time.perf_counter()
        result = await loop.run_in_executor(None, _run_rag)
        elapsed = round(time.perf_counter() - start, 2)
        return JSONResponse({"mode": "rag", "elapsed": elapsed, **result})

    if mode == "claude":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return JSONResponse({"error": "ANTHROPIC_API_KEY가 서버에 설정되지 않았습니다"}, status_code=400)

        def _call_claude() -> dict:
            # RAG·근거 없이 Claude 자체 지식으로만 답하게 한다 — "우리 파이프라인(RAG/
            # 파인튜닝) 없이 그냥 강력한 범용 모델에 물어보면 어떤가"를 비교하기 위한
            # 대조군이라, 일부러 근거 자료를 안 준다.
            body = json.dumps({
                "model": "claude-sonnet-5",
                # 화면에서 답변을 textContent로 그대로 넣기 때문에(마크다운 렌더러 없음),
                # 이모지·제목(#)·표 같은 서식을 쓰면 기호가 그대로 깨져 보인다(2026-09-19
                # 사용자 지적) — 평문 문장만 쓰도록 명시. 서식에 토큰을 안 쓰게 되면서
                # max_tokens도 400→600으로 올림(비교 질문에서 문장 중간에 잘리는 문제
                # 동시 발견·수정, 2026-09-19).
                "max_tokens": 600,
                "system": "당신은 중학교 사회 선생님입니다. 학생 질문에 학생 눈높이로 친절하게 답합니다. 이모지나 마크다운 서식(#, *, - 목록, 표 등) 없이 평범한 문장으로만 답하세요.",
                "messages": [{"role": "user", "content": question}],
            }).encode("utf-8")
            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages", data=body,
                headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())
            # content[0]이 무조건 text 블록이 아니다 — claude-sonnet-5는 기본적으로
            # thinking 블록을 먼저 반환하고 그 뒤에 text 블록을 준다(실제로 겪은
            # KeyError('text')). type으로 걸러서 text 블록만 이어붙인다.
            text_blocks = [b["text"] for b in result.get("content", []) if b.get("type") == "text"]
            if not text_blocks:
                raise RuntimeError(f"응답에 텍스트 블록이 없음: {json.dumps(result, ensure_ascii=False)[:300]}")
            usage = result.get("usage", {})
            return {
                "answer": "\n".join(text_blocks),
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
            }

        try:
            start = time.perf_counter()
            call_result = await loop.run_in_executor(None, _call_claude)
            elapsed = round(time.perf_counter() - start, 2)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            return JSONResponse({"error": f"Claude API 호출 실패({exc.code}): {detail}"}, status_code=502)
        except Exception as exc:  # noqa: BLE001 — 데모 화면에 원인을 그대로 보여주기 위함
            return JSONResponse({"error": f"Claude API 호출 실패: {exc}"}, status_code=502)

        # 2026-09-19 확인한 claude-sonnet-5 공식 단가(입력 $2/1M, 출력 $10/1M) — 사용자가
        # "비용이 발생하니 눈에 보이게 하자"고 요청한 항목이라, 토큰 수뿐 아니라 실제
        # 비용 환산까지 같이 보여준다. 원화 표시는 사용자 요청(2026-09-19)으로 환율
        # 1,400원 고정 — 실시간 환율 API를 새로 붙이는 것보단, "대략 얼마인지 감"만
        # 잡으면 되는 데모 용도에 고정값이 더 단순하고 충분하다고 판단.
        input_tokens = call_result["input_tokens"]
        output_tokens = call_result["output_tokens"]
        cost_usd = input_tokens / 1_000_000 * 2.0 + output_tokens / 1_000_000 * 10.0
        cost_krw = round(cost_usd * 1400)
        return JSONResponse({
            "mode": "claude", "answer": call_result["answer"], "sources": [], "elapsed": elapsed,
            "input_tokens": input_tokens, "output_tokens": output_tokens, "cost_krw": cost_krw,
        })

    if mode == "finetuned":
        adapter_dir = _ROOT / "experiments" / "edu-social-lora" / "final"
        if not adapter_dir.exists():
            return JSONResponse({"error": "파인튜닝 어댑터가 없습니다 — 먼저 학습을 실행하세요"}, status_code=400)
        if _gpu_llm_busy:
            # train과 동일한 이유 — 이 서브프로세스도 transformers로 같은 물리 GPU에
            # 모델을 올린다(2026-09-19).
            return JSONResponse({"error": "통합관제가 GPU로 추론 중입니다. 잠시 후 다시 시도해주세요."}, status_code=409)

        def _run_subprocess():
            return subprocess.run(
                [str(_EDU_PYTHON_BIN), "-m", "train.ask_edu_adapter", question],
                cwd=str(_ROOT), env={**os.environ, "PYTHONPATH": str(_ROOT / "src")},
                capture_output=True, text=True, timeout=120,
            )

        start = time.perf_counter()
        proc = await loop.run_in_executor(None, _run_subprocess)
        elapsed = round(time.perf_counter() - start, 2)
        if proc.returncode != 0:
            return JSONResponse({"error": f"어댑터 실행 실패: {proc.stderr[-800:]}"}, status_code=500)
        try:
            parsed = json.loads(proc.stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            return JSONResponse({"error": "어댑터 응답 파싱 실패"}, status_code=500)
        return JSONResponse({"mode": "finetuned", "answer": parsed["answer"], "sources": [], "elapsed": elapsed})

    return JSONResponse({"error": "알 수 없는 모드"}, status_code=400)


_QUIZ_SYSTEM_PROMPT = (
    "당신은 친근한 중학교 사회 선생님입니다. 아래 [참고 자료]에 실제로 쓰여 있는 문장 하나를 그대로 "
    "활용해 객관식 퀴즈 1문제를 만드세요.\n"
    "반드시 지켜야 할 규칙:\n"
    "1. [참고 자료]에 등장하지 않는 지역·나라·수치는 절대 언급하지 마세요(자료에 없는 대상과 "
    "비교하는 문제를 만들지 마세요).\n"
    "2. 정답과 오답 선택지 모두 [참고 자료]에 나온 표현을 최대한 그대로 사용하세요.\n"
    "3. question 필드는 학생에게 말하듯 친근한 구어체로 쓰세요(예: '~할까요?', '~인 거 아시나요?' 같은 "
    "말투). choices와 explanation은 자료 표현을 그대로 유지하세요.\n"
    "4. 설명 없이 반드시 아래 JSON 형식으로만 답하세요:\n"
    '{"question": "...", "choices": ["...", "...", "...", "..."], "answer_index": 0, "explanation": "..."}'
)


def _quiz_choice_grounded(choice_text: str, context: str) -> bool:
    """정답 선택지가 실제로 [참고 자료]에서 나온 표현인지 코드로 한 번 더 확인한다.
    프롬프트로 "자료에 없는 걸 지어내지 마라"고 시켜도 4B급 소형 모델이 완전히
    지키진 못한다(2026-09-19 실측 — 지시 전엔 "서울 90%, 부산 70%" 식으로 자료에
    없는 비교를 만들어냄). 정답 문장의 명사·숫자 토큰 중 상당수가 실제로 context에
    그대로 있어야 '근거 있음'으로 인정 — 완벽하진 않지만(토큰이 다른 조합으로 재구성돼도
    통과할 수 있음) 아예 새로 지어낸 내용은 걸러낸다."""
    tokens = [t for t in re.findall(r"[가-힣]{2,}|\d+(?:\.\d+)?%?", choice_text) if len(t) > 1]
    if not tokens:
        return True
    matched = sum(1 for t in tokens if t in context)
    return (matched / len(tokens)) >= 0.7


@app.post("/api/edu-admin/quiz")
async def edu_admin_quiz(request: Request):
    """RAG로 검색한 근거 문서를 그대로 재료 삼아 객관식 퀴즈 1문제를 만든다 — 학생이
    방금 물어본 질문과 같은 자료를 활용(사용자 요청, 2026-09-19)."""
    payload = await request.json()
    question = (payload.get("question") or "").strip()
    school_level = payload.get("school_level", _DEFAULT_SCHOOL_LEVEL)
    subject = payload.get("subject", _DEFAULT_SUBJECT)
    if not question:
        return JSONResponse({"error": "질문을 입력하세요"}, status_code=400)
    if _state.get("edu_index") is None:
        return JSONResponse({"error": "RAG 인덱스가 없습니다"}, status_code=400)
    if _edu_job_status().get("status") == "running":
        return JSONResponse({"error": "학습이 진행 중입니다. 완료 후 다시 시도해주세요."}, status_code=409)

    allowed_source_ids = {cfg.source_id for cfg in collect_registry.sources_for(school_level, subject)}
    loop = asyncio.get_event_loop()

    def _run_quiz():
        hits = query_edu.retrieve(question, _state["edu_embed_model"], _state["edu_index"],
                                   _state["edu_meta"], _state["edu_bm25"], allowed_source_ids=allowed_source_ids)
        if not hits:
            return None, "", []
        context = "\n".join(f"[{h['title']}] {h['text'][:400]}" for h, _ in hits)
        messages = [
            {"role": "system", "content": _QUIZ_SYSTEM_PROMPT},
            {"role": "user", "content": f"[참고 자료]\n{context}"},
        ]
        # _state["llm"]을 통합관제와 공유하므로 RAG 모드와 동일하게 락으로 직렬화한다
        # (SIGSEGV 원인, 89번 참고).
        with _llm_lock:
            result = _state["llm"].create_chat_completion(messages=messages, temperature=0.0, max_tokens=400)
        raw = result["choices"][0]["message"]["content"]
        return raw, context, [h["title"] for h, _ in hits]

    start = time.perf_counter()
    raw, context, source_titles = await loop.run_in_executor(None, _run_quiz)
    elapsed = round(time.perf_counter() - start, 2)

    if raw is None:
        return JSONResponse({"error": "선택한 학교급·과목에는 아직 수집된 자료가 없습니다."}, status_code=400)

    try:
        quiz = json.loads(raw)
        choices = quiz["choices"]
        answer_index = int(quiz["answer_index"])
        if not (isinstance(choices, list) and len(choices) == 4 and 0 <= answer_index < 4):
            raise ValueError("형식 불일치")
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return JSONResponse({"error": "퀴즈 형식 생성에 실패했습니다 — 다시 시도해 주세요"}, status_code=502)

    if not _quiz_choice_grounded(choices[answer_index], context):
        return JSONResponse({"error": "이 자료로는 근거가 확실한 퀴즈를 만들지 못했습니다 — 다른 질문으로 시도해 주세요"}, status_code=502)

    return JSONResponse({
        "quiz": quiz, "sources": source_titles, "elapsed": elapsed,
    })


# ── AI 토론(2026-09-19) ──────────────────────────────────────────────────
# 설계 원칙은 이 프로젝트 전체와 동일 — "판정은 코드, LLM은 언어만". 처음엔 LLM에게
# "사전 정의한 2~4개 입장 중 하나로 분류하라"고 시켰는데, 실사용 중 학생 15명 전원이
# 한 팀에 쏠리는 걸 실제로 겪었다(LLM 분류가 통째로 실패하면 방어적 기본값이 전부
# 첫 카테고리로 떨어지는 구조였음). 대신 이미 RAG에 쓰는 임베딩 모델로 의견을
# 벡터화해서 표준 k-means로 자연스러운 묶음을 찾고, 그 위에 "정원을 넘지 않는 선에서
# 제일 가까운 팀에 배정"하는 그리디 규칙으로 크기를 강제로 맞춘다 — 팀을 나누는
# 결정 자체가 전부 코드/수학이고, LLM은 그렇게 정해진 팀에 이름만 붙인다.

_DEBATE_TOPIC_PROMPT = (
    "당신은 중학교 사회 토론 수업을 준비하는 선생님입니다. 아래 키워드로 학생들이 토론할 만한 "
    "질문을 하나 만드세요. 찬반이나 여러 입장으로 의견이 나뉠 수 있는 질문이어야 하고, 특정 "
    "입장 쪽으로 기울지 않은 균형 잡힌 질문이어야 합니다.\n"
    "설명 없이 반드시 아래 JSON 형식으로만 답하세요:\n"
    '{"topic": "..."}'
)

_DEBATE_OPINIONS_PROMPT_TMPL = (
    "당신은 중학교 사회 수업의 토론 진행자입니다. 아래 [토론 주제]에 대해, 중학생 {n}명이 각자 "
    "낼 법한 의견을 만드세요. 학생마다 서로 다른 생각을 갖도록 다양하게 작성하세요(전원이 같은 "
    "의견이면 안 됩니다 — 찬성·반대뿐 아니라 여러 각도의 의견이 골고루 섞이게 하세요). 이름은 "
    "실제로 쓸 법한 흔한 한국 이름(성+이름)으로 학생마다 다르게 지으세요. 의견은 50~100자, "
    "중학생 말투로 쓰세요.\n"
    "설명 없이 반드시 아래 JSON 형식으로만 답하세요(정확히 {n}명):\n"
    '{{"students": [{{"name": "...", "opinion": "..."}}, ...]}}'
)

_DEBATE_TEAM_LABEL_PROMPT_TMPL = (
    "당신은 중학교 사회 토론 수업을 준비하는 선생님입니다. 아래는 [토론 주제]에 대해 이미 "
    "비슷한 의견끼리 묶여 있는 {k}개 그룹입니다(그룹을 다시 나누지 마세요 — 이미 확정된 그룹임). "
    "각 그룹 학생들의 공통된 관점을 5~15자의 짧은 표현으로 요약해 이름만 붙이세요.\n"
    "설명 없이 반드시 아래 JSON 형식으로만 답하세요(그룹 순서와 정확히 같은 순서, {k}개):\n"
    '{{"labels": ["...", ...]}}'
)

_DEBATE_SUMMARY_PROMPT = (
    "당신은 중학교 사회 토론 수업을 진행한 선생님입니다. 아래는 토론 전/후 학생 입장 변화를 "
    "코드로 집계한 통계입니다(숫자는 이미 정확히 계산되어 있음). 이 결과를 학급에 설명하듯 "
    "두세 문장으로 요약하세요. 주어진 숫자 외의 새로운 수치를 지어내지 마세요.\n"
    "설명 없이 반드시 아래 JSON 형식으로만 답하세요:\n"
    '{"summary": "..."}'
)

_DEBATE_REOPINION_PROMPT = (
    "당신은 중학교 사회 수업의 토론 진행자입니다. 아래는 학생별 토론 전 의견과, 그 학생이 속한 "
    "팀이 읽은 참고 자료입니다. 이 자료를 읽은 뒤 학생이 다시 의견을 낸다면 어떻게 답할지 "
    "만드세요 — 원래 의견이 자료를 반영해 바뀌거나, 더 구체적인 근거를 들거나, 그대로 유지될 "
    "수도 있습니다. 자연스럽게 다양한 반응으로 쓰세요. 50~100자, 중학생 말투.\n"
    "설명 없이 반드시 아래 JSON 형식으로만 답하세요(학생 순서와 정확히 같은 순서, 같은 개수):\n"
    '{"students": [{"opinion": "..."}, ...]}'
)


def _llm_json_call(system_prompt: str, user_content: str, temperature: float = 0.0, max_tokens: int = 800) -> dict:
    """공유 LLM으로 JSON 응답 하나를 받는다 — 퀴즈·AI토론이 반복하는 패턴이라 공통
    헬퍼로 뺐다. _llm_lock으로 직렬화(SIGSEGV 원인, 89번 참고) — 반드시 executor
    스레드 안에서만 호출할 것(락 획득 자체가 블로킹이라 이벤트 루프에서 직접 부르면 안 됨)."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    with _llm_lock:
        result = _state["llm"].create_chat_completion(messages=messages, temperature=temperature, max_tokens=max_tokens)
    return json.loads(result["choices"][0]["message"]["content"])


def _llm_raw_call(system_prompt: str, user_content: str, temperature: float = 0.0, max_tokens: int = 800) -> str:
    """_llm_json_call과 같지만 파싱하지 않고 원문을 그대로 돌려준다 — 여러 항목의
    배열(학생 N명 등)을 만들 때는 JSON 전체 파싱보다 관대한(정규식 기반) 복구가
    필요해서(_parse_student_list/_parse_opinion_list 참고), 호출부에서 직접 처리한다."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    with _llm_lock:
        result = _state["llm"].create_chat_completion(messages=messages, temperature=temperature, max_tokens=max_tokens)
    return result["choices"][0]["message"]["content"]


def _parse_student_list(raw: str) -> list[dict]:
    """{"students": [{"name":..., "opinion":...}, ...]} 배열을 파싱하되, 전체 JSON이
    깨져도 개별 항목은 정규식으로 복구를 시도한다(2026-09-19 실측 발견) — 15명 생성을
    시켰더니 한 항목에서 모델이 `"}` 대신 `")`를 써서(중학생 말투 문장 끝에 괄호를
    섞어 쓰다 실수) 배열 전체 파싱이 깨졌다. 학생 한둘이 유실되는 건 허용 오차로
    보고(호출부에서 최소 인원만 확인), 항목 몇 개 때문에 전체를 재시도시키지 않는다."""
    try:
        data = json.loads(raw)
        items = data.get("students", [])
        if isinstance(items, list) and items:
            return items
    except json.JSONDecodeError:
        pass
    pattern = re.compile(r'"name"\s*:\s*"([^"]*)"\s*,\s*"opinion"\s*:\s*"([^"]*)"')
    return [{"name": m.group(1), "opinion": m.group(2)} for m in pattern.finditer(raw)]


def _parse_opinion_list(raw: str) -> list[str]:
    """{"students": [{"opinion":...}, ...]} 배열 전용 — _parse_student_list와 같은
    이유로 관대하게 복구한다(토론 후 재의견 수집에서도 같은 배열 파싱 위험이 있음)."""
    try:
        data = json.loads(raw)
        items = data.get("students", [])
        if isinstance(items, list) and items:
            return [(it.get("opinion") or "").strip() for it in items]
    except json.JSONDecodeError:
        pass
    pattern = re.compile(r'"opinion"\s*:\s*"([^"]*)"')
    return [m.group(1) for m in pattern.finditer(raw)]


def _balanced_cluster_assignment(vectors: np.ndarray, k: int, seed: int = 0) -> list[int]:
    """임베딩 벡터를 k개 그룹으로 최대한 균등하게 나눈다(사용자 요청, 2026-09-19) —
    LLM에게 직접 분류시켰다가 15명 전원이 한 팀으로 쏠리는 걸 실제로 겪은 뒤 도입.
    표준 k-means로 자연스러운 중심을 먼저 찾고(비슷한 의견끼리 뭉치는 성질은 유지),
    그 중심까지 거리 기준으로 "정원을 넘지 않는 선에서 제일 가까운 팀에 배정"하는
    그리디 방식으로 크기를 강제로 맞춘다. seed를 바꾸면 "재배치" 버튼이 다른 초기
    중심에서 시작해 다른 묶음을 만든다."""
    n = len(vectors)
    k = max(1, min(k, n))
    rng = np.random.RandomState(seed)
    idx = rng.choice(n, size=k, replace=False)
    centroids = vectors[idx].copy()
    labels = np.full(n, -1)
    for _ in range(20):
        dists = np.linalg.norm(vectors[:, None, :] - centroids[None, :, :], axis=2)
        new_labels = dists.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(k):
            pts = vectors[labels == c]
            if len(pts) > 0:
                centroids[c] = pts.mean(axis=0)

    dists = np.linalg.norm(vectors[:, None, :] - centroids[None, :, :], axis=2)
    base = n // k
    target = [base + (1 if i < n % k else 0) for i in range(k)]
    order = np.argsort(dists.min(axis=1))  # 자기 그룹이 확실한 학생부터 먼저 배정
    capacity = target.copy()
    assignment = [-1] * n
    for i in order:
        for c in np.argsort(dists[i]):
            if capacity[c] > 0:
                assignment[i] = int(c)
                capacity[c] -= 1
                break
    return assignment


@app.post("/api/edu-admin/debate/topic")
async def edu_admin_debate_topic(request: Request):
    """키워드 하나로 토론 주제를 자동 생성 — 직접 입력을 원하면 이 단계는 건너뛴다."""
    payload = await request.json()
    keyword = (payload.get("keyword") or "").strip()
    if not keyword:
        return JSONResponse({"error": "키워드를 입력하세요"}, status_code=400)
    if _edu_job_status().get("status") == "running":
        return JSONResponse({"error": "학습이 진행 중입니다. 완료 후 다시 시도해주세요."}, status_code=409)

    loop = asyncio.get_event_loop()

    def _run():
        data = _llm_json_call(_DEBATE_TOPIC_PROMPT, f"[키워드]\n{keyword}", temperature=0.7, max_tokens=200)
        topic = (data.get("topic") or "").strip()
        if not topic:
            raise ValueError("주제가 비어있음")
        return topic

    try:
        topic = await loop.run_in_executor(None, _run)
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return JSONResponse({"error": "주제 생성에 실패했습니다 — 다시 시도해 주세요"}, status_code=502)
    return JSONResponse({"topic": topic})


@app.post("/api/edu-admin/debate/opinions")
async def edu_admin_debate_opinions(request: Request):
    """학생 의견 에뮬레이션 — 이름·입장을 아직 정하지 않고 의견 텍스트만 다양하게
    생성한다(사용자 요청, 2026-09-19). 팀 배정은 별도 엔드포인트에서 코드가 결정한다
    — 의견 생성(자유 텍스트)과 그룹 배정(제약이 있는 결정)을 분리해야, 사용자가 팀
    개수를 바꾸거나 "재배치"를 눌러도 의견 자체는 다시 만들 필요가 없다."""
    payload = await request.json()
    topic = (payload.get("topic") or "").strip()
    num_students = max(4, min(int(payload.get("num_students", 10)), 20))
    if not topic:
        return JSONResponse({"error": "토론 주제를 입력하거나 키워드로 먼저 생성하세요"}, status_code=400)
    if _edu_job_status().get("status") == "running":
        return JSONResponse({"error": "학습이 진행 중입니다. 완료 후 다시 시도해주세요."}, status_code=409)

    loop = asyncio.get_event_loop()

    def _run():
        raw = _llm_raw_call(
            _DEBATE_OPINIONS_PROMPT_TMPL.format(n=num_students), f"[토론 주제]\n{topic}",
            temperature=0.9, max_tokens=120 * num_students,
        )
        students_raw = _parse_student_list(raw)
        if len(students_raw) < 2:
            raise ValueError("학생 의견 생성 실패")
        # 이름이 비었거나 중복되면 코드가 안전하게 보정 — LLM이 가짜 이름을 잘 못
        # 짓거나 같은 이름을 반복해도 화면에서 학생을 구분할 수 있어야 한다.
        seen_names: set[str] = set()
        students = []
        for i, s in enumerate(students_raw):
            name = (s.get("name") or "").strip() or f"학생{i + 1}"
            if name in seen_names:
                name = f"{name}({i + 1})"
            seen_names.add(name)
            opinion = (s.get("opinion") or "").strip()
            if not opinion:
                continue
            students.append({"id": len(students), "name": name, "opinion": opinion})
        if len(students) < 2:
            raise ValueError("유효한 학생 의견이 부족함")
        return students

    try:
        students = await loop.run_in_executor(None, _run)
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return JSONResponse({"error": "학생 의견 생성에 실패했습니다 — 다시 시도해 주세요"}, status_code=502)

    return JSONResponse({"topic": topic, "students": students})


@app.post("/api/edu-admin/debate/teams")
async def edu_admin_debate_teams(request: Request):
    """학생 의견을 임베딩해서 team_count개로 최대한 균등하게 묶는다(코드가 결정) —
    LLM은 이미 정해진 그룹에 이름만 붙인다. 팀 근거자료는 기존 RAG를 재사용, 근거가
    없으면 빈 목록을 그대로 보여준다(퀴즈의 grounding 원칙과 동일). seed를 바꾸면
    "재배치" 버튼이 다른 초기 중심에서 시작해 다른 묶음을 만든다."""
    payload = await request.json()
    topic = (payload.get("topic") or "").strip()
    students = payload.get("students", [])
    team_count = max(2, min(int(payload.get("team_count", 3)), 6))
    seed = int(payload.get("seed", 0))
    school_level = payload.get("school_level", _DEFAULT_SCHOOL_LEVEL)
    subject = payload.get("subject", _DEFAULT_SUBJECT)
    if not topic or len(students) < team_count:
        return JSONResponse({"error": "학생 수가 팀 수보다 적습니다"}, status_code=400)
    if _state.get("edu_index") is None:
        return JSONResponse({"error": "RAG 인덱스가 없습니다"}, status_code=400)
    if _edu_job_status().get("status") == "running":
        return JSONResponse({"error": "학습이 진행 중입니다. 완료 후 다시 시도해주세요."}, status_code=409)

    allowed_source_ids = {cfg.source_id for cfg in collect_registry.sources_for(school_level, subject)}
    loop = asyncio.get_event_loop()

    def _run():
        opinions = [s["opinion"] for s in students]
        vectors = np.asarray(_state["edu_embed_model"].encode(opinions, normalize_embeddings=True))
        assignment = _balanced_cluster_assignment(vectors, team_count, seed=seed)

        groups: dict[int, list[int]] = {}
        for student_idx, team_idx in enumerate(assignment):
            groups.setdefault(team_idx, []).append(students[student_idx]["id"])
        team_ids = sorted(groups.keys())

        sample_text = "\n".join(
            f"그룹{rank + 1}: " + " / ".join(
                s["opinion"] for s in students if s["id"] in groups[tid][:4]
            )
            for rank, tid in enumerate(team_ids)
        )
        label_data = _llm_json_call(
            _DEBATE_TEAM_LABEL_PROMPT_TMPL.format(k=len(team_ids)),
            f"[토론 주제]\n{topic}\n\n[그룹별 샘플 의견]\n{sample_text}",
            temperature=0.3, max_tokens=300,
        )
        labels = label_data.get("labels", [])
        if len(labels) != len(team_ids):
            labels = [f"팀 {i + 1}" for i in range(len(team_ids))]

        teams = []
        for rank, tid in enumerate(team_ids):
            label = (labels[rank] or "").strip() or f"팀 {rank + 1}"
            hits = query_edu.retrieve(
                f"{topic} {label}", _state["edu_embed_model"], _state["edu_index"], _state["edu_meta"],
                _state["edu_bm25"], top_k=3, allowed_source_ids=allowed_source_ids,
            )
            teams.append({
                "id": rank, "label": label, "student_ids": groups[tid],
                "materials": [{"title": h["title"], "text": h["text"][:500], "score": round(score, 3)} for h, score in hits],
            })
        return teams

    try:
        teams = await loop.run_in_executor(None, _run)
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return JSONResponse({"error": "팀 구성에 실패했습니다 — 다시 시도해 주세요"}, status_code=502)

    return JSONResponse({"teams": teams, "seed": seed})


@app.post("/api/edu-admin/debate/conclude")
async def edu_admin_debate_conclude(request: Request):
    """토론 종료 후 의견을 다시 수집하고, 토론 전/후 변화를 임베딩 거리로 집계한다
    (판정은 코드) — 각 팀의 "이전 의견들의 평균 벡터"를 중심으로 삼아, 토론 후 의견이
    자기 팀 중심과 다른 팀 중심 중 어디에 더 가까워졌는지로 "유지/이동"을 계산한다."""
    payload = await request.json()
    topic = (payload.get("topic") or "").strip()
    students = payload.get("students", [])
    teams = payload.get("teams", [])
    if not topic or not students or not teams:
        return JSONResponse({"error": "토론 데이터가 없습니다 — 먼저 토론을 시작하세요"}, status_code=400)
    if _edu_job_status().get("status") == "running":
        return JSONResponse({"error": "학습이 진행 중입니다. 완료 후 다시 시도해주세요."}, status_code=409)

    team_by_student: dict[int, dict] = {}
    for t in teams:
        for sid in t["student_ids"]:
            team_by_student[sid] = t

    loop = asyncio.get_event_loop()

    def _run():
        # 팀별로 한 번씩만 호출한다(2026-09-19 실측 발견) — 학생마다 팀 자료 전문을
        # 반복해서 넣었더니, 15명 정도만 돼도 입력이 컨텍스트 윈도우(4096)를 넘어
        # "Requested tokens exceed context window" 에러로 죽었다. 같은 팀 학생들은
        # 어차피 같은 자료를 읽었으니 자료는 팀당 한 번만 보여주면 충분하고, 호출을
        # 팀 단위로 쪼개면 학생 수·팀 수가 늘어도 한 호출의 크기는 항상 팀 하나
        # 분량으로 고정돼 안전하다.
        opinions_after_by_id: dict[int, str] = {}
        for t in teams:
            members = [s for s in students if s["id"] in t["student_ids"]]
            if not members:
                continue
            mats = t.get("materials", [])
            mat_text = " / ".join(m["text"][:300] for m in mats) or "(근거 자료 없음)"
            lines = [f"{i + 1}. 이름: {s['name']} / 원래 의견: {s['opinion']}" for i, s in enumerate(members)]
            user = (
                f"[토론 주제]\n{topic}\n\n[이 팀({t['label']})이 읽은 참고 자료]\n{mat_text}\n\n"
                "[학생별 토론 전 의견]\n" + "\n".join(lines)
            )
            raw = _llm_raw_call(_DEBATE_REOPINION_PROMPT, user, temperature=0.8, max_tokens=150 * len(members))
            after_opinions = _parse_opinion_list(raw)
            # 항목 하나가 깨져서 개수가 살짝 모자라도 전체를 실패시키지 않는다 —
            # 못 받은 학생은 "의견 유지"로 간주해 원래 의견을 그대로 쓴다(사용자
            # 실측 발견, 2026-09-19: 15명 중 한 명 파싱 실패로 전체가 502 나던 문제).
            for i, s in enumerate(members):
                opinions_after_by_id[s["id"]] = after_opinions[i].strip() if i < len(after_opinions) and after_opinions[i].strip() else s["opinion"]

        opinions_after = [opinions_after_by_id.get(s["id"], "") for s in students]

        before_vecs = np.asarray(_state["edu_embed_model"].encode([s["opinion"] for s in students], normalize_embeddings=True))
        after_vecs = np.asarray(_state["edu_embed_model"].encode(opinions_after, normalize_embeddings=True))

        id_to_idx = {s["id"]: i for i, s in enumerate(students)}
        centroids = []
        for t in teams:
            idxs = [id_to_idx[sid] for sid in t["student_ids"] if sid in id_to_idx]
            centroids.append(before_vecs[idxs].mean(axis=0) if idxs else np.zeros(before_vecs.shape[1]))
        centroids = np.asarray(centroids)

        results = []
        for i, s in enumerate(students):
            own_team = team_by_student[s["id"]]
            dists_after = np.linalg.norm(centroids - after_vecs[i], axis=1)
            nearest_team = teams[int(dists_after.argmin())]
            stayed = nearest_team["id"] == own_team["id"]
            results.append({
                **s, "opinion_after": opinions_after[i],
                "team_before": own_team["label"], "team_after": nearest_team["label"], "stayed": stayed,
            })
        return results

    try:
        results = await loop.run_in_executor(None, _run)
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return JSONResponse({"error": "토론 후 의견 수집에 실패했습니다 — 다시 시도해 주세요"}, status_code=502)

    team_stats = []
    for t in teams:
        members = [r for r in results if r["team_before"] == t["label"]]
        team_stats.append({
            "label": t["label"], "total": len(members),
            "stayed": sum(1 for m in members if m["stayed"]),
            "moved": sum(1 for m in members if not m["stayed"]),
        })
    overall = {"total": len(results), "stayed": sum(1 for r in results if r["stayed"]), "moved": sum(1 for r in results if not r["stayed"])}

    # 요약 의견(사용자 요청, 2026-09-19) — 숫자 자체는 이미 위에서 코드가 다 계산해
    # 뒀으니, LLM에게는 그 숫자를 문장으로 풀어 설명하는 것만 맡긴다(판정은 코드,
    # LLM은 언어만). 실패해도 그래프 자체는 이미 있으니 조용히 생략하고 넘어간다.
    def _gen_summary_text():
        stats_text = f"전체: 유지 {overall['stayed']}명 / 이동 {overall['moved']}명 (총 {overall['total']}명)\n"
        stats_text += "\n".join(f"- {t['label']}: 유지 {t['stayed']}명 / 이동 {t['moved']}명" for t in team_stats)
        try:
            data = _llm_json_call(_DEBATE_SUMMARY_PROMPT, f"[토론 주제]\n{topic}\n\n[통계]\n{stats_text}", temperature=0.3, max_tokens=300)
            return (data.get("summary") or "").strip()
        except (json.JSONDecodeError, KeyError, TypeError):
            return ""

    summary_text = await loop.run_in_executor(None, _gen_summary_text)

    return JSONResponse({
        "students": results,
        "summary_text": summary_text,
        "summary": overall,
        "team_stats": team_stats,
    })


# ── BidRadar 연동 API (2026-09-20) ────────────────────────────────────────
# 별개 프로젝트(BidRadar)와 코드·인프라를 섞지 않는다는 원칙(CLAUDE.md)은 지키되,
# 이건 "섞는" 게 아니라 이 프로젝트가 처음부터 서빙 API 용도로 아껴둔 포트(28081→8081,
# 의사결정_로그 5번 "나중에 실제 sLLM 서빙 API가 써야 한다")를 실제로 쓰는 것 —
# BidRadar의 요구사항 문서(2026-09-20)에서 합의한 공통 원칙 3가지를 그대로 반영한다:
# 판정 금지(분류·추출만, 최종 판정은 호출 측 코드) · 근거 필수(원문 인용) ·
# 조용한 실패 금지(실패 시 사유 명시 에러 응답). 유스케이스 C(첨부문서 분류)부터
# 구현 — 가장 입력이 짧고 스키마가 단순해 위험이 낮다.

_BIDRADAR_CLASSIFY_DOC_PROMPT = (
    "당신은 공공입찰 공고 첨부문서를 분류하는 보조 도구입니다. 아래 [문서 일부]가 실제 사업 "
    "내용이 아니라 제출서류 양식·법령 안내·공통 서식 같은 공통문서(boilerplate)인지 판단하세요. "
    "충족 여부나 사업 적합성 같은 다른 판정은 절대 하지 마세요 — 공통문서인지 아닌지만 분류합니다.\n"
    "reason에는 반드시 [문서 일부]에 실제로 등장하는 표현을 그대로 인용하세요 — 지어내지 마세요.\n"
    "설명 없이 반드시 아래 JSON 형식으로만 답하세요:\n"
    '{"is_boilerplate": true, "reason": "..."}'
)


_bidradar_call_log: list[dict] = []
_MAX_BIDRADAR_LOG = 300

_bidradar_jobs: dict[str, dict] = {}
_MAX_BIDRADAR_JOBS = 100


def _log_bidradar_call(
    endpoint: str, trace_id: str, success: bool,
    latency_ms: int = 0, tokens_in: int = 0, tokens_out: int = 0, error_code: str | None = None,
) -> None:
    """실제 BidRadar 호출 기록 — 시뮬레이션이 아니라 진짜 트래픽을 남긴다(사용자 요청,
    2026-09-20 "실제 연동되는 상황을 실시간으로 보여주면 좋겠다"). control-room의
    _simulation_history와 같은 패턴(런타임 동안만 유지, 배포 재시작 전까지)."""
    _bidradar_call_log.insert(0, {
        "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "endpoint": endpoint, "trace_id": trace_id, "success": success,
        "latency_ms": latency_ms, "tokens_in": tokens_in, "tokens_out": tokens_out, "error_code": error_code,
    })
    del _bidradar_call_log[_MAX_BIDRADAR_LOG:]


def _bidradar_check_auth(request: Request) -> JSONResponse | None:
    """내부망 토큰 인증(BidRadar와 합의, 8절) — 실패 시 반환용 에러 응답, 통과하면 None.
    토큰은 ANTHROPIC_API_KEY와 같은 패턴으로 systemctl --user set-environment로 서버에
    직접 등록한다(git·채팅에 평문 노출 금지, 이 프로젝트 전체 관행과 동일)."""
    expected = os.environ.get("BIDRADAR_API_TOKEN")
    if not expected:
        return JSONResponse({"error": {"code": "server_not_configured", "message": "BIDRADAR_API_TOKEN이 서버에 설정되지 않았습니다"}}, status_code=503)
    auth = request.headers.get("authorization", "")
    token = auth[7:] if auth.lower().startswith("bearer ") else ""
    if token != expected:
        return JSONResponse({"error": {"code": "unauthorized", "message": "인증 토큰이 없거나 올바르지 않습니다"}}, status_code=401)
    return None


@app.post("/v1/classify-doc")
async def bidradar_classify_doc(request: Request):
    """유스케이스 C — 첨부문서가 공통서식(boilerplate)인지 분류. BidRadar 요구사항
    문서 4절 스키마·5절 공통 계약을 그대로 따른다."""
    auth_error = _bidradar_check_auth(request)
    if auth_error is not None:
        return auth_error

    payload = await request.json()
    input_text = (payload.get("input_text") or "").strip()
    trace_id = payload.get("trace_id", "")
    max_tokens = min(int(payload.get("max_tokens", 200)), 500)
    if not input_text:
        _log_bidradar_call("classify-doc", trace_id, False, error_code="empty_input")
        return JSONResponse({"error": {"code": "empty_input", "message": "input_text가 비어있습니다"}, "trace_id": trace_id}, status_code=400)

    loop = asyncio.get_event_loop()
    start = time.perf_counter()

    def _run():
        messages = [
            {"role": "system", "content": _BIDRADAR_CLASSIFY_DOC_PROMPT},
            {"role": "user", "content": f"[문서 일부]\n{input_text}"},
        ]
        # _state["llm"]은 산업안전 통합관제·AI튜터·AI토론과 공유하는 동일 인스턴스라
        # _llm_lock으로 직렬화한다(SIGSEGV 원인, 89번) — 여기도 예외 없이 적용.
        with _llm_lock:
            result = _state["llm"].create_chat_completion(messages=messages, temperature=0.0, max_tokens=max_tokens)
        raw = result["choices"][0]["message"]["content"]
        usage = result.get("usage", {})
        return raw, usage

    try:
        raw, usage = await loop.run_in_executor(None, _run)
    except Exception as exc:  # noqa: BLE001 — 조용한 실패 금지(BidRadar 요구사항 1절) — 원인을 그대로 알려준다
        _log_bidradar_call("classify-doc", trace_id, False, latency_ms=round((time.perf_counter() - start) * 1000), error_code="inference_failed")
        return JSONResponse({"error": {"code": "inference_failed", "message": str(exc)}, "trace_id": trace_id}, status_code=502)

    try:
        data = json.loads(raw)
        is_boilerplate = bool(data["is_boilerplate"])
        reason = str(data.get("reason", "")).strip()
        if not reason:
            raise ValueError("reason 누락")
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        # 조용한 실패 금지 — 빈 결과 대신 사유를 명시한 실패 응답(BidRadar가 "확인 필요"로 넘김)
        _log_bidradar_call("classify-doc", trace_id, False, latency_ms=round((time.perf_counter() - start) * 1000), error_code="malformed_output")
        return JSONResponse({"error": {"code": "malformed_output", "message": "분류 결과 형식이 올바르지 않습니다"}, "trace_id": trace_id}, status_code=502)

    latency_ms = round((time.perf_counter() - start) * 1000)
    _log_bidradar_call("classify-doc", trace_id, True, latency_ms=latency_ms, tokens_in=usage.get("prompt_tokens", 0), tokens_out=usage.get("completion_tokens", 0))
    return JSONResponse({
        "output": {"is_boilerplate": is_boilerplate, "reason": reason},
        "model": "Qwen3-4B-Instruct-2507-Q4_K_M",
        "tokens_in": usage.get("prompt_tokens", 0),
        "tokens_out": usage.get("completion_tokens", 0),
        "latency_ms": latency_ms,
        "trace_id": trace_id,
    })


# ── 유스케이스 B — 관심주제 시맨틱 필터 (2026-09-20) ──────────────────────
# confidence는 절대 임계값으로 신뢰하지 않는다 — 이 프로젝트에서 실측으로 확인된
# 한계(8절, 90번 "점수 마진 게이팅 보류"와 같은 근본 원인)다. BidRadar와 합의한 대로
# 이 값은 키워드 매칭을 대체하지 않고 "사람이 검토할 후보"로만 쓰인다 — 서버 쪽에서도
# 강제로 걸러내지 않고(예: confidence<0.5는 제외 같은 임의 규칙) 그대로 다 돌려준다.

_BIDRADAR_CLASSIFY_TOPIC_PROMPT = (
    "당신은 입찰공고 제목을 고객 관심주제와 매칭하는 보조 도구입니다. 아래 [공고 제목]이 "
    "[관심주제 목록] 중 의미적으로 관련된 주제가 있는지 판단하세요. 핵심 키워드가 그대로 "
    "일치하지 않아도 같은 분야·기술을 가리키면 관련 있다고 판단하되, 확실하지 않으면 "
    "포함하지 마세요. [관심주제 목록]에 없는 topic_id는 절대 만들지 마세요.\n"
    "설명 없이 반드시 아래 JSON 형식으로만 답하세요(관련 주제가 없으면 빈 배열):\n"
    '{"matches": [{"topic_id": 0, "confidence": 0.0, "reason": "..."}]}'
)


def _parse_topic_matches(raw: str) -> list[dict]:
    """{"matches": [...]} 파싱 — 전체가 깨져도 개별 항목은 정규식으로 복구한다
    (AI토론 학생 목록 파싱에서 겪은 것과 같은 배열형 JSON 깨짐 위험에 동일 대응, 92번)."""
    try:
        data = json.loads(raw)
        items = data.get("matches", [])
        if isinstance(items, list):
            return items
    except json.JSONDecodeError:
        pass
    pattern = re.compile(r'"topic_id"\s*:\s*(\d+)[^{}]*?"confidence"\s*:\s*([\d.]+)[^{}]*?"reason"\s*:\s*"([^"]*)"')
    return [{"topic_id": int(m.group(1)), "confidence": float(m.group(2)), "reason": m.group(3)} for m in pattern.finditer(raw)]


@app.post("/v1/classify-topic")
async def bidradar_classify_topic(request: Request):
    """유스케이스 B — 공고 제목과 고객 관심주제의 의미적 매칭. 키워드 규칙이 놓치는
    표현 변형을 보조 신호로 잡는다(기존 키워드 매칭을 대체하지 않음, BidRadar 3절)."""
    auth_error = _bidradar_check_auth(request)
    if auth_error is not None:
        return auth_error

    payload = await request.json()
    title = (payload.get("input_text") or "").strip()
    topics = (payload.get("context") or {}).get("topics", [])
    trace_id = payload.get("trace_id", "")
    if not title:
        _log_bidradar_call("classify-topic", trace_id, False, error_code="empty_input")
        return JSONResponse({"error": {"code": "empty_input", "message": "input_text(공고 제목)가 비어있습니다"}, "trace_id": trace_id}, status_code=400)
    if not topics:
        _log_bidradar_call("classify-topic", trace_id, False, error_code="empty_topics")
        return JSONResponse({"error": {"code": "empty_topics", "message": "context.topics가 비어있습니다"}, "trace_id": trace_id}, status_code=400)

    valid_ids = set()
    topic_lines = []
    for t in topics:
        tid = t.get("topic_id")
        if tid is None:
            continue
        valid_ids.add(tid)
        kw = ", ".join(t.get("keywords", []))
        topic_lines.append(f"- topic_id={tid}: {t.get('name', '')}" + (f" (키워드: {kw})" if kw else ""))

    loop = asyncio.get_event_loop()
    start = time.perf_counter()

    def _run():
        user = f"[공고 제목]\n{title}\n\n[관심주제 목록]\n" + "\n".join(topic_lines)
        messages = [
            {"role": "system", "content": _BIDRADAR_CLASSIFY_TOPIC_PROMPT},
            {"role": "user", "content": user},
        ]
        with _llm_lock:
            result = _state["llm"].create_chat_completion(messages=messages, temperature=0.0, max_tokens=400)
        return result["choices"][0]["message"]["content"], result.get("usage", {})

    try:
        raw, usage = await loop.run_in_executor(None, _run)
    except Exception as exc:  # noqa: BLE001 — 조용한 실패 금지
        _log_bidradar_call("classify-topic", trace_id, False, latency_ms=round((time.perf_counter() - start) * 1000), error_code="inference_failed")
        return JSONResponse({"error": {"code": "inference_failed", "message": str(exc)}, "trace_id": trace_id}, status_code=502)

    matches = []
    for m in _parse_topic_matches(raw):
        try:
            tid = m["topic_id"]
            if tid not in valid_ids:
                continue  # 목록에 없는 topic_id를 만들어냈으면 버린다(코드가 재검증)
            conf = max(0.0, min(1.0, float(m.get("confidence", 0))))
            reason = str(m.get("reason", "")).strip()
            if not reason:
                continue
            matches.append({"topic_id": tid, "confidence": round(conf, 3), "reason": reason})
        except (KeyError, TypeError, ValueError):
            continue

    latency_ms = round((time.perf_counter() - start) * 1000)
    _log_bidradar_call("classify-topic", trace_id, True, latency_ms=latency_ms, tokens_in=usage.get("prompt_tokens", 0), tokens_out=usage.get("completion_tokens", 0))
    return JSONResponse({
        "output": {"matches": matches},
        "model": "Qwen3-4B-Instruct-2507-Q4_K_M",
        "tokens_in": usage.get("prompt_tokens", 0),
        "tokens_out": usage.get("completion_tokens", 0),
        "latency_ms": latency_ms,
        "trace_id": trace_id,
    })


# ── 유스케이스 A — 나라장터 A2 대체, 청킹 기반 (2026-09-20) ───────────────
# 컨텍스트 윈도우(4,096토큰)로 전체 문서(BidRadar 실측 중앙값 36,057자, p90 98,950자,
# 8절)를 한 번에 못 넣어서, rag/chunk.py의 기존 청킹 함수를 재사용해 문서를 조각내고
# 조각마다 요구사항을 뽑아 합친다 — RAG 파이프라인과 똑같은 도구를 재사용(검증된
# 컴포넌트를 다른 용도로 다시 쓰는 것뿐, 새 청킹 로직을 또 만들지 않음).
#
# v1 스코프 축소를 정직하게 기록: BidRadar의 원래 스키마(사업 항목별 세부, 평가항목,
# 예산조건, 자격요건, 제출정보 등)는 필드가 매우 많다 — 이번 세션에서 배열형 JSON도
# 항목이 늘수록 파싱이 깨지는 걸 반복 확인했는데(AI토론 92번), 필드가 훨씬 많은 중첩
# 객체 스키마는 그보다 더 위험하다고 판단해 requirements 배열 + summary 핵심 4개
# 필드만 먼저 구현했다. 나머지 필드는 이 v1이 안정적으로 도는 게 확인된 뒤 확장한다.

_BIDRADAR_EXTRACT_REQ_PROMPT = (
    "당신은 공공입찰 공고 문서에서 요구사항을 추출하는 보조 도구입니다. 아래 [문서 조각]은 "
    "긴 문서의 일부입니다. 이 조각 안에 있는 구체적인 요구사항(성능·인증·실적·인력 조건 등)만 "
    "추출하세요 — 이 조각에 없는 내용은 추출하지 마세요. 충족 여부는 절대 판단하지 마세요, "
    "요구사항 자체만 그대로 정리합니다. cite에는 이 조각에 실제로 있는 문장을 그대로 "
    "인용하세요 — 지어내지 마세요.\n"
    "설명 없이 반드시 아래 JSON 형식으로만 답하세요(이 조각에 요구사항이 없으면 빈 배열):\n"
    '{"requirements": [{"category": "성능|인증|실적|인력|기타", "req_text": "...", '
    '"req_value": "", "req_unit": "", "op": "gte|lte|eq|contains|manual", "cite": "..."}]}'
)

_BIDRADAR_EXTRACT_SUMMARY_PROMPT = (
    "당신은 공공입찰 공고 문서에서 사업 개요를 추출하는 보조 도구입니다. 아래 [문서 조각]은 "
    "긴 문서의 앞부분입니다. 이 안에서 찾을 수 있는 정보만 채우고, 없으면 빈 문자열로 두세요 "
    "— 지어내지 마세요. purpose는 원문을 그대로 발췌하세요(요약 금지).\n"
    "설명 없이 반드시 아래 JSON 형식으로만 답하세요:\n"
    '{"project_period": "", "project_budget": "", "purpose": "", '
    '"contact": {"department": "", "role": "", "phone": "", "email": ""}}'
)


def _parse_requirements(raw: str) -> list[dict]:
    """{"requirements": [...]} 파싱 — 전체가 깨져도 category/req_text/cite 세 핵심
    필드는 정규식으로 복구한다(92번과 동일한 배열형 JSON 깨짐 대응 패턴)."""
    try:
        data = json.loads(raw)
        items = data.get("requirements", [])
        if isinstance(items, list):
            return items
    except json.JSONDecodeError:
        pass
    pattern = re.compile(r'"category"\s*:\s*"([^"]*)"[^{}]*?"req_text"\s*:\s*"([^"]*)"[^{}]*?"cite"\s*:\s*"([^"]*)"')
    return [{"category": m.group(1), "req_text": m.group(2), "req_value": "", "req_unit": "", "op": "manual", "cite": m.group(3)} for m in pattern.finditer(raw)]


def _dedupe_requirements(requirements: list[dict]) -> list[dict]:
    """청크 겹침 구간(chunk_text의 overlap=200자)에 걸친 동일 요구사항이 인접한 두
    청크 양쪽에서 각각 추출돼 중복으로 남는 문제(의사결정_로그 95번에서 실측 확인,
    "실적"·"인증" 항목이 2번씩). 문장 유사도 기반 중복 제거는 "비슷하지만 다른 두
    요구사항"을 잘못 하나로 합칠 위험이 있어 일부러 피하고(95번에서 이미 그렇게
    판단해 보류했었음) — 겹침 구간에서 나온 중복은 원문 자체가 말 그대로 같다는
    점만 이용해, cite(원문을 그대로 인용해야 하는 필드)를 공백 정규화 후 완전히
    같을 때만 같은 요구사항으로 보고 먼저 나온 것만 남긴다."""
    seen: set[str] = set()
    deduped = []
    for r in requirements:
        key = " ".join(r["cite"].split())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    return deduped


def _bidradar_extract_job_worker(job_id: str, input_text: str) -> None:
    """백그라운드 스레드(run_in_executor)에서 실행 — 청크 수만큼 순차 LLM 호출이
    필요해(레이턴시 근본 원인) 동기 응답 대신 job_id를 먼저 돌려주고 여기서 진행한다
    (BidRadar 질의 2026-09-20, 의사결정_로그 참고). GPU 인스턴스가 있으면 그걸 쓰고
    (CPU 대비 실측 약 3.7배, quantization_pipeline 벤치마크), 없으면 CPU로 폴백한다.
    control-room의 GPU 프로파일 추론과 동일하게 _gpu_llm_busy를 세워서 학습이 끼어들지
    않게 한다."""
    global _gpu_llm_busy
    job = _bidradar_jobs[job_id]
    start = time.perf_counter()

    use_gpu = _state.get("llm_gpu") is not None
    llm = _state["llm_gpu"] if use_gpu else _state["llm"]
    lock = _llm_gpu_lock if use_gpu else _llm_lock

    if use_gpu:
        _gpu_llm_busy = True
    try:
        chunks = chunk_text(input_text, target_size=2500, overlap=200)
        job["chunks_total"] = len(chunks)
        all_requirements = []
        total_in = total_out = 0

        for i, chunk in enumerate(chunks):
            messages = [
                {"role": "system", "content": _BIDRADAR_EXTRACT_REQ_PROMPT},
                {"role": "user", "content": f"[문서 조각]\n{chunk}"},
            ]
            with lock:
                result = llm.create_chat_completion(messages=messages, temperature=0.0, max_tokens=800)
            raw = result["choices"][0]["message"]["content"]
            usage = result.get("usage", {})
            total_in += usage.get("prompt_tokens", 0)
            total_out += usage.get("completion_tokens", 0)
            for r in _parse_requirements(raw):
                cite = str(r.get("cite", "")).strip()
                req_value = str(r.get("req_value", "")).strip()
                # 근거 필수 원칙(BidRadar 1절) — cite가 실제로 이 조각 원문에 있는지
                # 코드가 재검증. 퀴즈 기능의 grounding 검증과 같은 함수 재사용(88·89번).
                if not cite or not _quiz_choice_grounded(cite, chunk):
                    continue
                # 숫자 치환 오염 방어(2026-09-20, "3초"→"eterminate"류 토큰 오염이
                # cite 필드에서 3회 관측됨, 의사결정_로그 94~96·100번). temperature=0으로
                # 재현해보니 같은 입력엔 매번 재현되지만 다른 문서·다른 숫자에서는 전혀
                # 재현 안 되는 드문 현상이었다 — 그래서 cite 문자열 전체를 엄격 매칭하는
                # 대신(줄바꿈 등으로 정상 인용도 오탐될 위험) req_value의 숫자가 cite에
                # 그대로 있는지만 좁게 검증한다. 토큰 중복도 검사는 단어 대부분이 겹치면
                # 숫자 하나가 바뀌어도 통과시키는 구조라 이 실패 패턴을 못 잡았다.
                if req_value and any(c.isdigit() for c in req_value) and req_value not in cite:
                    continue
                all_requirements.append({
                    "category": r.get("category", "기타"),
                    "req_text": str(r.get("req_text", "")).strip(),
                    "req_value": req_value,
                    "req_unit": str(r.get("req_unit", "")).strip(),
                    "op": r.get("op", "manual"),
                    "cite": cite,
                })
            job["chunks_processed"] = i + 1

        all_requirements = _dedupe_requirements(all_requirements)

        summary = {"project_period": "", "project_budget": "", "purpose": "", "contact": {}}
        if chunks:
            messages = [
                {"role": "system", "content": _BIDRADAR_EXTRACT_SUMMARY_PROMPT},
                {"role": "user", "content": f"[문서 조각]\n{chunks[0]}"},
            ]
            with lock:
                result = llm.create_chat_completion(messages=messages, temperature=0.0, max_tokens=400)
            raw = result["choices"][0]["message"]["content"]
            usage = result.get("usage", {})
            total_in += usage.get("prompt_tokens", 0)
            total_out += usage.get("completion_tokens", 0)
            try:
                summary_data = json.loads(raw)
                if isinstance(summary_data, dict):
                    summary.update(summary_data)
            except json.JSONDecodeError:
                pass

        latency_ms = round((time.perf_counter() - start) * 1000)
        job.update({
            "status": "done",
            "output": {"requirements": all_requirements, "summary": summary},
            "model": "Qwen3-4B-Instruct-2507-Q4_K_M" + ("-gpu" if use_gpu else "-cpu"),
            "chunks_processed": len(chunks),
            "tokens_in": total_in, "tokens_out": total_out, "latency_ms": latency_ms,
        })
        _log_bidradar_call("extract-requirements", job["trace_id"], True, latency_ms=latency_ms, tokens_in=total_in, tokens_out=total_out)
    except Exception as exc:  # noqa: BLE001 — 조용한 실패 금지
        latency_ms = round((time.perf_counter() - start) * 1000)
        job.update({"status": "error", "error": {"code": "inference_failed", "message": str(exc)}, "latency_ms": latency_ms})
        _log_bidradar_call("extract-requirements", job["trace_id"], False, latency_ms=latency_ms, error_code="inference_failed")
    finally:
        if use_gpu:
            _gpu_llm_busy = False


@app.post("/v1/extract-requirements")
async def bidradar_extract_requirements(request: Request):
    """유스케이스 A(v1) — 청크 순차 처리 특성상 동기 응답은 중앙값 문서(15청크) 기준
    수 분이 걸려 "상세페이지 열자마자 미리보기"와 맞지 않는다(BidRadar 질의
    2026-09-20). job_id를 즉시 반환하고 GET /v1/extract-requirements/{job_id}로
    폴링하는 방식으로 전환했다 — AI튜터 학습의 job_status 폴링과 같은 패턴."""
    auth_error = _bidradar_check_auth(request)
    if auth_error is not None:
        return auth_error

    payload = await request.json()
    input_text = (payload.get("input_text") or "").strip()
    trace_id = payload.get("trace_id", "")
    if not input_text:
        _log_bidradar_call("extract-requirements", trace_id, False, error_code="empty_input")
        return JSONResponse({"error": {"code": "empty_input", "message": "input_text가 비어있습니다"}, "trace_id": trace_id}, status_code=400)
    if _edu_job_status().get("status") == "running":
        return JSONResponse({"error": {"code": "gpu_busy", "message": "AI튜터 학습이 진행 중입니다. 잠시 후 다시 시도해주세요."}, "trace_id": trace_id}, status_code=409)

    job_id = uuid.uuid4().hex
    _bidradar_jobs[job_id] = {
        "status": "queued", "trace_id": trace_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "chunks_processed": 0, "chunks_total": None,
    }
    if len(_bidradar_jobs) > _MAX_BIDRADAR_JOBS:
        oldest = min(_bidradar_jobs, key=lambda k: _bidradar_jobs[k]["created_at"])
        del _bidradar_jobs[oldest]

    def _start():
        _bidradar_jobs[job_id]["status"] = "running"
        _bidradar_extract_job_worker(job_id, input_text)

    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _start)

    return JSONResponse({"job_id": job_id, "status": "queued", "trace_id": trace_id}, status_code=202)


@app.get("/v1/extract-requirements/{job_id}")
def bidradar_extract_requirements_status(job_id: str, request: Request):
    """폴링용 — BidRadar가 job_id로 완료 여부·진행률(chunks_processed/chunks_total)을
    확인한다. status: queued|running|done|error."""
    auth_error = _bidradar_check_auth(request)
    if auth_error is not None:
        return auth_error
    job = _bidradar_jobs.get(job_id)
    if job is None:
        return JSONResponse({"error": {"code": "job_not_found", "message": "존재하지 않거나 만료된 job_id입니다"}}, status_code=404)
    return JSONResponse(job)
