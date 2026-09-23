"""산업안전 RAG 검색 품질 드리프트 감지 — evidently(오픈소스)로 "기준(reference) 분포"
대비 "현재(current) 분포"의 PSI(Population Stability Index)를 계산한다.

배경(의사결정_로그 123·124번): 그립이 실제로 수주한 화장품 제조 AX 실증 사업 제안서에
"Evidently AI 기반 Data/Concept Drift 감지"가 MLOps 파이프라인 요소로 명시돼 있었고,
학습 삼아 이 프로젝트(산업안전 RAG) 자체에 적용한다.

- reference: `rag/build_drift_reference.py`가 만든 top-1 유사도 점수 분포(n=116, 의사결정_로그
  125번) — 골든셋 7문항만으론 PSI 계산 자체가 불안정해서(아래 임계치 설명 참고)
  LLM이 실제 코퍼스에서 초안한 질문 다수로 표본을 넓혔다. `safety_golden_set.jsonl`
  (7문항, hit-rate 정확도 회귀 테스트용)과는 별개 파일 — 그쪽은 안 건드렸다.
- current: `rag/query.py`의 `retrieve(..., log=True)`가 실서비스 요청마다 쌓는
  `retrieval_query_log.jsonl`의 최근 N건 top-1 유사도.

판정(PSI 임계치 초과 여부)은 이 파일의 순수 함수(`status_from_psi`)가 규칙으로 정하고,
LLM은 전혀 관여하지 않는다 — 이 프로젝트의 "판정은 규칙, 실행은 사람 승인" 원칙
(agent/rules.py와 동일)을 여기서도 그대로 지킨다. 이 모듈은 드리프트 여부를 계산·보고만
할 뿐, Rollback을 스스로 트리거하지 않는다(`train/rollback.py`는 사람이 웹 UI에서 버튼을
눌러야만 호출된다).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from evidently import Dataset, DataDefinition, Report
from evidently.metrics import ValueDrift
from evidently.ui.workspace import Workspace

_ROOT = Path(__file__).resolve().parents[2]
_REFERENCE_PATH = _ROOT / "data" / "processed" / "retrieval_score_reference.json"
_QUERY_LOG_PATH = _ROOT / "data" / "processed" / "retrieval_query_log.jsonl"

_EVIDENTLY_WORKSPACE_PATH = _ROOT / "evidently_workspace"
_EVIDENTLY_PROJECT_NAME = "safety-rag-drift"
"""Evidently 셀프호스팅 UI(`python -m evidently.cli ui`)가 읽는 로컬 워크스페이스
(의사결정_로그 126번). 이 파일의 PSI 계산 자체와는 독립된 "부가 기능"이다 — 매번
check_drift()가 실제로 PSI를 계산할 때(status가 stable/moderate/drift일 때만, "판단
불가" 상태에선 계산된 Report가 없으므로 저장할 것도 없다) 그 시점의 Report(Snapshot)를
여기 저장해서, UI가 스크린샷처럼 "시간에 따른 결과" 차트를 그릴 수 있게 한다."""

# evidently.ui.workspace의 실제 API(0.7.23 기준, 문서 대신 `inspect`로 직접 확인 —
# Workspace/Project/add_run/search_project 등 버전마다 바뀌는 걸 이전에도 겪었다,
# 124번). Workspace(로컬 파일 기반)와 CloudWorkspace는 별개 클래스 — Cloud 계정을
# 안 쓰기로 했으므로(작업 지시) Workspace만 쓴다.
_workspace_cache: dict[str, Workspace] = {}
_project_cache: dict[tuple[str, str], object] = {}


def _get_workspace(workspace_path: Path) -> Workspace:
    key = str(workspace_path)
    if key not in _workspace_cache:
        _workspace_cache[key] = Workspace.create(key)
    return _workspace_cache[key]


def _get_or_create_project(workspace: Workspace, workspace_path: Path, name: str):
    """`search_project`는 이름 완전일치로 검색한다(0.7.23 소스 직접 확인) — 매번
    check_drift()가 호출될 때마다 프로젝트를 새로 만들지 않고 기존 프로젝트에 계속
    스냅샷을 추가하려면 이 함수로 먼저 찾아야 한다."""
    cache_key = (str(workspace_path), name)
    if cache_key in _project_cache:
        return _project_cache[cache_key]
    existing = workspace.search_project(name)
    project = existing[0] if existing else workspace.create_project(
        name,
        description="산업안전 RAG top-1 검색 유사도 PSI 드리프트 이력 — rag/drift_check.py가 "
        "매번 실제 계산할 때마다(스냅샷 저장, /control-room \"데이터 드리프트 감지\" 카드와 "
        "같은 계산) 여기에도 기록한다(의사결정_로그 126번).",
    )
    _project_cache[cache_key] = project
    return project


def record_snapshot_to_workspace(
    snapshot,
    workspace_path: Path = _EVIDENTLY_WORKSPACE_PATH,
    project_name: str = _EVIDENTLY_PROJECT_NAME,
) -> None:
    """계산된 Report(Snapshot)를 Evidently UI 워크스페이스에 저장한다. 이 저장이
    실패해도(워크스페이스 파일 권한 문제 등) check_drift()의 PSI 판정 자체(반환값)에는
    영향을 주지 않는다 — /quality·/control-room 페이지의 핵심 경로가 부가 기능(UI 이력
    저장) 때문에 죽으면 안 된다는 이 프로젝트의 기존 방어 원칙(app.py의 evidently
    import guard와 같은 이유)을 여기서도 지킨다."""
    try:
        workspace = _get_workspace(workspace_path)
        project = _get_or_create_project(workspace, workspace_path, project_name)
        workspace.add_run(project.id, snapshot, include_data=False)
    except Exception as exc:  # noqa: BLE001 — UI 이력 저장 실패는 무시하고 계속
        print(f"[drift_check] Evidently UI 워크스페이스 저장 실패(무시): {exc}")

_CURRENT_WINDOW = 200
"""최근 몇 건의 실서비스 쿼리 로그를 "현재 분포"로 볼지. 너무 크면 오래된(이미 지나간)
쿼리까지 섞여 최신 드리프트를 못 잡고, 너무 작으면 노이즈에 흔들린다 — 정답은 없지만
200건이면 하루~며칠치 트래픽을 적당히 대표한다고 보고 시작값으로 잡음."""

_MIN_CURRENT_N = 30
"""current 표본이 이보다 적으면 PSI를 억지로 계산하지 않고 "데이터 부족"을 그대로
반환한다 — 이 프로젝트의 원칙(CLAUDE.md, "지어낸 수치·가짜 성공 사례 금지")과 같은
이유로, 통계적으로 의미 없는 값을 판정에 쓰지 않는다. 처음엔 10이었는데, null(진짜
드리프트 없는) 분포로 실측해보니 n=10에서는 표본 잡음만으로 PSI가 최대 4.3까지
튀어(40회 반복) "중간" 등급을 오탐할 위험이 있었다 — n=30에서는 잡음이 최대 1.7로
가라앉는 걸 확인하고 여기로 올렸다(의사결정_로그 125번)."""

# PSI 임계치 — 처음엔 Evidently 문서·업계 관례값(<0.1 안정/0.1~0.25 드리프트)을 그대로
# 썼는데(의사결정_로그 124번), 실측해보니 이 관례값은 이 프로젝트의 지표(RAG top-1
# 코사인 유사도 — 0.85~0.95처럼 폭이 아주 좁고 위로 쏠린 분포)에는 전혀 안 맞았다:
# **완전히 같은 분포에서 뽑은 두 표본을 비교해도**(진짜 드리프트가 없는 정상 상황) PSI가
# 0.06~1.69(40회 반복 샘플링 실측, 중앙값 0.38)까지 나와서 관례 임계치(0.25)를 표본
# 잡음만으로 넘어버렸다 — 가짜 드리프트 경보가 뜰 수밖에 없는 상태였다. 반면 실제로
# 분포가 눈에 띄게 벌어진 경우(예: 현재 평균이 기준보다 0.1~0.15 이상 낮음)는 PSI가
# 10~17 구간으로 뚜렷이 분리됐다. 그래서 Evidently 관례값 대신 이 지표 자체의 null
# 분포(정상 잡음)를 직접 측정해서 그보다 확실히 위, 실제 이동 사례보다는 확실히 아래에
# 새 임계치를 잡았다(의사결정_로그 125번) — "왜 2·8인가"의 근거는 이 실측 자체다.
PSI_STABLE_MAX = 2.0
PSI_MODERATE_MAX = 8.0


def status_from_psi(psi: float) -> str:
    """PSI 값을 세 등급으로 분류하는 순수 함수 — 판정은 여기(코드/규칙)가 하고
    LLM은 관여하지 않는다."""
    if psi < PSI_STABLE_MAX:
        return "stable"
    if psi < PSI_MODERATE_MAX:
        return "moderate"
    return "drift"


def load_reference(path: Path = _REFERENCE_PATH) -> dict | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not data.get("top1_scores"):
        return None
    return data


def load_recent_current(path: Path = _QUERY_LOG_PATH, n: int = _CURRENT_WINDOW) -> list[dict]:
    if not path.exists():
        return []
    # rag/query.py의 load_meta()와 같은 이유로 .splitlines() 대신 .split("\n") 사용.
    text = path.read_text(encoding="utf-8")
    rows = [json.loads(line) for line in text.split("\n") if line.strip()]
    return rows[-n:]


def build_report(reference_scores: list[float], current_scores: list[float]):
    """evidently의 실제 API(0.7.x)로 Report를 계산해서 Snapshot을 반환한다.
    `Report([ValueDrift(method="psi")])` + `Dataset.from_pandas()` — evidently 0.6 이전의
    `TestSuite`/`ColumnDriftMetric` API와는 다르다(버전마다 API가 바뀌어서 설치된 버전을
    직접 확인 후 이 형태로 작성함, 의사결정_로그 124번). PSI 숫자(`compute_psi`)뿐 아니라
    이 Snapshot 자체를 Evidently UI 워크스페이스에도 그대로 저장한다(126번,
    `record_snapshot_to_workspace`) — 두 번 계산하지 않고 한 Report를 재사용한다."""
    ref_df = pd.DataFrame({"top1_score": reference_scores})
    cur_df = pd.DataFrame({"top1_score": current_scores})
    ref_dataset = Dataset.from_pandas(ref_df, data_definition=DataDefinition())
    cur_dataset = Dataset.from_pandas(cur_df, data_definition=DataDefinition())

    report = Report([ValueDrift(column="top1_score", method="psi")])
    return report.run(cur_dataset, ref_dataset)


def compute_psi(reference_scores: list[float], current_scores: list[float]) -> float:
    """PSI 숫자만 필요한 호출부(테스트 등)를 위한 얇은 래퍼 — 내부적으로는
    `build_report`와 동일한 Report를 계산한다."""
    snapshot = build_report(reference_scores, current_scores)
    return float(snapshot.dict()["metrics"][0]["value"])


def check_drift(
    reference_path: Path = _REFERENCE_PATH,
    query_log_path: Path = _QUERY_LOG_PATH,
    window: int = _CURRENT_WINDOW,
    min_current_n: int = _MIN_CURRENT_N,
    record_to_workspace: bool = False,
    workspace_path: Path = _EVIDENTLY_WORKSPACE_PATH,
) -> dict:
    """드리프트 감지 메인 함수. 항상 다음 중 하나의 status를 반환한다:

    - "no_reference": 기준 분포 파일이 없음(골든셋 평가를 먼저 돌려야 함)
    - "insufficient_data": current 로그가 min_current_n건 미만 — 아직 판단 불가
    - "stable" / "moderate" / "drift": 실제로 계산된 PSI와 그 등급

    데이터가 부족한데 억지로 PSI를 계산해 보여주지 않는다 — "아직 드리프트 없음"과
    "아직 판단할 수 없음"을 구분해서 정직하게 반환한다(CLAUDE.md 원칙).

    `record_to_workspace`(기본 False)는 실제로 PSI를 계산한 경우에 한해(no_reference·
    insufficient_data는 계산된 Report가 없어 저장할 것도 없음) 그 Report를 Evidently UI
    워크스페이스에도 이력으로 남긴다(126번). 기본값을 False로 둔 이유: 이 함수는
    tests/test_drift.py에서 tmp_path 기준으로 반복 호출되는데, 기본이 True면 테스트가
    매번 저장소의 실제 evidently_workspace/를 건드리는 부작용이 생긴다 — 운영 호출부
    (web/app.py의 `_rag_drift_summary()`)에서만 명시적으로 True를 넘긴다."""
    reference = load_reference(reference_path)
    if reference is None:
        return {
            "status": "no_reference",
            "message": "기준(reference) 분포가 없습니다 — rag/evaluate_retrieval_safety.py를 먼저 실행하세요.",
        }

    current_rows = load_recent_current(query_log_path, window)
    current_scores = [r["top1_score"] for r in current_rows if isinstance(r.get("top1_score"), (int, float))]

    if len(current_scores) < min_current_n:
        return {
            "status": "insufficient_data",
            "message": f"실서비스 쿼리 로그가 {min_current_n}건 미만입니다(현재 {len(current_scores)}건) — 아직 드리프트를 판단할 만큼 쌓이지 않았습니다.",
            "n_current": len(current_scores),
            "min_required": min_current_n,
            "n_reference": reference["n"],
        }

    reference_scores = reference["top1_scores"]
    snapshot = build_report(reference_scores, current_scores)
    psi = float(snapshot.dict()["metrics"][0]["value"])
    status = status_from_psi(psi)

    if record_to_workspace:
        record_snapshot_to_workspace(snapshot, workspace_path=workspace_path)

    return {
        "status": status,
        "psi": round(psi, 4),
        "n_reference": len(reference_scores),
        "n_current": len(current_scores),
        "reference_mean": round(sum(reference_scores) / len(reference_scores), 4),
        "current_mean": round(sum(current_scores) / len(current_scores), 4),
        "reference_created_at": reference.get("created_at"),
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def main() -> None:
    result = check_drift()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
