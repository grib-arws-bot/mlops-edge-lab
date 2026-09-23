"""산업안전 RAG 검색 품질 드리프트 감지 — evidently(오픈소스)로 "기준(reference) 분포"
대비 "현재(current) 분포"의 PSI(Population Stability Index)를 계산한다.

배경(의사결정_로그 123·124번): 그립이 실제로 수주한 화장품 제조 AX 실증 사업 제안서에
"Evidently AI 기반 Data/Concept Drift 감지"가 MLOps 파이프라인 요소로 명시돼 있었고,
학습 삼아 이 프로젝트(산업안전 RAG) 자체에 적용한다.

- reference: `rag/evaluate_retrieval_safety.py`가 골든셋(사람이 검수한 질문+정답)을
  채점하는 시점에 함께 뽑아 저장한 top-1 유사도 점수 분포 — "검증된 정상 상태".
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

_ROOT = Path(__file__).resolve().parents[2]
_REFERENCE_PATH = _ROOT / "data" / "processed" / "retrieval_score_reference.json"
_QUERY_LOG_PATH = _ROOT / "data" / "processed" / "retrieval_query_log.jsonl"

_CURRENT_WINDOW = 200
"""최근 몇 건의 실서비스 쿼리 로그를 "현재 분포"로 볼지. 너무 크면 오래된(이미 지나간)
쿼리까지 섞여 최신 드리프트를 못 잡고, 너무 작으면 노이즈에 흔들린다 — 정답은 없지만
200건이면 하루~며칠치 트래픽을 적당히 대표한다고 보고 시작값으로 잡음."""

_MIN_CURRENT_N = 10
"""current 표본이 이보다 적으면 PSI를 억지로 계산하지 않고 "데이터 부족"을 그대로
반환한다 — 이 프로젝트의 원칙(CLAUDE.md, "지어낸 수치·가짜 성공 사례 금지")과 같은
이유로, 통계적으로 의미 없는 값을 판정에 쓰지 않는다."""

# PSI 임계치 — Evidently 문서·업계에서 관례적으로 쓰는 구간을 그대로 채택
# (< 0.1 안정 / 0.1~0.25 중간(주의) / > 0.25 유의미한 드리프트). 이 프로젝트가 직접
# 유도한 값이 아니라 Evidently 생태계의 관례값을 그대로 가져다 쓴다는 점을 분명히 남긴다
# — "왜 0.1·0.25인가"에 대한 자체 근거는 없고, 표준적으로 통용되는 구간을 학습 삼아
# 적용한 것(의사결정_로그 124번).
PSI_STABLE_MAX = 0.1
PSI_MODERATE_MAX = 0.25


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


def compute_psi(reference_scores: list[float], current_scores: list[float]) -> float:
    """evidently의 실제 API(0.7.x)로 PSI를 계산한다. `Report([ValueDrift(method="psi")])`
    + `Dataset.from_pandas()` — evidently 0.6 이전의 `TestSuite`/`ColumnDriftMetric` API와는
    다르다(버전마다 API가 바뀌어서 설치된 버전을 직접 확인 후 이 형태로 작성함,
    의사결정_로그 124번)."""
    ref_df = pd.DataFrame({"top1_score": reference_scores})
    cur_df = pd.DataFrame({"top1_score": current_scores})
    ref_dataset = Dataset.from_pandas(ref_df, data_definition=DataDefinition())
    cur_dataset = Dataset.from_pandas(cur_df, data_definition=DataDefinition())

    report = Report([ValueDrift(column="top1_score", method="psi")])
    result = report.run(cur_dataset, ref_dataset)
    return float(result.dict()["metrics"][0]["value"])


def check_drift(
    reference_path: Path = _REFERENCE_PATH,
    query_log_path: Path = _QUERY_LOG_PATH,
    window: int = _CURRENT_WINDOW,
    min_current_n: int = _MIN_CURRENT_N,
) -> dict:
    """드리프트 감지 메인 함수. 항상 다음 중 하나의 status를 반환한다:

    - "no_reference": 기준 분포 파일이 없음(골든셋 평가를 먼저 돌려야 함)
    - "insufficient_data": current 로그가 min_current_n건 미만 — 아직 판단 불가
    - "stable" / "moderate" / "drift": 실제로 계산된 PSI와 그 등급

    데이터가 부족한데 억지로 PSI를 계산해 보여주지 않는다 — "아직 드리프트 없음"과
    "아직 판단할 수 없음"을 구분해서 정직하게 반환한다(CLAUDE.md 원칙)."""
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
    psi = compute_psi(reference_scores, current_scores)
    status = status_from_psi(psi)

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
