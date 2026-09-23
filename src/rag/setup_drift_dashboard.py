"""Evidently 셀프호스팅 UI의 "safety-rag-drift" 프로젝트에 대시보드 패널을 구성한다.

`rag/drift_check.py`가 매번 계산 결과를 스냅샷으로 저장해도(의사결정_로그 126번),
Evidently UI의 "Dashboard" 탭은 스냅샷이 쌓인 것과 별개로 어떤 지표를 어떤 형태
(카운터/꺾은선 등)로 보여줄지 패널을 직접 구성해야 뭔가 나타난다 — 안 하면
"This dashboard is currently empty"만 보인다(사용자 실측 확인, 2026-09-24).

패널이 참조하는 지표 이름·라벨 키는 문서에 없어서 실제 저장된 스냅샷을
`/api/v2/snapshots/{project_id}/metrics`·`.../labels`·`.../label_values`로
직접 조회해 확인했다 — `evidently:metric_v2:ValueDrift`, 라벨
`{"column": "top1_score", "method": "psi"}`(둘 다 `rag/drift_check.py`의
`build_report()`가 항상 이 값으로 고정해서 부른다).

일회성 설정 스크립트다 — 패널 구성은 Evidently 워크스페이스에 영구 저장되므로
재배포 때마다 다시 실행할 필요는 없다. 대시보드를 다시 초기화하고 싶을 때만
`workspace.py 콘솔에서 project.dashboard.clear_dashboard()`를 먼저 부르고 재실행.
"""

from __future__ import annotations

from pathlib import Path

from evidently.sdk.models import PanelMetric
from evidently.sdk.panels import counter_panel, line_plot_panel, text_panel
from evidently.ui.workspace import Workspace

_ROOT = Path(__file__).resolve().parents[2]
_WORKSPACE_PATH = _ROOT / "evidently_workspace"
_PROJECT_NAME = "safety-rag-drift"

_METRIC = "evidently:metric_v2:ValueDrift"
_LABELS = {"column": "top1_score", "method": "psi"}


def setup() -> None:
    workspace = Workspace.create(str(_WORKSPACE_PATH))
    projects = workspace.search_project(_PROJECT_NAME)
    if not projects:
        raise RuntimeError(f"프로젝트 '{_PROJECT_NAME}'가 없습니다 — drift_check.check_drift()를 먼저 한 번 실행해야 합니다")
    project = projects[0]

    project.dashboard.add_panel(text_panel(title="산업안전 RAG 검색 품질 — top-1 유사도 PSI 드리프트 이력"))
    project.dashboard.add_panel(
        counter_panel(
            title="최신 PSI",
            size="half",
            values=[PanelMetric(metric=_METRIC, metric_labels=_LABELS, legend="PSI")],
            aggregation="last",
        )
    )
    project.dashboard.add_panel(
        line_plot_panel(
            title="PSI 추이 (top1_score, psi)",
            size="full",
            values=[PanelMetric(metric=_METRIC, metric_labels=_LABELS, legend="PSI")],
        )
    )
    project.save()
    print(f"대시보드 패널 3개 구성 완료 — project id: {project.id}")


def main() -> None:
    setup()


if __name__ == "__main__":
    main()
