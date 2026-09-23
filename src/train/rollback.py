"""MLflow Model Registry 롤백 — 사람이 명시적으로 호출했을 때만 실행된다.

배경(의사결정_로그 124번): `rag/drift_check.py`가 검색 품질 드리프트를 감지해도 이
모듈이 자동으로 호출되는 경로는 코드 어디에도 없다 — 드리프트 감지는 "알람만" 띄우고,
Production 버전을 되돌리는(되돌리기 어려운) 동작은 반드시 사람이 `/quality` 페이지의
버튼을 눌러야 실행된다. `agent/rules.py`의 "고위험 장비는 자동 실행하지 않고 사람 승인
대기"와 같은 원칙 — 이 파일의 함수를 호출하는 코드 경로는 `web/app.py`의
`/api/rag-drift/rollback` 핸들러(POST 요청, 사람이 확인 팝업을 눌러야 프론트가 보냄)
하나뿐이다.

`train/auto_retrain.py`의 `_register_and_promote()`가 이미 쓰고 있는
`client.transition_model_version_stage(..., stage="Production", archive_existing_versions=True)`
패턴의 반대 방향 — Archived였던 직전 버전을 다시 Production으로 올리고, 지금 Production인
버전은 (같은 호출의 `archive_existing_versions=True`가) Archived로 내린다.

**주의(정직하게 남김)**: 이 프로젝트에 RAG 검색(FAISS 인덱스) 자체는 MLflow Model
Registry에 등록된 모델이 아니다 — 등록된 모델은 `toy-sensor-lora`(센서 알림 문구 생성용
파인튜닝 모델) 하나뿐이다. 즉 "RAG 검색 드리프트 감지 → 이 모델 롤백"은 인과적으로 직접
연결돼 있지 않고, 이 프로젝트가 가진 유일한 Registry 대상에 원본 제안서의 Rollback 패턴을
학습 삼아 적용한 것이다 — 실서비스라면 RAG 인덱스 자체를 버전 관리하는 별도 장치(DVC
태그 롤백 등)가 더 맞겠지만, 이번 범위는 "사람 승인 없이는 되돌리기 동작이 실행되지
않는 구조"를 실제로 만들어보는 것이었다.
"""

from __future__ import annotations

import os

from mlflow.tracking import MlflowClient

os.environ.setdefault("MLFLOW_TRACKING_URI", "http://127.0.0.1:8082/mlflow")
# _mlflow_finetune_metrics()(web/app.py)와 같은 이유 — MLflow가 안 떠 있을 때 기본
# 클라이언트가 몇 분씩 재시도하며 API 응답을 막는 걸 막기 위한 짧은 타임아웃.
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "3")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")

_REGISTERED_MODEL_NAME = "toy-sensor-lora"


def get_rollback_candidate(model_name: str = _REGISTERED_MODEL_NAME) -> dict | None:
    """현재 Production 버전과, 되돌릴 대상(그보다 낮은 버전 중 Archived 상태인 가장
    최근 버전)을 찾는다. 되돌릴 게 없으면(Production이 없거나, Archived 이력이 없으면)
    None — 예: 이 프로젝트는 지금 v1이 Production인 채로 재학습이 한 번도 실패해
    자동 재승격된 적이 없어서, 현재 시점엔 Archived 버전이 없다(정직하게 그대로 둠,
    억지로 더미 버전을 만들지 않는다)."""
    client = MlflowClient()
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
    except Exception:  # noqa: BLE001 — MLflow 연결 실패 시 조용히 None(UI에서 "확인 불가"로 표시)
        return None

    production = [v for v in versions if v.current_stage == "Production"]
    archived = [v for v in versions if v.current_stage == "Archived"]
    if not production:
        return None
    current = max(production, key=lambda v: int(v.version))
    older_archived = [v for v in archived if int(v.version) < int(current.version)]
    if not older_archived:
        return None
    target = max(older_archived, key=lambda v: int(v.version))
    return {
        "model_name": model_name,
        "current_production_version": current.version,
        "rollback_target_version": target.version,
    }


def rollback_to_previous(model_name: str = _REGISTERED_MODEL_NAME) -> dict:
    """실제로 Production ↔ Archived를 맞바꾼다. **오직 사람이 명시적으로 호출했을
    때만 실행되어야 한다** — 이 함수 자체엔 자동 트리거 로직이 없고, 호출하는 유일한
    지점은 web/app.py의 POST /api/rag-drift/rollback 핸들러뿐이다(드리프트 감지
    코드는 이 함수를 절대 직접 호출하지 않는다)."""
    candidate = get_rollback_candidate(model_name)
    if candidate is None:
        return {
            "success": False,
            "message": "되돌릴 이전(Archived) 버전이 없습니다 — 이 모델이 아직 재승격된 적이 없거나 MLflow에 연결할 수 없습니다.",
        }

    client = MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=candidate["rollback_target_version"],
        stage="Production",
        archive_existing_versions=True,
    )
    return {
        "success": True,
        "model_name": model_name,
        "rolled_back_from": candidate["current_production_version"],
        "rolled_back_to": candidate["rollback_target_version"],
    }


def main() -> None:
    candidate = get_rollback_candidate()
    if candidate is None:
        print("되돌릴 이전 버전이 없습니다.")
        return
    print(f"롤백 후보: v{candidate['current_production_version']} (현재 Production) → v{candidate['rollback_target_version']} (Archived)")
    print("이 스크립트는 실행만으로 롤백하지 않습니다 — rollback_to_previous()를 명시적으로 호출해야 합니다.")


if __name__ == "__main__":
    main()
