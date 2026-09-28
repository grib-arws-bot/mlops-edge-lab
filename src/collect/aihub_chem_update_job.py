"""웹(/msds) "물성치 조회" 카드의 "지금 업데이트" 버튼이 systemd-run 스코프로
띄우는 독립 프로세스 — msds_update_job.py와 동일한 이유(mlops-web 배포/재시작으로
죽지 않게 분리)로 같은 패턴을 따른다. 순서: 용량 시그니처 확인 → (바뀌었으면)
라벨링 zip 6개 재다운로드 → 재추출 → CAS 카탈로그 재빌드 → 시그니처 기록.
FAISS 재인덱싱 단계가 없다 — 이 카탈로그는 RAG 검색이 아니라 CAS 키 조회용
평문 JSON이라 서버가 그냥 다시 읽어들이면 된다(재시작 시 자동 반영)."""

from __future__ import annotations

import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

from collect import aihub_chem_ingest

_ROOT = Path(__file__).resolve().parents[2]
_JOB_STATUS_PATH = _ROOT / "data" / "processed" / "aihub_chem_update_job.json"


def _write_status(status: str, **extra) -> None:
    payload = {
        "status": status,
        "updated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        **extra,
    }
    _JOB_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _JOB_STATUS_PATH.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    try:
        _write_status("checking")
        check = aihub_chem_ingest.check_for_updates()
        if check.get("error"):
            _write_status("error", detail=f"최신본 확인 실패: {check['error']}")
            return
        if check.get("up_to_date"):
            _write_status("noop", detail="용량 시그니처 기준 이미 최신본입니다")
            return

        _write_status("downloading")
        aihub_chem_ingest.download_labels()

        _write_status("ingesting")
        count = aihub_chem_ingest.ingest()
        aihub_chem_ingest._record_ingested_signature(aihub_chem_ingest._list_file_tree())

        aihub_chem_ingest.check_for_updates()  # aihub_chem_sync_status.json도 최신 상태로 갱신
        _write_status("done", ingested_count=count)
    except Exception:  # noqa: BLE001 — 어떤 단계에서 실패하든 상태 파일에 남겨야 웹이 멈춰있지 않는다
        _write_status("error", detail=traceback.format_exc()[-2000:])


if __name__ == "__main__":
    main()
