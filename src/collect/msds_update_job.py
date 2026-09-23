"""웹(/msds)의 "지금 업데이트" 버튼이 systemd-run 스코프로 띄우는 독립 프로세스
— mlops-web 배포/재시작으로 죽지 않게 분리한다(edu 학습 job과 같은 이유,
train/run_edu_training_job.py 참고). 순서: 최신 리비전 확인 → (있으면) 새
리비전 다운로드 → 재적재(카탈로그 갱신 포함) → FAISS 인덱스 재구축 → 리비전
기록. 각 단계를 job_status.json에 남겨서 웹이 폴링으로 진행 상황을 보여줄 수
있게 한다.

무거운 작업이다(2026-09-23 최초 전체 적재 실측: 다운로드 ~1분 + 적재 ~수십초 +
인덱스 재구축(GPU) ~20분) — 그래서 버튼을 "새 버전이 있을 때만" 활성화하고,
사람이 명시적으로 눌러야 시작되게 했다(자동 실행 안 함 — msds_hf_ingest.py의
check_for_updates()가 "알려만 주고 실행은 안 함"으로 설계된 것과 같은 이유).
"""

from __future__ import annotations

import json
import os
import subprocess
import traceback
from datetime import datetime, timezone
from pathlib import Path

from collect import msds_hf_ingest

_ROOT = Path(__file__).resolve().parents[2]
_JOB_STATUS_PATH = _ROOT / "data" / "processed" / "msds_update_job.json"
_PYTHON_BIN = _ROOT / ".venv" / "bin" / "python"


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
        check = msds_hf_ingest.check_for_updates()
        if check.get("error"):
            _write_status("error", detail=f"최신본 확인 실패: {check['error']}")
            return
        latest = check.get("latest_revision")
        if not latest or check.get("up_to_date"):
            _write_status("noop", detail="이미 최신본입니다 — 업데이트할 게 없습니다")
            return

        _write_status("downloading", target_revision=latest)
        url = f"https://huggingface.co/datasets/{msds_hf_ingest._HF_DATASET_ID}/resolve/{latest}/train.jsonl"
        dest = msds_hf_ingest._RAW_MSDS_DIR / f"inconvenience-msds_train_{latest[:10]}.jsonl"
        # 새 리비전이라 예상 해시를 아직 모른다 — 검증은 생략하고 실제 해시를
        # 결과에 남긴다(msds_hf_ingest.download()의 expected_sha256=None 경로).
        actual_sha = msds_hf_ingest.download(dest, url=url, expected_sha256=None)

        _write_status("ingesting", target_revision=latest)
        count = msds_hf_ingest.ingest(dest)
        msds_hf_ingest._record_ingested_revision(latest)

        _write_status("indexing", target_revision=latest, ingested_count=count)
        # GPU 0은 통합관제 등 실시간 서비스가 쓰는 공유 인스턴스라(bidradar_parallel_pool
        # 메모 참고) 부딪히지 않게 GPU 1(파인튜닝 전용, 지금은 유휴)로 고정한다.
        subprocess.run(
            [str(_PYTHON_BIN), "-m", "rag.build_index"],
            cwd=str(_ROOT),
            env={**os.environ, "PYTHONPATH": str(_ROOT / "src"), "CUDA_VISIBLE_DEVICES": "1"},
            check=True,
        )

        msds_hf_ingest.check_for_updates()  # msds_sync_status.json도 최신 상태로 갱신
        _write_status("done", target_revision=latest, ingested_count=count, sha256=actual_sha)
    except Exception:  # noqa: BLE001 — 어떤 단계에서 실패하든 상태 파일에 남겨야 웹이 멈춰있지 않는다
        _write_status("error", detail=traceback.format_exc()[-2000:])


if __name__ == "__main__":
    main()
