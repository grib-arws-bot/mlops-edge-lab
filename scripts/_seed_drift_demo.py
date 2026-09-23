#!/usr/bin/env python3
"""[개발/검증 전용, 정식 파이프라인 아님] rag/drift_check.py + /quality 페이지의 드리프트
감지·UI가 실제로 동작하는지 눈으로 확인하기 위한 일회성 스크립트.

인위적으로 이동시킨(reference보다 뚜렷이 낮은) 테스트 로그를 실제 운영 로그 파일
(data/processed/retrieval_query_log.jsonl)에 잠깐 주입해서 PSI가 실제로 올라가고
/quality 화면에 "드리프트"로 뜨는지 확인한 뒤, 검증이 끝나면 반드시 restore로
원상복구해서 실 서비스 로그를 오염시키지 않는다. 이 스크립트의 존재 이유(파일명 앞
`_`도 의도적 — 정식 파이프라인 스크립트와 구분)와 사용 사실을 커밋 메시지에 남긴다
(의사결정_로그 124번).

사용법(서버에서):
    .venv/bin/python scripts/_seed_drift_demo.py seed      # 백업 후 이동된 분포 주입
    curl -s http://127.0.0.1:8081/api/rag-drift/check       # 여기서 실제 UI/API로 확인
    .venv/bin/python scripts/_seed_drift_demo.py restore    # 백업에서 원상복구
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag import drift_check  # noqa: E402 — sys.path 조정 후 임포트

_ROOT = Path(__file__).resolve().parents[1]
_LOG_PATH = _ROOT / "data" / "processed" / "retrieval_query_log.jsonl"
_BACKUP_PATH = _ROOT / "data" / "processed" / "retrieval_query_log.jsonl._seed_drift_demo.bak"

_N_SEED_ROWS = 30
# reference(2026-09-24 기준 n=116, 평균 0.90) 대비 현실적인 "중간 정도 저하" 시나리오로
# 조정(의사결정_로그 125번) — 처음엔 0.35로 아예 안 겹치게 잡아서 PSI가 14~17까지
# 튀었다. 실측해보니 reference 범위(0.85~0.95) 바로 아래로 걸치는 값이 두 자릿수보다
# 훨씬 "있을 법한" 크기(대략 10~11)의 PSI를 만든다.
_SHIFTED_LOW, _SHIFTED_HIGH = 0.72, 0.84


def seed() -> None:
    if _BACKUP_PATH.exists():
        print(f"이미 백업이 있습니다({_BACKUP_PATH}) — 먼저 restore를 실행하세요.")
        sys.exit(1)

    if _LOG_PATH.exists():
        _BACKUP_PATH.write_bytes(_LOG_PATH.read_bytes())
        print(f"기존 로그를 백업했습니다: {_BACKUP_PATH}")
    else:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        _BACKUP_PATH.write_text("", encoding="utf-8")
        print("기존 로그가 없었습니다 — restore 시 빈 파일로 되돌리도록 빈 백업을 만듭니다.")

    rng = random.Random(0)
    with _LOG_PATH.open("a", encoding="utf-8") as f:
        for i in range(_N_SEED_ROWS):
            row = {
                "ts": f"1970-01-01T00:{i // 60:02d}:{i % 60:02d}+00:00",
                "top1_score": round(rng.uniform(_SHIFTED_LOW, _SHIFTED_HIGH), 4),
                "query_len": 10,
                "_seed_drift_demo": True,  # 실 로그 레코드와 구분용 — 운영 코드는 이 필드를 읽지 않음
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"인위적 테스트 로그 {_N_SEED_ROWS}건 주입 완료 → {_LOG_PATH}")

    result = drift_check.check_drift()
    print(json.dumps(result, ensure_ascii=False, indent=2))


def restore() -> None:
    if not _BACKUP_PATH.exists():
        print("백업 파일이 없습니다 — seed를 먼저 실행했는지 확인하세요.")
        sys.exit(1)
    _LOG_PATH.write_bytes(_BACKUP_PATH.read_bytes())
    _BACKUP_PATH.unlink()
    print(f"원상복구 완료 — {_LOG_PATH}를 백업 시점 내용으로 되돌리고 백업 파일을 지웠습니다.")


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in ("seed", "restore"):
        print(__doc__)
        sys.exit(1)
    (seed if sys.argv[1] == "seed" else restore)()


if __name__ == "__main__":
    main()
