"""rag/drift_check.py에 대한 단위 테스트 — PSI 임계치 판정(순수 함수)과 데이터
부족/기준 없음 케이스를 검증한다. 실제 evidently 호출(compute_psi)도 포함해서
설치된 버전의 API(Report + ValueDrift(method="psi"))가 실제로 동작하는지까지
확인한다 — evidently는 CI(GitHub Actions)에도 설치해서 이 테스트가 돌게 했다
(.github/workflows/deploy.yml, 의사결정_로그 124번).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag import drift_check


def test_status_boundaries():
    assert drift_check.status_from_psi(0.0) == "stable"
    assert drift_check.status_from_psi(0.09) == "stable"
    assert drift_check.status_from_psi(0.1) == "moderate"  # 경계값은 "안정"이 아니라 "중간"
    assert drift_check.status_from_psi(0.24) == "moderate"
    assert drift_check.status_from_psi(0.25) == "drift"  # 경계값은 "중간"이 아니라 "드리프트"
    assert drift_check.status_from_psi(1.0) == "drift"


def test_check_drift_no_reference(tmp_path):
    result = drift_check.check_drift(
        reference_path=tmp_path / "no_such_reference.json",
        query_log_path=tmp_path / "no_such_log.jsonl",
    )
    assert result["status"] == "no_reference"


def test_check_drift_insufficient_current_data(tmp_path):
    ref_path = tmp_path / "reference.json"
    ref_path.write_text(json.dumps({"top1_scores": [0.8, 0.81, 0.79], "n": 3}), encoding="utf-8")
    log_path = tmp_path / "log.jsonl"
    # min_current_n(기본 10)보다 적은 3건만 기록
    with log_path.open("w", encoding="utf-8") as f:
        for score in [0.7, 0.71, 0.72]:
            f.write(json.dumps({"top1_score": score, "query_len": 5}) + "\n")

    result = drift_check.check_drift(reference_path=ref_path, query_log_path=log_path, min_current_n=10)
    assert result["status"] == "insufficient_data"
    assert result["n_current"] == 3


def test_check_drift_stable_when_distributions_match(tmp_path):
    ref_scores = [0.75 + 0.001 * i for i in range(30)]
    cur_scores = [0.75 + 0.001 * i for i in range(30)]  # 사실상 동일 분포

    ref_path = tmp_path / "reference.json"
    ref_path.write_text(json.dumps({"top1_scores": ref_scores, "n": len(ref_scores)}), encoding="utf-8")
    log_path = tmp_path / "log.jsonl"
    with log_path.open("w", encoding="utf-8") as f:
        for score in cur_scores:
            f.write(json.dumps({"top1_score": score, "query_len": 5}) + "\n")

    result = drift_check.check_drift(reference_path=ref_path, query_log_path=log_path, min_current_n=10)
    assert result["status"] == "stable"
    assert result["psi"] < drift_check.PSI_STABLE_MAX


def test_check_drift_flags_drift_when_distribution_shifts(tmp_path):
    # reference는 높은 유사도(정상), current는 훨씬 낮은 유사도로 크게 이동한 상황을
    # 인위적으로 구성 — 8번 항목의 "수동 트리거" 검증과 같은 방식을 테스트로도 고정.
    ref_scores = [0.85 + 0.002 * (i % 10) for i in range(30)]
    cur_scores = [0.40 + 0.002 * (i % 10) for i in range(30)]

    ref_path = tmp_path / "reference.json"
    ref_path.write_text(json.dumps({"top1_scores": ref_scores, "n": len(ref_scores)}), encoding="utf-8")
    log_path = tmp_path / "log.jsonl"
    with log_path.open("w", encoding="utf-8") as f:
        for score in cur_scores:
            f.write(json.dumps({"top1_score": score, "query_len": 5}) + "\n")

    result = drift_check.check_drift(reference_path=ref_path, query_log_path=log_path, min_current_n=10)
    assert result["status"] == "drift"
    assert result["psi"] > drift_check.PSI_MODERATE_MAX


def test_load_recent_current_uses_most_recent_window(tmp_path):
    log_path = tmp_path / "log.jsonl"
    with log_path.open("w", encoding="utf-8") as f:
        for i in range(20):
            f.write(json.dumps({"top1_score": i / 20, "query_len": 5}) + "\n")

    rows = drift_check.load_recent_current(log_path, n=5)
    assert len(rows) == 5
    assert rows[-1]["top1_score"] == 19 / 20  # 가장 최근 5건이어야 함
