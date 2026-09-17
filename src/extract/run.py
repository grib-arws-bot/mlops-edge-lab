"""data/raw 전체를 훑어서 추출하고 결과를 data/processed에 남긴다.

MLflow에도 요약을 기록한다 — "데이터 파이프라인 실행도 실험처럼 추적한다"는 게
이 프로젝트가 MLOps에서 배우려는 습관 중 하나라서다(학습 run뿐 아니라 데이터
전처리도 언제 몇 건을 어떤 방법으로 처리했는지 남겨야 나중에 재현·비교가 된다).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import mlflow

from extract.core import extract_text
from extract.walk import iter_source_files

_ROOT = Path(__file__).resolve().parents[2]
_RAW_DIR = _ROOT / "data" / "raw"
_PROCESSED_DIR = _ROOT / "data" / "processed"
_TEXT_DIR = _PROCESSED_DIR / "text"


def _safe_name(name: str) -> str:
    name = name.replace("/", "__").replace("\\", "__")
    return re.sub(r"[^\w가-힣().-]+", "_", name)[:180]


def main() -> None:
    _TEXT_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    for source_name, data in iter_source_files(_RAW_DIR):
        suffix = Path(source_name).suffix
        result = extract_text(data, suffix)
        record = {
            "source": source_name,
            "ok": result.ok,
            "method": result.method,
            "attempted": result.attempted,
            "chars": len(result.text) if result.ok else 0,
            "error": result.error,
        }
        records.append(record)

        status = "OK " if result.ok else "FAIL"
        print(f"[{status}] {result.method:12s} {len(result.text):7d}자  {source_name}")
        if not result.ok:
            print(f"       └ {result.error}")

        if result.ok:
            out_path = _TEXT_DIR / f"{_safe_name(source_name)}.txt"
            out_path.write_text(result.text, encoding="utf-8")

    _PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = _PROCESSED_DIR / "extraction_results.jsonl"
    with summary_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = len(records)
    ok_count = sum(r["ok"] for r in records)
    by_method: dict[str, int] = {}
    for r in records:
        key = r["method"] if r["ok"] else f"FAIL:{r['method']}"
        by_method[key] = by_method.get(key, 0) + 1

    print("\n=== 요약 ===")
    print(f"전체 {total}건 중 성공 {ok_count}건 ({ok_count / total:.0%})" if total else "처리할 파일 없음")
    for method, count in sorted(by_method.items(), key=lambda kv: -kv[1]):
        print(f"  {method:20s} {count}건")

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("data-extraction")
    with mlflow.start_run(run_name="extract-raw"):
        mlflow.log_metric("total_files", total)
        mlflow.log_metric("ok_files", ok_count)
        mlflow.log_metric("success_rate", ok_count / total if total else 0.0)
        for method, count in by_method.items():
            mlflow.log_metric(f"count__{method}", count)
        mlflow.log_artifact(str(summary_path))


if __name__ == "__main__":
    main()
