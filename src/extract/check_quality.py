"""이미 추출된 결과(data/processed/text/*.txt)에 사후 품질 신호를 계산해 붙인다.
재추출(OCR 등) 없이 텍스트 파일만 다시 읽어서 빠르게 돈다 — 의사결정_로그 13번에서
"성공 ≠ 정확"이라고 지적만 해두고 미뤄뒀던 항목의 후속(70번).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from extract.quality import assess

_ROOT = Path(__file__).resolve().parents[2]
_PROCESSED_DIR = _ROOT / "data" / "processed"
_TEXT_DIR = _PROCESSED_DIR / "text"


def _safe_name(name: str) -> str:
    name = name.replace("/", "__").replace("\\", "__")
    return re.sub(r"[^\w가-힣().-]+", "_", name)[:180]


def main() -> None:
    results_path = _PROCESSED_DIR / "extraction_results.jsonl"
    if not results_path.exists():
        print("extraction_results.jsonl 없음 — 먼저 src/extract/run.py를 실행하세요")
        return

    records = [json.loads(line) for line in results_path.read_text(encoding="utf-8").split("\n") if line.strip()]

    report = []
    for r in records:
        if not r.get("ok"):
            continue
        text_path = _TEXT_DIR / f"{_safe_name(r['source'])}.txt"
        if not text_path.exists():
            continue
        text = text_path.read_text(encoding="utf-8")
        signals = assess(text)
        report.append({
            "source": r["source"], "method": r["method"],
            "char_count": signals.char_count, "korean_ratio": round(signals.korean_ratio, 4),
            "repeated_line_ratio": round(signals.repeated_line_ratio, 4),
            "suspicious_char_ratio": round(signals.suspicious_char_ratio, 4),
            "flags": signals.flags, "needs_review": signals.needs_review,
        })

    out_path = _PROCESSED_DIR / "quality_report.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for entry in report:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    total = len(report)
    review = sum(1 for e in report if e["needs_review"])
    print(f"=== 품질 점검 완료: {total}건 중 {review}건 재검토 필요 ({review / total:.1%}) ===" if total else "점검할 문서 없음")
    for e in report:
        if e["needs_review"]:
            print(f"  [재검토] {e['source']} ({e['method']}) — {', '.join(e['flags'])}")


if __name__ == "__main__":
    main()
