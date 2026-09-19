"""수집 결과 저장 — edu/collected/<source_id>/ 아래에 원문 파일 + 메타데이터를 남긴다.

edu/raw/(사용자가 직접 넣은 교육과정 문서)와 분리한다 — 이쪽은 봇이 자동으로 채워나가는
영역이라 성격이 다르고, 나중에 DVC로 따로 버저닝하기도 더 깔끔하다.

중복 수집 방지: 소스 URL을 키로 이미 수집한 항목은 건너뛴다(파일 내용이 아니라 URL
기준 — 같은 URL이 갱신될 수도 있는 소스는 커넥터가 필요시 별도로 처리).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from collect.models import CollectedItem

_ROOT = Path(__file__).resolve().parents[2]
_COLLECTED_DIR = _ROOT / "edu" / "collected"


def _source_dir(source_id: str) -> Path:
    d = _COLLECTED_DIR / source_id
    (d / "raw").mkdir(parents=True, exist_ok=True)
    return d


def _items_path(source_id: str) -> Path:
    return _source_dir(source_id) / "items.jsonl"


def already_collected_urls(source_id: str) -> set[str]:
    path = _items_path(source_id)
    if not path.exists():
        return set()
    urls = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                urls.add(json.loads(line)["source_url"])
    return urls


def save_raw_file(source_id: str, filename: str, data: bytes) -> str:
    """원문 파일을 저장하고, storage 루트(edu/collected/) 기준 상대경로를 반환한다."""
    dest = _source_dir(source_id) / "raw" / filename
    dest.write_bytes(data)
    return str(dest.relative_to(_COLLECTED_DIR))


def append_item(item: CollectedItem) -> None:
    with _items_path(item.source_id).open("a", encoding="utf-8") as f:
        f.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def collection_summary() -> list[dict]:
    """운영 콘솔(웹)이 그대로 쓸 수 있는 소스별 요약 — 건수·최근 수집시각·라이선스 유형."""
    summaries = []
    if not _COLLECTED_DIR.exists():
        return summaries
    for source_dir in sorted(_COLLECTED_DIR.iterdir()):
        items_path = source_dir / "items.jsonl"
        if not items_path.exists():
            continue
        rows = [json.loads(line) for line in items_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows:
            continue
        summaries.append({
            "source_id": source_dir.name,
            "item_count": len(rows),
            "last_collected_at": max(r["fetched_at"] for r in rows),
            "license_types": sorted({r["license_type"] for r in rows}),
            "allows_modification": all(r["allows_modification"] for r in rows if r["allows_modification"] is not None) and any(r["allows_modification"] is not None for r in rows),
            "sub_domain": rows[0]["sub_domain"],
            "school_level": rows[0]["school_level"],
        })
    return summaries
