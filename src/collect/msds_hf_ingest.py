"""KOSHA MSDS(물질안전보건자료) 데이터를 HuggingFace의 공개 데이터셋에서 받아
data/processed/text/에 적재한다 — 이후 rag/build_index.py가 그대로 인덱싱한다.

**왜 공식 API로 직접 안 받았나**: data.go.kr의 물질안전보건자료 조회 서비스
(15157612, getChemList001)를 조사해보니 목록 조회가 "검색 전용"이다 — 물질명·CAS
번호 등으로 검색해야만 결과가 나오고, "전체 목록을 나열"하는 공식 파라미터가 없다.
사용자가 2026-09-23 활용신청 승인 후 공식 명세서(`docs/MSDS/1. 한국산업안전보건공단_
물질안전보건자료_오픈API활용가이드_260916수정.pdf`, v1.2)를 전달해줘서 확정 확인함
— searchWrd·searchCnd가 둘 다 필수 파라미터로 명시돼 있다. 실전에서 "전체"를
확보하려면 (a) chemId를 브루트포스로 탐색하거나(비공식, 수만 번 호출), (b) KOSHA
웹사이트를 스크래핑하거나, (c) 이미 이 작업을 해둔 공개 데이터셋을 재사용해야 한다
— 이 스크립트는 (c)를 택했다.

**출처**: huggingface.co/datasets/Yuyongkim/inconvenience-msds — KOSHA 공식 API로
수집한 48,966개 화학물질의 MSDS 16개 섹션 한글 원문(119.3M자)을 담은 데이터셋.
다운로드에 API 키·승인 절차가 필요 없다(공개 데이터셋, 익명 다운로드 가능).
라이선스: 데이터셋 자체는 CC BY 4.0, 원본 MSDS 데이터는 공공누리 제1유형(출처표시
필요, 변경·상업적 이용 자유) — RAG 인용은 물론 향후 파인튜닝 가공에도 문제없다.

**리비전 고정**: 2026-09-23 확인 시점 main HEAD 커밋으로 고정해서, 데이터셋이
나중에 갱신되더라도 이 스크립트가 받은 스냅샷이 무엇인지 항상 알 수 있게 한다.
데이터셋이 실제로 갱신됐다면 _HF_REVISION을 새 커밋으로 올리고 _EXPECTED_SHA256도
다시 확인해서 갱신할 것 — 해시 불일치는 자동으로 에러를 낸다(조용히 다른 데이터를
받지 않도록).

**공식 API는 완전히 버린 게 아님**: data.go.kr 키를 이미 받았다(2026-09-23, 개발계정
승인, `~/secrets/mlops-edge-lab/msds_api_key.env`) — 이 스냅샷 이후 신규·개정된
물질만 증분 갱신하는 용도로 쓸 수 있다. 공식 명세서 기준 오퍼레이션:
`getChemList001`(목록, searchWrd+searchCnd 필수 — 검색 전용이라 신규 물질을
"찾으려면" 물질명/CAS를 미리 알아야 함, 전체 나열 불가) +
`getChemDetail{01~16}1`(상세, chemId 필수, 섹션당 1회 호출) — 베이스 URL
`https://apis.data.go.kr/B552468/msdschem1`. 아직 구현 안 함(의사결정_로그 참고).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_TEXT_DIR = _ROOT / "data" / "processed" / "text"
_RAW_MSDS_DIR = _ROOT / "data" / "raw" / "msds"
_CATALOG_PATH = _ROOT / "data" / "processed" / "msds_catalog.json"
_SYNC_STATUS_PATH = _ROOT / "data" / "processed" / "msds_sync_status.json"
_INGESTED_REVISION_PATH = _ROOT / "data" / "processed" / "msds_ingested_revision.txt"

_HF_DATASET_ID = "Yuyongkim/inconvenience-msds"
_HF_REVISION = "5db49df655360dc69cc250ecb41058bf464553fa"
_HF_URL = f"https://huggingface.co/datasets/{_HF_DATASET_ID}/resolve/{_HF_REVISION}/train.jsonl"
_HF_API_URL = f"https://huggingface.co/api/datasets/{_HF_DATASET_ID}"
_EXPECTED_SHA256 = "2c342e638e403540076f0e0d13d0018f7671b11747b5a40cb67a5245d13a4227"

_SECTION_ORDER = list(range(1, 17))

_UNSET = object()


def current_pinned_revision() -> str:
    """"지금 실제로 적재돼 있는" 리비전 — 웹의 지금 업데이트 버튼으로 새 리비전을
    받은 적이 있으면 그 값을, 없으면(최초 상태) 이 파일의 하드코딩된 기본값을
    돌려준다. 업데이트 후에도 이 소스 파일의 _HF_REVISION 상수 자체는 건드리지
    않는다(실행 중인 작업이 자기 소스 코드를 고치는 건 사고 원인이 되기 쉽다) —
    "지금 뭘 갖고 있는지"는 이 상태 파일이 진실이다."""
    if _INGESTED_REVISION_PATH.exists():
        rev = _INGESTED_REVISION_PATH.read_text(encoding="utf-8").strip()
        if rev:
            return rev
    return _HF_REVISION


def _record_ingested_revision(revision: str) -> None:
    _INGESTED_REVISION_PATH.parent.mkdir(parents=True, exist_ok=True)
    _INGESTED_REVISION_PATH.write_text(revision, encoding="utf-8")


def _safe_name(name: str, max_bytes: int = 80) -> str:
    """파일명으로 안전하게 쓸 문자열로 정리 — 2026-09-23 실측: 문자 수(120자)로만
    잘랐더니 한글은 UTF-8로 1글자당 3바이트라 파일시스템의 255바이트 파일명 제한을
    넘겨 `OSError: File name too long`으로 2,355건째에서 전체가 죽었다. 바이트
    단위로 안전하게(멀티바이트 문자 중간을 자르지 않게) 잘라야 한다."""
    name = name.replace("/", "__").replace("\\", "__")
    name = re.sub(r"[^\w가-힣().-]+", "_", name)
    encoded = name.encode("utf-8")[:max_bytes]
    return encoded.decode("utf-8", errors="ignore")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(dest: Path, url: str = _UNSET, expected_sha256=_UNSET) -> str:
    """원본 JSONL을 저장 — 이미 있고 해시가 맞으면 재다운로드 생략(멱등). 실제
    받은 파일의 sha256을 반환한다(호출부가 "새 리비전을 정확히 이만큼 받았다"를
    기록할 수 있게).

    url/expected_sha256을 지정하면 그 리비전을 받는다(웹의 "지금 업데이트"가
    최신 리비전으로 받을 때 씀) — expected_sha256을 모르면(새 리비전이라 아직
    모름) None을 넘기면 검증을 생략하고 실제 해시를 로그로만 남긴다.

    urllib.request로 직접 받아봤더니 882MB 중 128KB만 받고 조용히 끊기는 문제가
    실제로 있었다(2026-09-23 실측 — 예외 없이 스트림이 일찍 끝남, HF의 CDN
    리다이렉트 체인과 urllib의 상호작용 문제로 추정). curl은 같은 URL을 문제없이
    완주했다(55초, 정확히 882,524,767바이트) — 그래서 subprocess로 curl을 그대로
    쓴다. 실패해도 이어받기 가능(-C -)."""
    url = _HF_URL if url is _UNSET else url
    expected_sha256 = _EXPECTED_SHA256 if expected_sha256 is _UNSET else expected_sha256

    if dest.exists() and expected_sha256 is not None and _sha256(dest) == expected_sha256:
        print(f"이미 있음(해시 일치) — 다운로드 생략: {dest}")
        return expected_sha256
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"다운로드 시작(curl): {url}")
    subprocess.run(
        ["curl", "-fL", "-C", "-", "--retry", "3", "-o", str(tmp), url],
        check=True,
    )
    actual = _sha256(tmp)
    if expected_sha256 is not None and actual != expected_sha256:
        tmp.unlink(missing_ok=True)
        raise ValueError(
            f"해시 불일치 — 예상 {expected_sha256}, 실제 {actual}. "
            "데이터셋이 갱신됐을 수 있음 — revision/expected_sha256를 huggingface.co에서 재확인할 것."
        )
    tmp.replace(dest)
    print(f"다운로드 완료(해시 {'검증됨' if expected_sha256 is not None else '미검증, 실제값 ' + actual}): {dest}")
    return actual


def _format_record(rec: dict) -> str:
    lines = [
        f"물질명: {rec.get('name_ko', '')} ({rec.get('name_en', '')})",
        f"CAS 번호: {rec.get('cas_no') or '(없음)'}",
        f"KOSHA 물질ID: {rec.get('chem_id', '')}",
        "",
    ]
    sections = {s["section_no"]: s for s in rec.get("sections", [])}
    for no in _SECTION_ORDER:
        s = sections.get(no)
        if not s:
            continue
        text = (s.get("text_ko") or "").strip()
        if not text:
            continue
        lines.append(f"[{no}. {s.get('title', '')}]")
        lines.append(text)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def ingest(jsonl_path: Path, limit: int | None = None) -> int:
    """JSONL을 한 줄씩 읽어 data/processed/text/msds_<chem_id>_<물질명>.txt로 적재.

    이미 추출된 순수 텍스트라 extract/run.py(PDF/HWP 등 포맷 변환용) 단계를 건너뛰고
    build_index.py가 바로 읽는 위치에 놓는다 — 변환할 게 없는 데이터를 그 파이프라인에
    억지로 통과시키는 건 불필요한 재작업일 뿐이다.

    같은 패스에서 물질 카탈로그(msds_catalog.json)도 같이 만든다 — 웹 페이지의
    "물질 선택" 기능이 RAG 청크를 뒤지지 않고 이 가벼운 목록에서 바로 찾게 하기
    위함(청크는 문서 조각이라 물질 단위 브라우징에 안 맞음)."""
    _TEXT_DIR.mkdir(parents=True, exist_ok=True)
    count = 0
    catalog: list[dict] = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            chem_id = rec.get("chem_id", "")
            name_ko = rec.get("name_ko", "unknown")
            name = _safe_name(name_ko)
            file_name = f"msds_{chem_id}_{name}.txt"
            out_path = _TEXT_DIR / file_name
            out_path.write_text(_format_record(rec), encoding="utf-8")
            catalog.append({
                "chem_id": chem_id,
                "name_ko": name_ko,
                "name_en": rec.get("name_en", ""),
                "cas_no": rec.get("cas_no") or "",
                "file": file_name,
            })
            count += 1
            if count % 2000 == 0:
                print(f"  {count}건 적재...")
            if limit and count >= limit:
                break

    _CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CATALOG_PATH.write_text(json.dumps(catalog, ensure_ascii=False), encoding="utf-8")
    print(f"카탈로그 저장: {_CATALOG_PATH} ({len(catalog)}건)")
    return count


def check_for_updates(write_status: bool = True) -> dict:
    """huggingface.co API로 데이터셋의 현재 최신 커밋(sha)을 조회해서 우리가 받아둔
    _HF_REVISION과 비교한다. 자동으로 재다운로드·재인덱싱하지는 않는다 — 48,966건
    재처리 + FAISS 재구축은 무거운 작업이라 사람이 보고 판단해야 한다(이 함수는
    "새 버전이 있다"를 알려주는 용도까지만).

    호출 실패(네트워크 등)도 조용히 삼키지 않고 status에 남긴다 — healthcheck류
    패턴과 동일(이 프로젝트 전반의 원칙, alerts.log 참고)."""
    pinned = current_pinned_revision()
    result = {
        "checked_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "pinned_revision": pinned,
        "latest_revision": None,
        "up_to_date": None,
        "error": None,
    }
    req = urllib.request.Request(_HF_API_URL, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        latest = data.get("sha")
        result["latest_revision"] = latest
        result["up_to_date"] = (latest == pinned) if latest else None
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as e:
        result["error"] = str(e)

    if write_status:
        _SYNC_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        _SYNC_STATUS_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", default=str(_RAW_MSDS_DIR / "inconvenience-msds_train.jsonl"))
    parser.add_argument("--limit", type=int, default=None, help="테스트용 — 앞 N건만 적재")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--check-updates-only", action="store_true", help="다운로드·적재 없이 최신 리비전만 확인")
    args = parser.parse_args()

    if args.check_updates_only:
        result = check_for_updates()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    jsonl_path = Path(args.jsonl)
    if not args.skip_download:
        download(jsonl_path)

    count = ingest(jsonl_path, limit=args.limit)
    _record_ingested_revision(_HF_REVISION)
    check_for_updates()
    print(f"완료 — {count}건을 {_TEXT_DIR}에 적재")


if __name__ == "__main__":
    main()
