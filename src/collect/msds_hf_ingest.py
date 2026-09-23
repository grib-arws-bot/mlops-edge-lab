"""KOSHA MSDS(물질안전보건자료) 데이터를 HuggingFace의 공개 데이터셋에서 받아
data/processed/text/에 적재한다 — 이후 rag/build_index.py가 그대로 인덱싱한다.

**왜 공식 API로 직접 안 받았나**: data.go.kr의 물질안전보건자료 조회 서비스
(15157612, getChemList001)를 조사해보니 목록 조회가 "검색 전용"이다 — 물질명·CAS
번호 등으로 검색해야만 결과가 나오고, "전체 목록을 나열"하는 공식 파라미터가 없다
(공식 HWP 명세서에도 명시됨). 실전에서 "전체"를 확보하려면 (a) chemId를 브루트포스로
탐색하거나(비공식, 수만 번 호출), (b) KOSHA 웹사이트를 스크래핑하거나, (c) 이미 이
작업을 해둔 공개 데이터셋을 재사용해야 한다 — 이 스크립트는 (c)를 택했다.

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

**공식 API는 완전히 버린 게 아님**: data.go.kr 키를 받으면(활용신청은 별개로 진행
중) 이 스냅샷 이후 신규·개정된 물질만 증분 갱신하는 용도로 여전히 쓸모 있다 — 그
경우엔 getChemDetail{NN}1 오퍼레이션(section 1~16)을 개별 chemId로 호출하면 된다
(의사결정_로그 참고, 엔드포인트는 조사로 확인해뒀음).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_TEXT_DIR = _ROOT / "data" / "processed" / "text"
_RAW_MSDS_DIR = _ROOT / "data" / "raw" / "msds"

_HF_REVISION = "5db49df655360dc69cc250ecb41058bf464553fa"
_HF_URL = f"https://huggingface.co/datasets/Yuyongkim/inconvenience-msds/resolve/{_HF_REVISION}/train.jsonl"
_EXPECTED_SHA256 = "2c342e638e403540076f0e0d13d0018f7671b11747b5a40cb67a5245d13a4227"

_SECTION_ORDER = list(range(1, 17))


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


def download(dest: Path) -> None:
    """원본 JSONL을 저장 — 이미 있고 해시가 맞으면 재다운로드 생략(멱등).

    urllib.request로 직접 받아봤더니 882MB 중 128KB만 받고 조용히 끊기는 문제가
    실제로 있었다(2026-09-23 실측 — 예외 없이 스트림이 일찍 끝남, HF의 CDN
    리다이렉트 체인과 urllib의 상호작용 문제로 추정). curl은 같은 URL을 문제없이
    완주했다(55초, 정확히 882,524,767바이트) — 그래서 subprocess로 curl을 그대로
    쓴다. 실패해도 이어받기 가능(-C -)."""
    if dest.exists() and _sha256(dest) == _EXPECTED_SHA256:
        print(f"이미 있음(해시 일치) — 다운로드 생략: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"다운로드 시작(curl): {_HF_URL}")
    subprocess.run(
        ["curl", "-fL", "-C", "-", "--retry", "3", "-o", str(tmp), _HF_URL],
        check=True,
    )
    actual = _sha256(tmp)
    if actual != _EXPECTED_SHA256:
        tmp.unlink(missing_ok=True)
        raise ValueError(
            f"해시 불일치 — 예상 {_EXPECTED_SHA256}, 실제 {actual}. "
            "데이터셋이 갱신됐을 수 있음 — _HF_REVISION/_EXPECTED_SHA256를 huggingface.co에서 재확인할 것."
        )
    tmp.replace(dest)
    print(f"다운로드 완료·해시 검증됨: {dest}")


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
    억지로 통과시키는 건 불필요한 재작업일 뿐이다."""
    _TEXT_DIR.mkdir(parents=True, exist_ok=True)
    count = 0
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            chem_id = rec.get("chem_id", "")
            name = _safe_name(rec.get("name_ko", "unknown"))
            out_path = _TEXT_DIR / f"msds_{chem_id}_{name}.txt"
            out_path.write_text(_format_record(rec), encoding="utf-8")
            count += 1
            if count % 2000 == 0:
                print(f"  {count}건 적재...")
            if limit and count >= limit:
                break
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", default=str(_RAW_MSDS_DIR / "inconvenience-msds_train.jsonl"))
    parser.add_argument("--limit", type=int, default=None, help="테스트용 — 앞 N건만 적재")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl)
    if not args.skip_download:
        download(jsonl_path)

    count = ingest(jsonl_path, limit=args.limit)
    print(f"완료 — {count}건을 {_TEXT_DIR}에 적재")


if __name__ == "__main__":
    main()
