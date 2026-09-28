"""AI Hub "화학물질 위험성 예측 데이터"(dataset 71816)를 받아 CAS 번호 기준
물성치 카탈로그(data/processed/aihub_chem_properties.json)로 병합한다.

msds_hf_ingest.py와 같은 역할(원본 확보 → 적재 → 갱신 확인)이지만 출처 성격이
달라 별도 모듈로 둔다:
  - HuggingFace는 리비전(커밋 sha)이라는 명확한 버전 신호가 있다. AI Hub
    aihubshell -mode l은 그런 게 없고, 파일 트리에 파일명·용량(MB 단위, 반올림)만
    나온다. 그래서 "갱신 확인"은 용량 시그니처 비교로 근사한다 — 파일 내용이
    바뀌어도 반올림된 MB 값이 같으면 감지하지 못하는 한계가 있다(실측 가능한
    선에서 최선, docs/의사결정_로그.md 참고).
  - 데이터셋마다 사전 다운로드 승인이 필요하다(API 키만으로는 안 됨) — 이미
    승인된 파일키만 재다운로드 가능하고, 새 파일키(예: 원천데이터)는 별도 승인
    필요.

**리터러시 데이터(라벨링데이터)만 받는다** — 원천데이터(SDF+PNG)는 계산된
물성치가 이미 라벨링 JSON의 common_properties에 다 들어있어(2026-09-28 실측,
샘플 1건 확인) 조회 기능에 불필요하다.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_RAW_DIR = _ROOT / "data" / "raw" / "aihub_chem_hazard"
_DATA_SUBPATH = _RAW_DIR / "57.화학물질_위험성_예측_데이터" / "3.개방데이터" / "1.데이터"
_LABELS_BYSRC_DIR = _ROOT / "data" / "processed" / "aihub_chem_labels_bysrc"
_PROPERTIES_PATH = _ROOT / "data" / "processed" / "aihub_chem_properties.json"
_SYNC_STATUS_PATH = _ROOT / "data" / "processed" / "aihub_chem_sync_status.json"
_SIGNATURE_PATH = _ROOT / "data" / "processed" / "aihub_chem_ingested_signature.json"
_AIHUBSHELL = _RAW_DIR / "aihubshell"
_API_KEY_ENV_PATH = Path.home() / "secrets" / "mlops-edge-lab" / "aihub_api_key.env"

_DATASET_KEY = "71816"
# (zip 안의 상대 경로, filekey) — 라벨링데이터 6개(Train/Val x 증기압/연소열/인화점)
_LABEL_FILEKEYS = {
    "Training/02.라벨링데이터/TL_1.화학물질 위험성 예측 데이터_1.증기압.zip": "553640",
    "Training/02.라벨링데이터/TL_1.화학물질 위험성 예측 데이터_2.연소열.zip": "553641",
    "Training/02.라벨링데이터/TL_1.화학물질 위험성 예측 데이터_3.인화점.zip": "553642",
    "Validation/02.라벨링데이터/VL_1.화학물질 위험성 예측 데이터_1.증기압.zip": "553646",
    "Validation/02.라벨링데이터/VL_1.화학물질 위험성 예측 데이터_2.연소열.zip": "553647",
    "Validation/02.라벨링데이터/VL_1.화학물질 위험성 예측 데이터_3.인화점.zip": "553648",
}

_COMMON_KEYS = [
    "iupac_name", "organic_compound", "cid", "cas_number", "smiles", "inchi",
    "molecular_formula", "mw", "h_donor", "h_acceptor", "density", "logp", "tpsa",
    "aromatic_ring", "bp", "mp", "refractive_index", "water_solubility",
    "dieletric_constant",
]
_PROPERTY_BLOCKS = {
    "vapor_pressure_properties": "vapor_pressure",
    "heat_of_combustion_properties": "heat_combustion",
    "flash_point_properties": "flash_point",
}

_TREE_LINE_PATTERN = re.compile(r"([^│├└─\n]+\.zip)\s*\|\s*(\d+)\s*MB\s*\|\s*(\d+)")


def _read_api_key() -> str:
    text = _API_KEY_ENV_PATH.read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.startswith("MLOPS_AIHUB_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise ValueError(f"MLOPS_AIHUB_API_KEY를 {_API_KEY_ENV_PATH}에서 찾지 못했습니다")


def _list_file_tree() -> dict[str, int]:
    """aihubshell -mode l로 현재 AI Hub의 파일 트리를 조회 — {filekey: 용량(MB)}.
    라벨링데이터 6개 파일키만 추린다(원천데이터는 우리가 안 쓰므로 시그니처에서도
    제외 — 바뀌어도 갱신 알림을 낼 이유가 없음)."""
    api_key = _read_api_key()
    proc = subprocess.run(
        [str(_AIHUBSHELL), "-mode", "l", "-datasetkey", _DATASET_KEY, "-aihubapikey", api_key],
        cwd=str(_RAW_DIR), capture_output=True, text=True, timeout=30, check=True,
    )
    sizes_by_key: dict[str, int] = {}
    for m in _TREE_LINE_PATTERN.finditer(proc.stdout):
        _name, size_mb, filekey = m.groups()
        sizes_by_key[filekey] = int(size_mb)
    wanted_keys = set(_LABEL_FILEKEYS.values())
    return {k: v for k, v in sizes_by_key.items() if k in wanted_keys}


def current_pinned_signature() -> dict[str, int]:
    if _SIGNATURE_PATH.exists():
        return json.loads(_SIGNATURE_PATH.read_text(encoding="utf-8"))
    return {}


def _record_ingested_signature(sig: dict[str, int]) -> None:
    _SIGNATURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _SIGNATURE_PATH.write_text(json.dumps(sig, ensure_ascii=False), encoding="utf-8")


def check_for_updates(write_status: bool = True) -> dict:
    """현재 AI Hub 파일 트리의 용량 시그니처를 우리가 마지막으로 받아둔 시그니처와
    비교. HuggingFace 리비전 비교와 달리 정확한 버전 식별이 아니라 근사치임을
    result.note에 명시한다 — 화면에서도 이 한계를 그대로 보여줘야 한다."""
    pinned = current_pinned_signature()
    result = {
        "checked_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "pinned_signature": pinned,
        "latest_signature": None,
        "up_to_date": None,
        "error": None,
        "note": "AI Hub는 HuggingFace처럼 커밋 리비전을 제공하지 않아, 용량(MB, 반올림) 변화로만 갱신 여부를 근사 판단합니다.",
    }
    try:
        latest = _list_file_tree()
        result["latest_signature"] = latest
        # 시그니처를 아직 한 번도 기록한 적이 없으면(최초 상태) "같다/다르다"를
        # 판단할 기준이 없다 — None(모름)이 사실과 다른 확신보다 정확하다.
        result["up_to_date"] = (latest == pinned) if pinned else None
    except Exception as e:  # noqa: BLE001 — 네트워크/승인 오류 등 무엇이든 상태에 남긴다
        result["error"] = str(e)

    if write_status:
        _SYNC_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        _SYNC_STATUS_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def download_labels() -> None:
    api_key = _read_api_key()
    filekeys = ",".join(_LABEL_FILEKEYS.values())
    subprocess.run(
        [str(_AIHUBSHELL), "-mode", "d", "-datasetkey", _DATASET_KEY,
         "-filekey", filekeys, "-aihubapikey", api_key],
        cwd=str(_RAW_DIR), check=True, timeout=1800,
    )


def _extract_labels() -> None:
    """6개 라벨링 zip을 각각 자기 이름의 하위 폴더로 압축 해제 — 하나의 폴더에
    다 풀면 파일명이 zip마다 겹쳐 덮어써질 위험이 있어(2026-09-28에 실측 확인,
    이번엔 실제로는 충돌 없었지만 재발 방지 차원에서 zip별로 분리) 이 구조를
    표준으로 삼는다."""
    for rel_path in _LABEL_FILEKEYS:
        zip_path = _DATA_SUBPATH / rel_path
        out_dir = _LABELS_BYSRC_DIR / zip_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.namelist():
                # aihubshell이 만드는 zip 안에는 종종 절대경로 스펙("/NN_...json")이
                # 있어 압축 해제 시 경로 앞부분을 벗겨내야 한다(실측, 2026-09-28).
                target_name = Path(member).name
                if not target_name:
                    continue
                with zf.open(member) as src, (out_dir / target_name).open("wb") as dst:
                    dst.write(src.read())


def _is_populated(measurements: list) -> bool:
    return any(v is not None for m in measurements for k, v in m.items() if k != "datasource")


def build_catalog() -> int:
    """압축 해제된 라벨 JSON 전체를 CAS 번호로 병합해 aihub_chem_properties.json을
    새로 쓴다. 같은 화합물이 여러 zip(증기압 zip, 인화점 zip 등)에 나뉘어
    등장하므로 CAS 단위로 합쳐야 한 화합물의 여러 물성치를 한 화면에서 보여줄 수
    있다."""
    by_cas: dict[str, dict] = {}
    for src_dir in sorted(_LABELS_BYSRC_DIR.iterdir()):
        if not src_dir.is_dir():
            continue
        for jf in src_dir.glob("*.json"):
            data = json.loads(jf.read_text(encoding="utf-8"))
            common = data.get("common_properties", {})
            cas = common.get("cas_number")
            if not cas or cas == "Not available":
                continue
            rec = by_cas.get(cas)
            if rec is None:
                rec = {k: common.get(k) for k in _COMMON_KEYS}
                rec["ghs_classification"] = common.get("ghs_classification")
                rec["properties"] = {}
                by_cas[cas] = rec
            for block_key, list_key in _PROPERTY_BLOCKS.items():
                measurements = (data.get(block_key) or {}).get(list_key) or []
                if not _is_populated(measurements):
                    continue
                existing = rec["properties"].setdefault(list_key, [])
                for m in measurements:
                    if m not in existing:
                        existing.append(m)

    out = sorted(by_cas.values(), key=lambda r: r["cas_number"])
    _PROPERTIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    _PROPERTIES_PATH.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return len(out)


def ingest() -> int:
    """다운로드 없이(이미 받아둔 zip으로) 압축 해제 + 카탈로그 재빌드만 수행 —
    update_job이 다운로드 단계와 상태를 분리해서 보여줄 수 있게 함수를 나눈다."""
    _extract_labels()
    return build_catalog()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-updates-only", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    if args.check_updates_only:
        print(json.dumps(check_for_updates(), ensure_ascii=False, indent=2))
        return

    if not args.skip_download:
        download_labels()
    count = ingest()
    _record_ingested_signature(_list_file_tree())
    check_for_updates()
    print(f"완료 — CAS 기준 {count}개 화합물을 {_PROPERTIES_PATH}에 적재")


if __name__ == "__main__":
    main()
