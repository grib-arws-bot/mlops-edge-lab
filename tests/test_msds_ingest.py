"""collect/msds_hf_ingest.py의 순수 포맷팅 로직에 대한 최소 단위 테스트 —
외부 의존성(네트워크 다운로드) 없이 _format_record만 검증한다. 샘플 레코드는
2026-09-23 huggingface.co/datasets/Yuyongkim/inconvenience-msds에서 실제로
받아본 응답의 첫 레코드(chem_id 000001)를 그대로 옮겨온 것."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from collect import msds_hf_ingest
from collect.msds_hf_ingest import _format_record, _safe_name

_SAMPLE = {
    "chem_id": "000001",
    "name_ko": "염산 구아니딘 (Guanidinium chloride)",
    "cas_no": "50-01-1",
    "name_en": "Guanidine Hydrochloride",
    "sections": [
        {"section_no": 1, "title": "화학제품과 회사에 관한 정보", "text_ko": "제품명: 염산 구아니딘", "braille": "..."},
        {"section_no": 2, "title": "유해성·위험성", "text_ko": "유해성·위험성 분류: 급성 독성(경구) : 구분4", "braille": "..."},
        {"section_no": 3, "title": "구성성분의 명칭 및 함유량", "text_ko": "", "braille": "..."},  # 빈 섹션
    ],
}


def test_format_record_includes_identity_header():
    out = _format_record(_SAMPLE)
    assert "염산 구아니딘" in out
    assert "50-01-1" in out
    assert "000001" in out


def test_format_record_includes_nonempty_sections_in_order():
    out = _format_record(_SAMPLE)
    idx1 = out.index("[1. 화학제품과 회사에 관한 정보]")
    idx2 = out.index("[2. 유해성·위험성]")
    assert idx1 < idx2
    assert "급성 독성" in out


def test_format_record_skips_empty_section_text():
    out = _format_record(_SAMPLE)
    assert "[3. 구성성분의 명칭 및 함유량]" not in out


def test_format_record_handles_missing_sections_list():
    out = _format_record({"chem_id": "999999", "name_ko": "테스트물질", "cas_no": None, "name_en": ""})
    assert "테스트물질" in out
    assert "(없음)" in out  # cas_no 없을 때


def test_safe_name_strips_slashes_and_parens_kept():
    assert "/" not in _safe_name("a/b(c)")
    assert "\\" not in _safe_name("a\\b")


def test_safe_name_bounds_utf8_byte_length():
    # 2026-09-23 실측 버그: 문자 수만 자르면 한글(3바이트/글자)이 파일시스템의
    # 255바이트 파일명 제한을 넘길 수 있었다 — 바이트 단위로 잘려야 한다.
    long_korean = "가" * 200
    safe = _safe_name(long_korean, max_bytes=80)
    assert len(safe.encode("utf-8")) <= 80


def test_ingest_writes_text_files_and_catalog(tmp_path, monkeypatch):
    text_dir = tmp_path / "text"
    catalog_path = tmp_path / "msds_catalog.json"
    monkeypatch.setattr(msds_hf_ingest, "_TEXT_DIR", text_dir)
    monkeypatch.setattr(msds_hf_ingest, "_CATALOG_PATH", catalog_path)

    jsonl_path = tmp_path / "sample.jsonl"
    jsonl_path.write_text(
        json.dumps(_SAMPLE, ensure_ascii=False) + "\n"
        + json.dumps({"chem_id": "000002", "name_ko": "물질2", "cas_no": "1-1-1", "name_en": "Chem2", "sections": []}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    count = msds_hf_ingest.ingest(jsonl_path)

    assert count == 2
    written = list(text_dir.glob("msds_*.txt"))
    assert len(written) == 2
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    assert len(catalog) == 2
    assert catalog[0]["chem_id"] == "000001"
    assert catalog[0]["cas_no"] == "50-01-1"
