"""문서 텍스트 추출 — 형식별 폴백 사슬.

설계 원칙 (BidRadar `extract.py`에서 가져옴, docs/의사결정_로그.md 7번 참고):
- **폴백 사슬**: 형식마다 "가장 빠르고 정확한 방법"부터 시도하고, 실패하면 다음 방법으로.
- **조용한 빈 결과 금지**: 실패하면 무엇을 시도했고 왜 실패했는지 그대로 담아 반환한다.
  빈 문자열을 성공으로 위장하지 않는다.

레거시 포맷(HWP·DOC·XLS) 통합 전략: 포맷별 전용 파서를 따로 만드는 대신, **LibreOffice로
PDF 변환 후 이미 만들어둔 PDF 추출기를 재사용**한다. 이렇게 하면 "텍스트 레이어 없음 → OCR"
로직을 한 곳(PDF 경로)에만 두면 되고, 포맷이 늘어나도 변환 단계만 추가하면 된다.
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile

import pymupdf as fitz
import openpyxl
import pytesseract
from docx import Document
from PIL import Image

_HWPX_TEXT_TAG = "{http://www.hancom.co.kr/hwpml/2011/paragraph}t"

# 파일 시그니처(매직 바이트) — 확장자가 실제 내부 구조와 다른 경우(오표기)를 잡아내기 위함.
# BidRadar 팀이 실제로 이런 사례(.hwpx인데 구버전 OLE 구조, .hwp인데 실제론 zip 구조)를
# 발견해서 재분기 처리하고 있어, 같은 패턴을 그대로 가져온다.
_PK_SIG = b"PK\x03\x04"  # zip 계열(HWPX/DOCX/XLSX/PPTX)
_CFBF_SIG = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"  # 구버전 OLE 계열(HWP/DOC/XLS/PPT)

# PDF 텍스트 레이어가 있다고 판단할 최소 글자 수(페이지 평균). 이보다 적으면 스캔 이미지로
# 보고 OCR로 넘어간다 — 텍스트 레이어가 아예 없는 경우뿐 아니라, 표지/도장만 있는 페이지처럼
# "있긴 한데 의미 없는 수준"도 걸러내기 위함.
_PDF_MIN_CHARS_PER_PAGE = 20

# OCR 언어 — 한국어 산업안전 문서가 대상이라 한국어+영어(수치·약어) 동시 인식
_OCR_LANG = "kor+eng"


@dataclass
class ExtractResult:
    ok: bool
    method: str
    text: str = ""
    error: str | None = None
    attempted: list[str] = field(default_factory=list)


def _ocr_image(img: Image.Image) -> str:
    return pytesseract.image_to_string(img, lang=_OCR_LANG)


def _detect_signature(data: bytes) -> str:
    if data.startswith(_PK_SIG):
        return "zip"
    if data.startswith(_CFBF_SIG):
        return "ole"
    return "unknown"


def _effective_suffix(data: bytes, suffix: str) -> str:
    """확장자와 실제 내부 구조가 다르면(오표기) 실제 구조에 맞는 확장자로 바꿔서 반환한다."""
    suffix = suffix.lower()
    if suffix == ".hwpx" and _detect_signature(data) == "ole":
        return ".hwp"  # 오표기: 실제론 구버전 OLE 구조
    if suffix == ".hwp" and _detect_signature(data) == "zip":
        return ".hwpx"  # 오표기: 실제론 zip(HWPX) 구조
    return suffix


def _extract_hwpx(data: bytes) -> tuple[bool, str, str | None]:
    try:
        with ZipFile(BytesIO(data)) as zf:
            section_names = sorted(
                n for n in zf.namelist() if n.startswith("Contents/section") and n.endswith(".xml")
            )
            if not section_names:
                return False, "", "Contents/section*.xml을 찾을 수 없습니다(HWPX 구조가 아닙니다)"
            chunks: list[str] = []
            for name in section_names:
                root = ElementTree.fromstring(zf.read(name))
                chunks.extend(node.text or "" for node in root.iter(_HWPX_TEXT_TAG))
            text = "\n".join(c for c in chunks if c.strip())
            if not text.strip():
                return False, "", "본문 텍스트를 찾지 못했습니다(빈 문서이거나 태그 구조가 다릅니다)"
            return True, text, None
    except (BadZipFile, ElementTree.ParseError) as exc:
        return False, "", f"HWPX 파싱 실패: {exc}"


def _extract_pdf(data: bytes) -> tuple[bool, str, str | None]:
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:  # noqa: BLE001 — 손상 PDF에서 다양한 예외가 날 수 있음
        return False, "", f"PDF 열기 실패: {exc}"

    try:
        page_texts = [page.get_text() or "" for page in doc]
        text = "\n".join(t for t in page_texts if t.strip())
        avg_chars = len(text) / max(len(doc), 1)
        if avg_chars < _PDF_MIN_CHARS_PER_PAGE:
            return False, "", (
                f"텍스트 레이어가 부족합니다(페이지당 평균 {avg_chars:.0f}자) — 스캔 이미지 PDF로 추정, OCR 필요"
            )
        return True, text, None
    finally:
        doc.close()


def _extract_pdf_ocr(data: bytes) -> tuple[bool, str, str | None]:
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:  # noqa: BLE001
        return False, "", f"PDF 열기 실패(OCR 단계): {exc}"

    try:
        chunks: list[str] = []
        for page in doc:
            pix = page.get_pixmap(dpi=200)
            img = Image.open(BytesIO(pix.tobytes("png")))
            chunks.append(_ocr_image(img))
        text = "\n".join(c for c in chunks if c.strip())
        if not text.strip():
            return False, "", "OCR로도 텍스트를 찾지 못했습니다"
        return True, text, None
    except Exception as exc:  # noqa: BLE001 — tesseract 미설치 등
        return False, "", f"OCR 실패: {exc}"
    finally:
        doc.close()


def _extract_hwp_pyhwp(data: bytes) -> tuple[bool, str, str | None]:
    """구버전 바이너리 HWP(OLE 구조) — pyhwp(`hwp5txt`)로 추출한다.

    LibreOffice 변환 대신 이걸 1차로 쓰는 이유: BidRadar가 실제 문서로 검증해본 결과
    pyhwp만으로 LibreOffice 없이 완전히 동작했다(BidRadar 문서 추출 현황 참고). 더 가볍고
    이미 검증된 경로라 우선 시도하고, 실패하면 LibreOffice를 2차로 둔다.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        in_path = tmp_path / "input.hwp"
        in_path.write_bytes(data)
        out_path = tmp_path / "output.txt"
        try:
            subprocess.run(
                ["hwp5txt", str(in_path), "--output", str(out_path)],
                capture_output=True, timeout=60, check=True,
            )
        except FileNotFoundError:
            return False, "", "pyhwp(hwp5txt)가 설치되어 있지 않습니다"
        except subprocess.TimeoutExpired:
            return False, "", "hwp5txt 변환 타임아웃(60s)"
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
            return False, "", f"hwp5txt 변환 실패: {stderr[:300]}"

        if not out_path.exists():
            return False, "", "hwp5txt가 출력 파일을 생성하지 않았습니다"
        text = out_path.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            return False, "", "본문 텍스트를 찾지 못했습니다(빈 문서)"
        return True, text, None


def _extract_txt(data: bytes) -> tuple[bool, str, str | None]:
    """인코딩을 알 수 없는 일반 텍스트 — UTF-8 → CP949 → EUC-KR 순서로 시도한다."""
    for enc in ("utf-8", "cp949", "euc-kr"):
        try:
            text = data.decode(enc)
        except UnicodeDecodeError:
            continue
        if text.strip():
            return True, text, None
    return False, "", "UTF-8/CP949/EUC-KR 인코딩으로 디코딩하지 못했습니다"


def _extract_docx(data: bytes) -> tuple[bool, str, str | None]:
    try:
        doc = Document(BytesIO(data))
        parts = [p.text for p in doc.paragraphs]
        for table in doc.tables:
            for row in table.rows:
                parts.extend(cell.text for cell in row.cells)
        text = "\n".join(p for p in parts if p.strip())
        if not text.strip():
            return False, "", "본문 텍스트를 찾지 못했습니다(빈 문서)"
        return True, text, None
    except Exception as exc:  # noqa: BLE001 — 손상 파일 등
        return False, "", f"DOCX 파싱 실패: {exc}"


def _extract_xlsx(data: bytes) -> tuple[bool, str, str | None]:
    try:
        wb = openpyxl.load_workbook(BytesIO(data), data_only=True, read_only=True)
        lines: list[str] = []
        for sheet in wb.worksheets:
            lines.append(f"# {sheet.title}")
            for row in sheet.iter_rows(values_only=True):
                cells = [str(c) for c in row if c is not None]
                if cells:
                    lines.append("\t".join(cells))
        text = "\n".join(lines)
        if not text.strip():
            return False, "", "본문 텍스트를 찾지 못했습니다(빈 워크북)"
        return True, text, None
    except Exception as exc:  # noqa: BLE001
        return False, "", f"XLSX 파싱 실패: {exc}"


def _extract_image(data: bytes) -> tuple[bool, str, str | None]:
    try:
        img = Image.open(BytesIO(data))
        text = _ocr_image(img)
        if not text.strip():
            return False, "", "OCR로 텍스트를 찾지 못했습니다"
        return True, text, None
    except Exception as exc:  # noqa: BLE001
        return False, "", f"이미지 OCR 실패: {exc}"


def _libreoffice_to_pdf(data: bytes, suffix: str, timeout: int = 120) -> tuple[bool, bytes | None, str | None]:
    """레거시 포맷(HWP/DOC/XLS 등)을 LibreOffice headless로 PDF 변환한다.

    포맷별 전용 파서 대신 이 방법을 쓰는 이유: HWP 같은 구형 바이너리 포맷은 신뢰할 만한
    순수 파이썬 라이브러리가 없다. LibreOffice 하나로 여러 레거시 포맷을 동일한 방식으로
    처리할 수 있어 유지보수 대상이 줄어든다.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        in_path = tmp_path / f"input{suffix}"
        in_path.write_bytes(data)
        try:
            subprocess.run(
                [
                    "soffice", "--headless", "--norestore",
                    "--convert-to", "pdf", "--outdir", str(tmp_path), str(in_path),
                ],
                capture_output=True, timeout=timeout, check=True,
            )
        except FileNotFoundError:
            return False, None, "LibreOffice(soffice)가 설치되어 있지 않습니다"
        except subprocess.TimeoutExpired:
            return False, None, f"LibreOffice 변환 타임아웃({timeout}s)"
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
            return False, None, f"LibreOffice 변환 실패: {stderr[:300]}"

        out_path = tmp_path / "input.pdf"
        if not out_path.exists():
            return False, None, "LibreOffice가 PDF를 생성하지 않았습니다"
        return True, out_path.read_bytes(), None


def _extract_via_libreoffice(data: bytes, suffix: str) -> tuple[bool, str, str | None]:
    ok, pdf_bytes, error = _libreoffice_to_pdf(data, suffix)
    if not ok:
        return False, "", error
    ok, text, error = _extract_pdf(pdf_bytes)
    if ok:
        return True, text, None
    # 변환은 됐는데 텍스트 레이어가 없는 경우(예: 스캔본을 hwp로 감싼 경우) — OCR까지 시도
    ok, text, error = _extract_pdf_ocr(pdf_bytes)
    return ok, text, error


# 확장자별 시도 순서. (method_name, fn) 튜플의 리스트 — 앞에서부터 시도.
# HWP: pyhwp(hwp5txt)를 1차로, LibreOffice를 2차 안전망으로 (근거는 위 함수 docstring 참고)
_DISPATCH: dict[str, list[tuple[str, object]]] = {
    ".hwpx": [
        ("hwpx", _extract_hwpx),
        ("hwp_pyhwp", _extract_hwp_pyhwp),
        ("libreoffice", lambda d: _extract_via_libreoffice(d, ".hwpx")),
    ],
    ".hwp": [
        ("hwp_pyhwp", _extract_hwp_pyhwp),
        ("libreoffice", lambda d: _extract_via_libreoffice(d, ".hwp")),
    ],
    ".pdf": [("pdf_text", _extract_pdf), ("pdf_ocr", _extract_pdf_ocr)],
    ".doc": [("libreoffice", lambda d: _extract_via_libreoffice(d, ".doc"))],
    ".docx": [("docx", _extract_docx), ("libreoffice", lambda d: _extract_via_libreoffice(d, ".docx"))],
    ".xls": [("libreoffice", lambda d: _extract_via_libreoffice(d, ".xls"))],
    ".xlsx": [("xlsx", _extract_xlsx), ("libreoffice", lambda d: _extract_via_libreoffice(d, ".xlsx"))],
    ".txt": [("txt", _extract_txt)],
}
for _img_ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"):
    _DISPATCH[_img_ext] = [("image_ocr", _extract_image)]


def extract_text(data: bytes, suffix: str) -> ExtractResult:
    """확장자에 맞는 단계부터 순서대로 시도한다. 전부 실패하면 시도한 단계 목록과 마지막
    오류를 그대로 반환한다 — 빈 문자열을 성공으로 위장하지 않는다.

    확장자만 보지 않고 파일 시그니처로 재확인한다(`_effective_suffix`) — HWP/HWPX는
    확장자가 실제 내부 구조와 다르게 붙어 있는 경우가 실제로 있다(오표기).
    """
    suffix = suffix.lower()
    effective = _effective_suffix(data, suffix)
    stages = _DISPATCH.get(effective, [])
    attempted: list[str] = []
    if effective != suffix:
        attempted.append(f"reroute:{suffix}->{effective}")
    last_error: str | None = None
    for method, fn in stages:
        attempted.append(method)
        ok, text, error = fn(data)
        if ok:
            return ExtractResult(ok=True, method=method, text=text, attempted=attempted)
        last_error = error

    return ExtractResult(
        ok=False,
        method=attempted[-1] if attempted else "unsupported",
        error=last_error or f"지원하지 않는 확장자입니다: {suffix}",
        attempted=attempted,
    )
