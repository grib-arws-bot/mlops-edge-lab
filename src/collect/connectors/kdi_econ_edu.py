"""KDI 경제정보센터 「교육과정별 경제교육」 커넥터 — API 인증 불필요.

⚠️ 이 소스는 공공누리 제3유형(변경금지)이다. `registry.py`의 license가 이미
allows_modification=False로 박혀 있으니, 이 커넥터가 수집한 항목은 RAG 인용에는
써도 되지만 파인튜닝 합성 데이터 생성 단계(Phase C)에서 반드시 제외해야 한다 —
그 강제는 이 커넥터가 아니라 Phase C 코드가 allows_modification 필드를 보고 해야 한다.

사이트에 브라우저 흉내 User-Agent가 없으면 WAF가 "정상적인 접근이 아닙니다" 에러
페이지를 돌려준다(직접 확인) — robots.txt는 전면 허용인데도 그렇다.

edu_gubun=M(중학교)/E(초등학교)/H(고등학교) 쿼리 파라미터로 학교급 필터링 가능함을
목록 페이지의 버튼 onclick에서 직접 확인. 상세 페이지의 첨부파일 링크 텍스트에 실제
파일명(확장자 포함)이 그대로 들어있어, 그걸로 파일 형식을 판단한다(URL 자체엔 확장자가
없음).
"""

from __future__ import annotations

import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from extract.core import extract_text

from collect.models import CollectedItem
from collect.registry import SourceConfig
from collect.storage import already_collected_urls, append_item, now_iso, save_raw_file

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}
_BASE = "https://eiec.kdi.re.kr"
_LIST_URL = f"{_BASE}/material/ecoEduList.do?edu_gubun=M&pp=100"
_DETAIL_URL = f"{_BASE}/material/ecoEduView.do?idx={{idx}}"

_SUPPORTED_SUFFIXES = {".pdf", ".hwp", ".hwpx", ".docx", ".doc"}


def _fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def _list_idxs() -> list[int]:
    html = _fetch(_LIST_URL).decode("utf-8", errors="replace")
    return sorted(set(int(m) for m in re.findall(r"ecoEduView\.do\?idx=(\d+)", html)))


def _attachments(detail_html: str) -> list[tuple[str, str]]:
    """(다운로드 경로, 원본 파일명) 쌍 목록. 파일명에서 실제 확장자를 얻는다 —
    다운로드 URL 자체엔 확장자가 없어서 이 방법 말고는 형식을 알 방법이 없다."""
    return re.findall(r'href="(/material/callDownload\.do\?[^"]*)">([^<]*)</a>', detail_html)


def collect(config: SourceConfig) -> list[CollectedItem]:
    seen = already_collected_urls(config.source_id)
    new_items: list[CollectedItem] = []

    for idx in _list_idxs():
        detail_html = _fetch(_DETAIL_URL.format(idx=idx)).decode("utf-8", errors="replace")

        for path, filename in _attachments(detail_html):
            url = f"{_BASE}{path}"
            if url in seen:
                continue

            suffix = Path(filename).suffix.lower()
            if suffix not in _SUPPORTED_SUFFIXES:
                continue  # 알 수 없는 형식은 건너뜀(추측해서 잘못 파싱하지 않음)

            data = _fetch(url)
            raw_path = save_raw_file(config.source_id, f"{idx}_{filename}", data)
            result = extract_text(data, suffix)

            item = CollectedItem(
                source_id=config.source_id,
                school_level="중학교",
                subject="사회",
                sub_domain="경제",
                title=Path(filename).stem,
                source_url=url,
                fetch_method="http_download",
                fetched_at=now_iso(),
                license=config.license,
                raw_path=raw_path,
                text_char_count=len(result.text),
            )
            append_item(item)
            new_items.append(item)

    return new_items
