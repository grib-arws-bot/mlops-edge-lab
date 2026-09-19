"""국토지리정보원 「대한민국 국가지도집」 청소년판(2022년판) 커넥터 — API 인증 불필요.

이 사이트(nationalatlas.ngii.go.kr)는 오래된 CMS(OZ Homebuilder)라 페이지 ID가 순차
번호일 뿐 의미가 없고, 학년·단원별 목록을 안정적으로 긁어올 API가 없다. 그래서 "청소년판
2022년판" 메뉴(페이지 3852)를 실제로 열어서 확인한 다운로드 링크 7건을 직접 나열한다 —
매번 크롤링해서 목록을 재구성하지 않는 이유는, 이 사이트가 페이지 구조를 바꾸면 크롤러가
엉뚱한(예: 영문판) 콘텐츠를 잘못 가져올 위험이 실제로 있었기 때문이다(리서치 중 페이지
2408을 잘못 열어 영문 수업계획 문서를 가져올 뻔함). 목록이 바뀌면 이 리스트를 다시 확인해서
갱신한다.

HTTPS 인증서가 다른 호스트(stat.ngii.go.kr) 것으로 잘못 발급돼 있어(2026-09-19 확인)
HTTP로만 접근한다. 다운로드 URL의 파일명에 한글이 들어있어 반드시 URL 인코딩해야 한다 —
안 하면 "400 Bad request" HTML이 돌아오는데 파일 크기가 작아서 자칫 정상 PDF로 착각하기
쉽다(직접 겪음).
"""

from __future__ import annotations

import urllib.parse
import urllib.request
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from extract.core import extract_text

from collect.models import CollectedItem
from collect.registry import SourceConfig
from collect.storage import already_collected_urls, append_item, now_iso, save_raw_file

_BASE = "http://nationalatlas.ngii.go.kr/pages/download.php"
_DIR = "/nationalatlas/kor_hi_2022/pdf"

# (파일명, 대표 주제 — title에 쓰임)
_FILES = [
    ("2022_청소년판_서문.pdf", "서문"),
    ("2022_청소년판_1_지도읽기.pdf", "지도읽기"),
    ("2022_청소년판_2_세계속의대한민국.pdf", "세계 속의 대한민국"),
    ("2022_청소년판_3_우리국토의모습.pdf", "우리 국토의 모습"),
    ("2022_청소년판_4_우리국토의자연환경.pdf", "우리 국토의 자연환경"),
    ("2022_청소년판_5_우리국토의인문환경.pdf", "우리 국토의 인문환경"),
    ("2022_청소년판_6_부록.pdf", "부록"),
]


def _download_url(filename: str) -> str:
    return f"{_BASE}?dir={urllib.parse.quote(_DIR)}&file={urllib.parse.quote(filename)}"


def collect(config: SourceConfig) -> list[CollectedItem]:
    seen = already_collected_urls(config.source_id)
    new_items: list[CollectedItem] = []

    for filename, topic in _FILES:
        url = _download_url(filename)
        if url in seen:
            continue

        req = urllib.request.Request(url, headers={"User-Agent": "MLOps-Edge-Lab-edu-collector/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()

        if data[:4] != b"%PDF":
            raise RuntimeError(f"PDF가 아닌 응답을 받음(다운로드 URL 확인 필요): {url}")

        raw_path = save_raw_file(config.source_id, filename, data)
        result = extract_text(data, ".pdf")

        item = CollectedItem(
            source_id=config.source_id,
            school_level="중학교",
            subject="사회",
            sub_domain="지리",
            title=f"대한민국 국가지도집 청소년판(2022) — {topic}",
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
