"""data/raw 밑의 파일들을 순회한다. ZIP은 재귀적으로 풀어서 내부 파일까지 후보로 낸다.

**한글 zip 파일명 인코딩 문제**: Windows 탐색기로 만든 zip은 파일명을 CP949로 인코딩하는데,
zip 표준의 UTF-8 플래그(flag_bits bit 11)를 세팅하지 않는 경우가 많다. 이 플래그가 없으면
파이썬 zipfile은 파일명을 CP437로 잘못 해석한다 — 그 결과를 CP437로 다시 인코딩해 원래
바이트를 복원한 뒤 CP949로 재해석하면 원래 한글 파일명이 나온다(실제로 이번 데이터에서
발생한 문제 — `unzip -l` 결과가 깨져 나온 걸 보고 확인함).
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from zipfile import ZipFile, ZipInfo


def fix_zip_filename(info: ZipInfo) -> str:
    if info.flag_bits & 0x800:  # UTF-8로 정상 인코딩된 경우
        return info.filename
    try:
        return info.filename.encode("cp437").decode("cp949")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return info.filename  # 복구 실패 — 깨진 이름이라도 그대로 사용(중단하지 않음)


def walk_zip(data: bytes, prefix: str = ""):
    """zip 내부 파일을 (경로, 바이트) 튜플로 재귀적으로 낸다. 중첩 zip도 풀어서 들어간다."""
    with ZipFile(BytesIO(data)) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = fix_zip_filename(info)
            member_bytes = zf.read(info)
            full_name = f"{prefix}{name}"
            if Path(name).suffix.lower() == ".zip":
                yield from walk_zip(member_bytes, prefix=f"{full_name}/")
            else:
                yield full_name, member_bytes


def iter_source_files(raw_dir: Path):
    """raw_dir 바로 아래 파일들을 순회. zip이면 내부까지 풀어서 낸다."""
    for path in sorted(raw_dir.iterdir()):
        if path.is_dir():
            continue
        data = path.read_bytes()
        if path.suffix.lower() == ".zip":
            for member_name, member_bytes in walk_zip(data):
                yield f"{path.name}/{member_name}", member_bytes
        else:
            yield path.name, data
