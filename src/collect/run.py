"""수집 CLI — registry에 등록된 (학교급, 과목) 소스를 순회하며 구현된 커넥터만 실행한다.

미구현(API 키 대기 등) 소스는 건너뛰고 이유를 출력한다 — 조용히 빠지지 않고 "왜 안
돌았는지"가 항상 보이게 한다(품질 점검·헬스체크에서 이미 써온 원칙과 동일).
"""

from __future__ import annotations

import argparse

from collect.connectors import kdi_econ_edu, national_atlas
from collect.registry import sources_for

_CONNECTORS = {
    "nationalatlas_youth": national_atlas.collect,
    "kdi_econ_edu": kdi_econ_edu.collect,
}


def run(school_level: str, subject: str) -> None:
    configs = sources_for(school_level, subject)
    if not configs:
        print(f"등록된 소스 없음: {school_level} / {subject}")
        return

    for config in configs:
        if not config.implemented:
            print(f"건너뜀 — {config.name}: 미구현(사유: {config.notes})")
            continue

        connector = _CONNECTORS.get(config.source_id)
        if connector is None:
            print(f"건너뜀 — {config.name}: implemented=True인데 커넥터 함수가 등록 안 됨(코드 버그)")
            continue

        print(f"수집 시작 — {config.name}")
        new_items = connector(config)
        print(f"  신규 {len(new_items)}건 (라이선스: {config.license.license_type})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--school", default="중학교")
    parser.add_argument("--subject", default="사회")
    args = parser.parse_args()
    run(args.school, args.subject)


if __name__ == "__main__":
    main()
