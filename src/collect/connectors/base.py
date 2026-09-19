"""모든 커넥터가 따르는 공통 인터페이스 — collect()는 새로 수집한 CollectedItem만 반환한다
(이미 수집된 항목은 커넥터 내부에서 storage.already_collected_urls()로 걸러낸다)."""

from __future__ import annotations

from typing import Protocol

from collect.models import CollectedItem
from collect.registry import SourceConfig


class Connector(Protocol):
    def collect(self, config: SourceConfig) -> list[CollectedItem]:
        ...
