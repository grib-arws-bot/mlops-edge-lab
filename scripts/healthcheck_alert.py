#!/usr/bin/env python3
"""경량 장애 감지("Alertmanager-lite") — 풀 Prometheus Alertmanager 스택 대신, systemd
서비스 상태를 직접 물어보고 상태가 바뀔 때만(UP<->DOWN) 기록한다.

오늘 실제로 겪은 사고(MLflow tmux 세션이 죽은 채 아무도 모름, actions-runner가 47분간
죽어있었음)를 계기로 만든다 — "모니터링이 있다"고 해도 사람이 안 보고 있으면 무용지물이라는
걸 직접 겪었다. systemd 상태를 직접 물어보는 이유: HTTP 헬스체크보다 더 정확하다(프로세스가
아예 없는지, 응답만 없는지 구분됨) — actions-runner처럼 웹 엔드포인트가 없는 서비스도
커버된다.

2분마다 systemd 타이머(infra/systemd/mlops-healthcheck.timer)로 실행된다. 웹훅 URL은
아직 미정이라(사용자 결정, 2026-09-18) 지금은 로컬 로그(logs/alerts.log)에만 남기고,
ALERT_WEBHOOK_URL 환경변수가 설정되면 자동으로 그쪽에도 보내도록 만들어뒀다 — 나중에
채널이 정해지면 코드 변경 없이 바로 연결된다.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_STATE_PATH = _ROOT / ".health_state.json"
_LOG_PATH = _ROOT / "logs" / "alerts.log"
_WEBHOOK_URL = os.environ.get("ALERT_WEBHOOK_URL")

_SYSTEMD_UNITS = ["mlops-web", "mlops-mlflow", "mlops-actions-runner"]
_HTTP_CHECKS = {
    "web": "http://127.0.0.1:8081/",
    "mlflow": "http://127.0.0.1:8082/mlflow/",
}


def _systemd_active(unit: str) -> bool:
    result = subprocess.run(
        ["systemctl", "--user", "is-active", unit], capture_output=True, text=True, timeout=10,
    )
    return result.stdout.strip() == "active"


def _http_ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return resp.status < 500
    except Exception:
        return False


def _check_all() -> dict[str, bool]:
    result: dict[str, bool] = {}
    for unit in _SYSTEMD_UNITS:
        result[f"systemd:{unit}"] = _systemd_active(unit)
    for name, url in _HTTP_CHECKS.items():
        result[f"http:{name}"] = _http_ok(url)
    return result


def _load_state() -> dict:
    if _STATE_PATH.exists():
        try:
            return json.loads(_STATE_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def _save_state(state: dict) -> None:
    _STATE_PATH.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")


def _notify(message: str) -> None:
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with _LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(f"{ts} {message}\n")
    print(f"{ts} {message}")
    if _WEBHOOK_URL:
        try:
            payload = json.dumps({"text": message}).encode("utf-8")
            req = urllib.request.Request(
                _WEBHOOK_URL, data=payload, headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception as exc:  # noqa: BLE001 — 알림 실패로 헬스체크 자체가 죽으면 안 됨
            print(f"웹훅 전송 실패: {exc}")


def main() -> None:
    current = _check_all()
    previous = _load_state()

    for key, is_up in current.items():
        # 처음 실행되는 항목은 "원래 정상이었다"고 가정 — 재부팅 직후 등 상태 파일이 없을 때
        # 모든 서비스를 "방금 복구됨"으로 오탐하지 않기 위함.
        was_up = previous.get(key, True)
        if was_up and not is_up:
            _notify(f"DOWN {key}")
        elif not was_up and is_up:
            _notify(f"UP {key} (복구됨)")

    _save_state(current)


if __name__ == "__main__":
    main()
