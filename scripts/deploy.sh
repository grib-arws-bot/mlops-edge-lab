#!/bin/bash
# AI 서버에서 최신 코드를 받아 웹 서비스를 재시작한다. 지금까지 scp로 하나씩 복사하던
# 걸 git pull로 통일 — CI/CD(자체 호스팅 러너)가 이 스크립트를 그대로 호출한다.
set -euo pipefail
cd ~/mlops-edge-lab

git pull origin main

tmux kill-session -t web 2>/dev/null || true
tmux new-session -d -s web 'cd ~/mlops-edge-lab && source .venv/bin/activate && PYTHONPATH=src uvicorn web.app:app --host 0.0.0.0 --port 8081'

# 고정 sleep 대신 최대 60초 폴링 — LLM 로딩 + (필요 시) 교육자료 PDF 변환까지 걸리는
# 시간이 매번 달라서, 짧은 고정 대기로는 실제로는 성공했는데 헬스체크만 실패로 오판하는
# 경우가 있었다(/education 추가로 soffice 변환이 붙은 뒤 실측).
for _ in $(seq 1 30); do
  if curl -sf -o /dev/null http://127.0.0.1:8081/; then
    echo "배포 성공: $(git rev-parse --short HEAD)"
    exit 0
  fi
  sleep 2
done

echo "배포 후 헬스체크 실패(60초 초과) — tmux attach -t web 으로 로그 확인 필요" >&2
exit 1
