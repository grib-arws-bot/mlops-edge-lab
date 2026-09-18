# systemd --user 서비스

tmux는 사람이 그 세션에 실수로 키를 잘못 치면(예: `exit`, `Ctrl+C`) 바로 죽어버린다 —
실제로 이 프로젝트에서 `actions-runner`·`mlflow` 세션이 이렇게 두 번 죽었다(의사결정_로그
68번). `systemctl --user`는 sudo 없이 홈 디렉토리 권한만으로 등록 가능하고, `Restart=always`로
죽으면 자동 재시작되며, 사람이 터미널에 뭘 치는 것과 서비스 생명주기가 완전히 분리된다.

## 설치 (서버에서, sudo 불필요)

```bash
mkdir -p ~/.config/systemd/user
cp infra/systemd/*.service infra/systemd/*.timer ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now mlops-web mlops-mlflow mlops-actions-runner
systemctl --user enable --now mlops-healthcheck.timer
```

## 장애 알림 (Alertmanager-lite)

`mlops-healthcheck.timer`가 2분마다 `scripts/healthcheck_alert.py`를 실행해서 `mlops-web`·
`mlops-mlflow`·`mlops-actions-runner`의 systemd 상태와 web/MLflow의 HTTP 응답을 확인한다.
상태가 바뀔 때만(UP↔DOWN) `logs/alerts.log`에 기록한다. `ALERT_WEBHOOK_URL` 환경변수를
설정하면(예: `systemctl --user set-environment ALERT_WEBHOOK_URL=https://...`) 같은 내용을
그 웹훅으로도 보낸다 — 채널(Slack/Discord 등)이 정해지면 코드 수정 없이 바로 연결된다.

풀 Prometheus Alertmanager 대신 이 방식을 쓴 이유: MLflow·actions-runner처럼 `/metrics`를
노출 안 하는 서비스도 systemd 상태로는 바로 확인되고, 별도 컨테이너·설정 문법 없이 지금
규모에 필요한 만큼만 가볍게 해결된다.

## 로그 확인 (tmux capture-pane 대신)

```bash
journalctl --user -u mlops-web -f
journalctl --user -u mlops-mlflow -f
```

## `mlops-actions-runner`만 `RestartSec=60`인 이유

GitHub Actions 러너를 중단(재시작 포함)하면, GitHub 서버 쪽이 "이 러너의 이전 세션"을 완전히
해제하는 데 1~2분 정도 걸린다. 다른 서비스처럼 `RestartSec=5`로 두면 systemd가 세션이 채 안
풀린 상태에서 바로 재연결을 시도해 `A session for this runner already exists`(Conflict)
에러가 나고, 이게 반복돼서 8~16초마다 재시작되는 루프에 빠진다(실제로 겪음). `RestartSec=60`으로
늘려서 GitHub 쪽 세션이 풀릴 시간을 주면 깨끗하게 재연결된다 — systemd 자체의 문제가 아니라
GitHub 브로커의 세션 해제 지연 때문이었다.

## 재부팅에도 살아남게 하기 (linger)

기본적으로 `systemctl --user` 서비스는 그 유저의 SSH 세션이 전부 끊기면 같이 죽는다(로그아웃 시
유저 systemd 인스턴스 자체가 내려감). **로그아웃 이후에도, 서버 재부팅 이후에도** 계속 떠있게
하려면 linger를 켜야 한다:

```bash
loginctl enable-linger $(whoami)
```

이건 계정에 따라 sudo가 필요할 수도 있다(polkit 정책에 따라 다름) — 안 되면 서버 관리자에게
"내 계정에 linger를 켜달라"고 요청하면 된다.

## 기존 tmux 세션과의 관계

`deploy.sh`는 이제 `systemctl --user restart mlops-web`을 쓴다(예전엔 `tmux kill-session`+
`tmux new-session`). `mlflow`·`actions-runner`는 tmux 세션을 그대로 두고 systemd로 이전
가능 — 이전 후엔 기존 tmux 세션을 `tmux kill-session -t <이름>`으로 정리해도 된다(계속 떠있어도
포트 충돌만 없으면 무해하지만 헷갈리니 정리 권장).
