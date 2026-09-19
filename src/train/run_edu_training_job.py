"""운영 콘솔(웹)의 "학습 시작" 버튼이 서브프로세스로 띄우는 작업.

왜 서브프로세스인가: 웹 서비스(src/web/app.py)는 이미 인프로세스로 llama_cpp GPU
모델을 하나 띄워놓고 있다(/simulate, /control-room용). 여기서 transformers로 또
다른 모델을 GPU에 올려 학습시키면 그 프로세스와 메모리·CUDA 컨텍스트를 다투게 되고,
학습은 몇 초~몇 분씩 걸려서 웹 요청 핸들러 안에서 동기로 돌리면 타임아웃난다.
그래서 완전히 별도 프로세스로 띄우고, 웹은 상태 파일(job_status.json)만 폴링한다
— .health_state.json(healthcheck_alert.py)과 같은 패턴.

단계마다 상태 파일을 갱신해서, 운영 콘솔이 "지금 뭘 하고 있는지"를 보여줄 수 있게 한다.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_STATUS_PATH = _ROOT / "experiments" / "edu-social-lora" / "job_status.json"


def _write_status(**fields) -> None:
    _STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    current = {}
    if _STATUS_PATH.exists():
        try:
            current = json.loads(_STATUS_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            current = {}
    current.update(fields)
    current["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _STATUS_PATH.write_text(json.dumps(current, ensure_ascii=False), encoding="utf-8")


def run(source_ids: list[str], allow_restricted: bool = False) -> None:
    from train.evaluate import run_eval
    from train.finetune_lora import run_finetune
    from train.make_edu_dataset import build_and_save

    # started_at_epoch은 운영 콘솔이 "학습 중" 동안 경과 시간을 카운팅해서 보여주는
    # 데 쓴다(사용자 요청, 2026-09-19) — time.time()(UTC epoch)을 쓰는 이유: 로컬시간
    # 문자열(time.strftime)을 브라우저에서 파싱하면 서버 타임존을 UTC로 잘못 해석해
    # 몇 시간씩 어긋날 수 있는데, epoch 숫자는 타임존 자체가 없어 그 문제가 없다.
    started_at_epoch = time.time()
    _write_status(
        status="running", stage="generating_dataset", sources=source_ids, error=None, result=None,
        started_at_epoch=started_at_epoch, elapsed_seconds=None, progress=None,
    )

    def _progress(done: int, total: int) -> None:
        # 사용자 지적(2026-09-19) — 데이터 생성이 몇 분씩 걸리는데 진행률이 하나도
        # 안 보여서 멈춘 것처럼 보였다. 매 항목마다 상태 파일을 갱신해 몇 번째인지 보여준다.
        _write_status(progress=f"{done}/{total}")

    try:
        train_n, val_n = build_and_save(source_ids=source_ids, allow_restricted=allow_restricted, progress_callback=_progress)

        _write_status(stage="finetuning", train_examples=train_n, val_examples=val_n, progress=None)
        adapter_dir = run_finetune(
            num_train_epochs=5,
            run_name="edu-social-lora",
            output_subdir="edu-social-lora",
            train_file="edu_social_train.jsonl",
            val_file="edu_social_val.jsonl",
        )

        _write_status(stage="evaluating")
        score = run_eval(
            adapter_dir=adapter_dir,
            run_name="edu-social-lora-eval",
            val_path=_ROOT / "data" / "processed" / "edu_social_val.jsonl",
            out_name="eval_edu_social.jsonl",
        )

        # elapsed_seconds를 서버가 직접 계산해서 최종 상태에 남긴다(사용자 지적,
        # 2026-09-19) — 데이터셋이 작으면 학습이 프론트 폴링 주기(3초)보다 빨리
        # 끝나서 "진행 중" 상태를 한 번도 못 보고 바로 "완료"로 건너뛸 수 있는데,
        # 그러면 클라이언트 쪽 카운팅 타이머가 아예 시작을 못 해 경과 시간이 계속
        # 빈 채로 남는다 — 서버가 실측한 총 소요 시간을 최종 상태에 직접 넣어주면
        # 프론트가 타이머를 못 돌렸어도 최종 값은 항상 정확히 보여줄 수 있다.
        _write_status(
            status="done", stage="done", elapsed_seconds=round(time.time() - started_at_epoch, 1),
            result={"avg_rougeL": score, "train_examples": train_n, "val_examples": val_n, "adapter_dir": str(adapter_dir)},
        )
    except Exception as exc:  # noqa: BLE001 — 운영 콘솔에 그대로 보여주기 위해 잡아서 기록
        _write_status(
            status="error", stage="error", elapsed_seconds=round(time.time() - started_at_epoch, 1),
            error=f"{exc}\n{traceback.format_exc()[-2000:]}",
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", required=True, help="쉼표로 구분한 source_id 목록")
    parser.add_argument(
        "--allow-restricted", action="store_true",
        help="라이선스상 변경(파인튜닝) 금지 소스도 포함 — 웹 콘솔의 명시적 동의를 거친 경우에만 전달됨(2026-09-19)",
    )
    args = parser.parse_args()
    run(args.sources.split(","), allow_restricted=args.allow_restricted)


if __name__ == "__main__":
    main()
