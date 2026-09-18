"""⚠️ 여전히 합성/버릴 데이터 생성기 — 단, 카테고리·측정물질은 사용자가 준 실제 센서 스펙을 반영함
(공기질/가스/조리흄, docs/의사결정_로그.md 18번).

"센서 이벤트 → 자연어 알림" 파인튜닝 배관(학습→MLflow 기록→평가)이 실제로 도는지 검증하는
용도다. 실제 알림 문구·정확한 임계값이 아니다.

⚠️ **임계값은 공식 기준이 아니다.** 산업안전보건법 작업환경기준·실내공기질 관리법 등 정식
기준치를 확인하지 않고 일러스트레이션 수준으로 채운 수치다. 실제 서비스에 쓰려면 반드시
정식 기준으로 교체해야 한다(특히 벤조에이피렌·헤테로사이클릭아민·오일미스트는 저자가
정확한 기준을 모른다고 명시하고 넘어간 항목).

템플릿 기반으로 결정론적으로 생성한다(LLM 호출 없음 — 비용 없고 디버깅 쉬움).
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sensors import CATEGORIES, LOCS, is_lower_is_worse  # noqa: E402 — 웹 시뮬레이터(web/app.py)와 정의 공유

_OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

_LOCS = LOCS
_CATEGORIES = [(name, table, action) for name, (table, action) in CATEGORIES.items()]


def _round(val: float) -> str:
    return f"{val:.1f}" if val < 10 else f"{val:.0f}"


def generate(n: int = 60, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    examples = []
    for i in range(n):
        cat_name, substances, action = rng.choice(_CATEGORIES)
        subst, unit, threshold, (lo, hi) = rng.choice(substances)
        # 산소농도처럼 '낮을수록 위험'한 물질(sensors.LOWER_IS_WORSE)은 위험 쪽 값이
        # threshold보다 낮은 쪽에 있어서, 어느 구간에서 80%를 뽑을지 방향을 뒤집어야 한다.
        inverted = is_lower_is_worse(subst)
        if rng.random() < 0.8:
            val = rng.uniform(lo, threshold) if inverted else rng.uniform(threshold, hi)
        else:
            val = rng.uniform(threshold, hi) if inverted else rng.uniform(lo, threshold)
        loc = rng.choice(_LOCS)
        exceeded = val < threshold if inverted else val > threshold

        user_msg = f"{loc} {cat_name}센서({subst}), 측정값 {_round(val)}{unit}, 임계값 {_round(threshold)}{unit}"
        if exceeded:
            assistant_msg = f"{loc}에서 {subst} 농도가 {_round(val)}{unit}로 임계값({_round(threshold)}{unit})을 초과했습니다. {action}"
        else:
            assistant_msg = f"{loc}의 {subst} 농도는 {_round(val)}{unit}로 임계값({_round(threshold)}{unit}) 이내입니다. 정상 범위입니다."

        examples.append({
            "messages": [
                {"role": "system", "content": "당신은 산업 현장 안전관리 보조 시스템입니다. 센서 이벤트를 받아 현장 담당자에게 보낼 간결한 한국어 알림 문장을 작성합니다."},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ],
            "kind": cat_name,
        })
    return examples


def main() -> None:
    examples = generate()
    rng = random.Random(7)
    rng.shuffle(examples)
    split = int(len(examples) * 0.8)
    train, val = examples[:split], examples[split:]

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in (("toy_sensor_alerts_train.jsonl", train), ("toy_sensor_alerts_val.jsonl", val)):
        path = _OUT_DIR / name
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{path} — {len(rows)}건")


if __name__ == "__main__":
    main()
