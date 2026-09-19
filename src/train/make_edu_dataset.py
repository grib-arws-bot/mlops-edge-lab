"""교육(중학교 사회과) 파인튜닝용 합성 QA 데이터 생성.

`make_toy_dataset.py`(안전 도메인)와 같은 원칙 — 원문(공문서체)을 그대로 학습시키지
않고, "학생이 물어볼 법한 질문 + 중학생 눈높이 답변"으로 재작성한 합성 데이터를 만든다.
차이점: toy_sensor_alerts는 결정론적 템플릿(LLM 호출 없음)이지만, 여기는 실제 산문
텍스트가 원료라 재작성 자체를 LLM에 시켜야 한다.

**라이선스 강제 지점**: 이 재작성이 정확히 "변경"이므로, 수집 항목의
`allows_modification=False`(예: KDI, 공공누리 3유형)는 이 스크립트가 하드하게
제외한다 — 소스를 하드코딩하지 않고 매 항목의 메타데이터를 직접 읽고 판단하므로,
법령정보센터 등 인증키 대기 중인 소스가 나중에 들어와도 코드 변경 없이 자동 반영된다
(의사결정_로그 79번, 사용자·법적 검토 원칙).

지금은 nationalatlas_youth(공공누리 1유형) 7건만 조건을 만족한다 — 소량이라
sample_size=60(toy_sensor_alerts와 동일 규모)으로 "배관이 실제로 도는가"부터
검증하고, 라이선스가 깨끗한 소스가 늘어나면 그때 규모를 키운다.
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

from llama_cpp import Llama

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extract.core import extract_text
from rag.chunk import chunk_text

_ROOT = Path(__file__).resolve().parents[2]
_COLLECTED_DIR = _ROOT / "edu" / "collected"
_OUT_DIR = _ROOT / "data" / "processed"
_GGUF_PATH = _ROOT / "experiments" / "toy-sensor-lora" / "model-f16.gguf"

_SYSTEM_PROMPT = "당신은 중학교 사회 선생님입니다. 학생 질문에 학생 눈높이로 친절하게 답합니다."

_GEN_INSTRUCTION = """다음은 교육 자료의 한 부분입니다. 이 내용만 근거로, 중학생이 물어볼 법한
질문 하나와 그에 대한 답변을 만들어주세요. 답변은 이 자료에 없는 내용을 지어내지 말고,
중학생이 이해하기 쉬운 문장으로 2~4문장으로 작성하세요.

반드시 아래 JSON 형식으로만 답하세요(다른 말은 붙이지 마세요):
{{"question": "...", "answer": "..."}}

[자료]
{passage}
"""


def _load_modifiable_chunks(source_ids: list[str] | None = None) -> list[tuple[str, str]]:
    """(sub_domain, chunk) 목록 — allows_modification=True인 수집 항목의 청크만.

    source_ids를 주면 그 소스들로 제한한다 — 운영 콘솔에서 사용자가 소스를 골라
    학습에 포함시키는 기능(2026-09-19) 때문에 추가. None이면 기존처럼 전체."""
    chunks: list[tuple[str, str]] = []
    if not _COLLECTED_DIR.exists():
        return chunks

    for source_dir in sorted(_COLLECTED_DIR.iterdir()):
        if source_ids is not None and source_dir.name not in source_ids:
            continue
        items_path = source_dir / "items.jsonl"
        if not items_path.exists():
            continue
        for line in items_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if not item.get("allows_modification"):
                continue
            raw_path = _COLLECTED_DIR / item["raw_path"]
            data = raw_path.read_bytes()
            result = extract_text(data, raw_path.suffix)
            for chunk in chunk_text(result.text):
                if len(chunk) >= 200:  # 목차·짧은 표 등은 질문거리가 안 됨
                    chunks.append((item["sub_domain"], chunk))
    return chunks


def _parse_response(text: str) -> tuple[str, str] | None:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    question, answer = obj.get("question", "").strip(), obj.get("answer", "").strip()
    return (question, answer) if question and answer else None


def generate(sample_size: int = 60, seed: int = 42, source_ids: list[str] | None = None) -> list[dict]:
    chunks = _load_modifiable_chunks(source_ids)
    if not chunks:
        raise RuntimeError("파인튜닝 가능(allows_modification=True) 수집 데이터가 없음 — src/collect/run.py 먼저 실행 필요")

    rng = random.Random(seed)
    sample = rng.sample(chunks, min(sample_size, len(chunks)))

    llm = Llama(model_path=str(_GGUF_PATH), n_ctx=4096, n_threads=8, verbose=False)

    examples = []
    skipped = 0
    for sub_domain, passage in sample:
        result = llm.create_chat_completion(
            messages=[{"role": "user", "content": _GEN_INSTRUCTION.format(passage=passage)}],
            temperature=0.7, max_tokens=400,
        )
        parsed = _parse_response(result["choices"][0]["message"]["content"])
        if parsed is None:
            skipped += 1
            continue
        question, answer = parsed
        examples.append({
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
            "kind": sub_domain,
        })

    print(f"생성 {len(examples)}건, JSON 파싱 실패로 제외 {skipped}건 (후보 청크 {len(chunks)}개 중 {len(sample)}개 샘플링)")
    return examples


def build_and_save(
    sample_size: int = 60,
    source_ids: list[str] | None = None,
    train_name: str = "edu_social_train.jsonl",
    val_name: str = "edu_social_val.jsonl",
) -> tuple[int, int]:
    """generate() 결과를 8:2로 나눠 저장하고 (train건수, val건수)를 반환한다.
    운영 콘솔(웹)의 학습 작업(run_edu_training_job.py)이 소스를 골라 호출할 때도
    이 함수를 그대로 쓴다 — CLI(main)와 웹 트리거가 같은 경로를 타야 동작이 갈리지 않는다."""
    examples = generate(sample_size=sample_size, source_ids=source_ids)
    rng = random.Random(7)
    rng.shuffle(examples)
    split = int(len(examples) * 0.8)
    train, val = examples[:split], examples[split:]

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in ((train_name, train), (val_name, val)):
        path = _OUT_DIR / name
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{path} — {len(rows)}건")

    return len(train), len(val)


def main() -> None:
    build_and_save()


if __name__ == "__main__":
    main()
