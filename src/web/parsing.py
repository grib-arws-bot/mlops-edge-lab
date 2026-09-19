"""app.py가 쓰는 순수 파싱/판정 로직 모음.

LLM·임베딩 모델·FastAPI에 대한 의존성이 전혀 없다(json/re/numpy만 사용) — 일부러
app.py에서 분리했다. app.py를 직접 import하면 llama-cpp-python·sentence-transformers·
faiss·kiwipiepy 같은 무거운 의존성이 전부 같이 로드돼서, CI(GitHub 호스팅 러너,
GPU·모델 다운로드 없음이 원칙, tests/test_rules.py 참고)에서 이 로직만 가볍게
단위 테스트할 방법이 없었다. app.py는 이 모듈의 함수를 그대로 import해서 쓴다 —
로직 자체는 옮기기 전과 동일, 동작 변화 없음.
"""

from __future__ import annotations

import json
import re

import numpy as np


def quiz_choice_grounded(choice_text: str, context: str) -> bool:
    """정답 선택지가 실제로 [참고 자료]에서 나온 표현인지 코드로 한 번 더 확인한다.
    프롬프트로 "자료에 없는 걸 지어내지 마라"고 시켜도 4B급 소형 모델이 완전히
    지키진 못한다(2026-09-19 실측 — 지시 전엔 "서울 90%, 부산 70%" 식으로 자료에
    없는 비교를 만들어냄). 정답 문장의 명사·숫자 토큰 중 상당수가 실제로 context에
    그대로 있어야 '근거 있음'으로 인정 — 완벽하진 않지만(토큰이 다른 조합으로 재구성돼도
    통과할 수 있음) 아예 새로 지어낸 내용은 걸러낸다."""
    tokens = [t for t in re.findall(r"[가-힣]{2,}|\d+(?:\.\d+)?%?", choice_text) if len(t) > 1]
    if not tokens:
        return True
    matched = sum(1 for t in tokens if t in context)
    return (matched / len(tokens)) >= 0.7


def parse_student_list(raw: str) -> list[dict]:
    """{"students": [{"name":..., "opinion":...}, ...]} 배열을 파싱하되, 전체 JSON이
    깨져도 개별 항목은 정규식으로 복구를 시도한다(2026-09-19 실측 발견) — 15명 생성을
    시켰더니 한 항목에서 모델이 `"}` 대신 `")`를 써서(중학생 말투 문장 끝에 괄호를
    섞어 쓰다 실수) 배열 전체 파싱이 깨졌다. 학생 한둘이 유실되는 건 허용 오차로
    보고(호출부에서 최소 인원만 확인), 항목 몇 개 때문에 전체를 재시도시키지 않는다."""
    try:
        data = json.loads(raw)
        items = data.get("students", [])
        if isinstance(items, list) and items:
            return items
    except json.JSONDecodeError:
        pass
    pattern = re.compile(r'"name"\s*:\s*"([^"]*)"\s*,\s*"opinion"\s*:\s*"([^"]*)"')
    return [{"name": m.group(1), "opinion": m.group(2)} for m in pattern.finditer(raw)]


def parse_opinion_list(raw: str) -> list[str]:
    """{"students": [{"opinion":...}, ...]} 배열 전용 — parse_student_list와 같은
    이유로 관대하게 복구한다(토론 후 재의견 수집에서도 같은 배열 파싱 위험이 있음)."""
    try:
        data = json.loads(raw)
        items = data.get("students", [])
        if isinstance(items, list) and items:
            return [(it.get("opinion") or "").strip() for it in items]
    except json.JSONDecodeError:
        pass
    pattern = re.compile(r'"opinion"\s*:\s*"([^"]*)"')
    return [m.group(1) for m in pattern.finditer(raw)]


def balanced_cluster_assignment(vectors: np.ndarray, k: int, seed: int = 0) -> list[int]:
    """임베딩 벡터를 k개 그룹으로 최대한 균등하게 나눈다(사용자 요청, 2026-09-19) —
    LLM에게 직접 분류시켰다가 15명 전원이 한 팀으로 쏠리는 걸 실제로 겪은 뒤 도입.
    표준 k-means로 자연스러운 중심을 먼저 찾고(비슷한 의견끼리 뭉치는 성질은 유지),
    그 중심까지 거리 기준으로 "정원을 넘지 않는 선에서 제일 가까운 팀에 배정"하는
    그리디 방식으로 크기를 강제로 맞춘다. seed를 바꾸면 "재배치" 버튼이 다른 초기
    중심에서 시작해 다른 묶음을 만든다."""
    n = len(vectors)
    k = max(1, min(k, n))
    rng = np.random.RandomState(seed)
    idx = rng.choice(n, size=k, replace=False)
    centroids = vectors[idx].copy()
    labels = np.full(n, -1)
    for _ in range(20):
        dists = np.linalg.norm(vectors[:, None, :] - centroids[None, :, :], axis=2)
        new_labels = dists.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(k):
            pts = vectors[labels == c]
            if len(pts) > 0:
                centroids[c] = pts.mean(axis=0)

    dists = np.linalg.norm(vectors[:, None, :] - centroids[None, :, :], axis=2)
    base = n // k
    target = [base + (1 if i < n % k else 0) for i in range(k)]
    order = np.argsort(dists.min(axis=1))  # 자기 그룹이 확실한 학생부터 먼저 배정
    capacity = target.copy()
    assignment = [-1] * n
    for i in order:
        for c in np.argsort(dists[i]):
            if capacity[c] > 0:
                assignment[i] = int(c)
                capacity[c] -= 1
                break
    return assignment


def parse_topic_matches(raw: str) -> list[dict]:
    """{"matches": [...]} 파싱 — 전체가 깨져도 개별 항목은 정규식으로 복구한다
    (AI토론 학생 목록 파싱에서 겪은 것과 같은 배열형 JSON 깨짐 위험에 동일 대응, 92번)."""
    try:
        data = json.loads(raw)
        items = data.get("matches", [])
        if isinstance(items, list):
            return items
    except json.JSONDecodeError:
        pass
    pattern = re.compile(r'"topic_id"\s*:\s*(\d+)[^{}]*?"confidence"\s*:\s*([\d.]+)[^{}]*?"reason"\s*:\s*"([^"]*)"')
    return [{"topic_id": int(m.group(1)), "confidence": float(m.group(2)), "reason": m.group(3)} for m in pattern.finditer(raw)]


def parse_requirements(raw: str) -> list[dict]:
    """{"requirements": [...]} 파싱 — 전체가 깨져도 category/req_text/cite 세 핵심
    필드는 정규식으로 복구한다(92번과 동일한 배열형 JSON 깨짐 대응 패턴)."""
    try:
        data = json.loads(raw)
        items = data.get("requirements", [])
        if isinstance(items, list):
            return items
    except json.JSONDecodeError:
        pass
    pattern = re.compile(r'"category"\s*:\s*"([^"]*)"[^{}]*?"req_text"\s*:\s*"([^"]*)"[^{}]*?"cite"\s*:\s*"([^"]*)"')
    return [{"category": m.group(1), "req_text": m.group(2), "req_value": "", "req_unit": "", "op": "manual", "cite": m.group(3)} for m in pattern.finditer(raw)]


def dedupe_requirements(requirements: list[dict]) -> list[dict]:
    """청크 겹침 구간(chunk_text의 overlap=200자)에 걸친 동일 요구사항이 인접한 두
    청크 양쪽에서 각각 추출돼 중복으로 남는 문제(의사결정_로그 95번에서 실측 확인,
    "실적"·"인증" 항목이 2번씩). 문장 유사도 기반 중복 제거는 "비슷하지만 다른 두
    요구사항"을 잘못 하나로 합칠 위험이 있어 일부러 피하고(95번에서 이미 그렇게
    판단해 보류했었음) — 겹침 구간에서 나온 중복은 원문 자체가 말 그대로 같다는
    점만 이용해, cite(원문을 그대로 인용해야 하는 필드)를 공백 정규화 후 완전히
    같을 때만 같은 요구사항으로 보고 먼저 나온 것만 남긴다."""
    seen: set[str] = set()
    deduped = []
    for r in requirements:
        key = " ".join(r["cite"].split())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    return deduped
