"""추출된 텍스트의 사후 품질 점검 — "738/738건 성공"이 "비어있지 않음"만 확인했을 뿐
내용이 맞는지는 확인 안 했다는 문제(의사결정_로그 13번)를 메꾼다.

사람이 원문과 직접 대조하는 건 자동화할 수 없지만, 사람 없이도 잡아낼 수 있는 이상
신호(반복 줄, 한글 비율, 대체/깨진 문자 비율)는 여기서 계산한다. "품질이 좋다"를 증명하는
게 아니라 "이 문서는 사람이 다시 봐야 할 것 같다"는 후보를 걸러내는 용도다.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

# 한글 문서인데 한글 비율이 너무 낮으면 인코딩이 깨졌거나 OCR이 엉뚱하게 인식한 것
_MIN_KOREAN_RATIO = 0.05
# 유니코드 대체문자(U+FFFD)·사용자 정의 영역 문자가 많으면 디코딩 실패의 직접 증거
_MAX_SUSPICIOUS_RATIO = 0.02
# 같은 줄이 계속 반복되면 스캔 실패로 같은 헤더/푸터만 반복되거나 OCR이 망가진 신호일 수
# 있지만, 실제 738건 코퍼스로 임계값을 여러 개 테스트해보니(의사결정_로그 72번) 길이와
# 무관하게 비율만 보면 오탐이 압도적으로 많았다 — 긴 공식 규정 문서는 원래 조항·표가
# 반복되는 구조라 87%까지도 정상인 경우가 실제로 있었다. 반대로 "짧은 문서인데 반복 비율도
# 높다"는 건 실질적으로 남는 고유 내용이 거의 없다는 뜻이라 훨씬 신뢰할 수 있는 신호였다 —
# 그래서 길이 조건을 같이 건다(길이만 조건에 넣으니 117건→7건, 이미 알던 문제 2건은 그대로
# 잡힘, 별도로 신뢰도 높다고 확인된 5건이 새로 드러남).
_MAX_REPEATED_LINE_RATIO = 0.4
_SHORT_DOC_CHAR_COUNT = 3000
_MIN_CHAR_COUNT = 50


@dataclass
class QualitySignals:
    char_count: int
    korean_ratio: float
    repeated_line_ratio: float
    suspicious_char_ratio: float
    flags: list[str] = field(default_factory=list)

    @property
    def needs_review(self) -> bool:
        return len(self.flags) > 0


def assess(text: str) -> QualitySignals:
    char_count = len(text)
    if char_count == 0:
        return QualitySignals(0, 0.0, 0.0, 0.0, ["빈 텍스트"])

    korean_chars = sum(1 for c in text if "가" <= c <= "힣")
    korean_ratio = korean_chars / char_count

    suspicious_chars = sum(1 for c in text if c == "�" or "" <= c <= "")
    suspicious_ratio = suspicious_chars / char_count

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if lines:
        counts = Counter(lines)
        repeated = sum(c for c in counts.values() if c > 1)
        repeated_ratio = repeated / len(lines)
    else:
        repeated_ratio = 0.0

    flags: list[str] = []
    if char_count < _MIN_CHAR_COUNT:
        flags.append("글자 수 지나치게 적음")
    if korean_ratio < _MIN_KOREAN_RATIO:
        flags.append(f"한글 비율 낮음({korean_ratio:.1%})")
    if suspicious_ratio > _MAX_SUSPICIOUS_RATIO:
        flags.append(f"대체/사용자정의 문자 과다({suspicious_ratio:.1%})")
    if repeated_ratio > _MAX_REPEATED_LINE_RATIO and char_count < _SHORT_DOC_CHAR_COUNT:
        flags.append(f"짧은 문서인데 반복 줄 과다({repeated_ratio:.1%}, {char_count}자)")

    return QualitySignals(char_count, korean_ratio, repeated_ratio, suspicious_ratio, flags)
