"""전사 결과 후처리 필터.

Whisper 계열 모델의 대표적인 환각 패턴을 정규식으로 정리한다:
  1. 동일 단어 5회 이상 연속 반복 → 1회로 축소
  2. 쉼표/공백으로 연결된 반복 패턴 ("이,이,이,이" → "이")
  3. 추임새(응/음/어/아/네) 4회 이상 연속 → 1회로 축소

engine.transcribe() 에서 자동으로 호출된다.
"""

from __future__ import annotations

import re
from typing import List


# ──────────────────────────────────────────────
# 정규식 패턴
# ──────────────────────────────────────────────

# 1) 동일 단어(공백 구분) 5회 이상 연속 반복
#    예: "안녕 안녕 안녕 안녕 안녕 안녕" → "안녕"
#    \S+  : 공백이 아닌 한 덩어리 (단어)
#    (?: \1){4,} : 공백+동일단어가 4번 더 (= 총 5번 이상)
_RE_REPEAT_WORD = re.compile(r"(\S+?)(?:[\s]+\1){4,}")

# 2) 쉼표/마침표/물음표/공백으로 연결된 동일 토큰 반복
#    예: "이,이,이,이" → "이"
#         "아. 아. 아. 아." → "아."
#    토큰은 구두점을 포함하지 않는 짧은 단위(최대 8자) — 과도한 매칭 방지
_RE_REPEAT_PUNCT = re.compile(r"([가-힣A-Za-z0-9]{1,8})([,.\?!·、][\s]*\1){3,}")

# 3) 추임새(응/음/어/아/네/예/아이고/어머) 4회 이상 연속 반복
#    쉼표/공백/마침표로 구분된 것도 포함
#    예: "네 네 네 네" → "네"
#         "음,음,음,음,음" → "음"
_FILLERS = ["응", "음", "어", "아", "네", "예"]
_FILLER_GROUP = "|".join(_FILLERS)
_RE_REPEAT_FILLER = re.compile(
    rf"(?P<f>{_FILLER_GROUP})(?:[\s,.\?!·、]+(?P=f)){{3,}}"
)


# ──────────────────────────────────────────────
# 필터 함수
# ──────────────────────────────────────────────

def collapse_repeated_words(text: str) -> str:
    """같은 단어가 공백으로 5회 이상 연속 반복되면 1회로 축소한다."""
    return _RE_REPEAT_WORD.sub(r"\1", text)


def collapse_punct_repetition(text: str) -> str:
    """쉼표/마침표 등으로 연결된 동일 토큰 반복을 1회로 축소한다."""
    return _RE_REPEAT_PUNCT.sub(r"\1", text)


def collapse_fillers(text: str) -> str:
    """추임새가 4회 이상 연속되면 1회로 축소한다."""
    return _RE_REPEAT_FILLER.sub(lambda m: m.group("f"), text)


def clean_text(text: str) -> str:
    """모든 후처리 필터를 순차 적용한다.

    순서가 중요하다:
      1) 구두점 반복 → (공백으로 반복된 패턴이 드러남)
      2) 단어 반복 축소
      3) 추임새 축소
      4) 연속된 공백 정리
    """
    if not text:
        return text

    cleaned = collapse_punct_repetition(text)
    cleaned = collapse_repeated_words(cleaned)
    cleaned = collapse_fillers(cleaned)

    # 줄 안의 중복 공백 정리 (줄바꿈은 유지)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    # 줄 끝 공백 정리
    cleaned = re.sub(r"[ \t]+(\n|$)", r"\1", cleaned)

    return cleaned.strip()


def clean_segments(segments: List[dict]) -> List[dict]:
    """세그먼트 리스트의 각 텍스트에 필터를 적용한다.

    segment 스키마(whisper 계열 공통): {"start": float, "end": float, "text": str, ...}
    """
    cleaned: List[dict] = []
    for seg in segments:
        new_seg = dict(seg)
        if "text" in new_seg and isinstance(new_seg["text"], str):
            new_seg["text"] = clean_text(new_seg["text"])
        cleaned.append(new_seg)
    return cleaned
