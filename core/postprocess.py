"""전사 결과 후처리 필터.

Whisper 계열 모델의 대표적인 환각 패턴을 정규식으로 정리한다:
  1. 동일 단어 5회 이상 연속 반복 → 1회로 축소
  2. 쉼표/공백으로 연결된 반복 패턴 ("이,이,이,이" → "이")
  3. 추임새(응/음/어/아/네) 4회 이상 연속 → 1회로 축소

engine.transcribe() 에서 자동으로 호출된다.
"""

from __future__ import annotations

import re
from typing import Callable, List, Optional


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


# ──────────────────────────────────────────────
# v0.6: 통합 후처리 파이프라인
# ──────────────────────────────────────────────

# clean_hallucinations 는 clean_text 의 명시적 이름 (기존 호환성 유지 + 의미 명확화)
clean_hallucinations = clean_text


def full_postprocess(
    text: str,
    use_glossary: bool = True,
    use_korean_norm: bool = True,
    glossary: "dict | None" = None,
    fuzzy_threshold: int = 85,
    paragraphs: bool = True,
    sentences_per_paragraph: int = 4,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    warning_callback: Optional[Callable[[str], None]] = None,
    debug: bool = False,
) -> str:
    """v0.6 통합 후처리 파이프라인.

    순서:
      1) clean_hallucinations (반복 환각 필터, 기존 로직)
      2) apply_glossary (사용자 용어집 치환, use_glossary=True 일 때)
      3) normalize_korean + fix_spacing (KSS 한국어 정규화, use_korean_norm=True 일 때)
      4) split_into_paragraphs (3~5 문장 단위 문단, paragraphs=True 일 때)

    Args:
        text: 원본 전사 텍스트
        use_glossary: 용어집 치환 적용 여부
        use_korean_norm: KSS 한국어 정규화 적용 여부
        glossary: 용어집 dict (None 이면 DEFAULT_GLOSSARY_PATH 에서 로드)
        fuzzy_threshold: 용어집 유사도 임계치 (0-100)
        paragraphs: 문단 분리 여부 (use_korean_norm=True 일 때만 동작)
        sentences_per_paragraph: 문단당 문장 수

    Returns:
        후처리 완료된 텍스트
    """
    if not text:
        return text

    def _progress(pct: float, message: str) -> None:
        if progress_callback:
            progress_callback(pct, message)

    # 1) 반복 환각 제거 (기존)
    _progress(10, "반복 패턴 정리 중...")
    out = clean_hallucinations(text)

    # 2) 용어집 치환 (optional)
    if use_glossary:
        _progress(30, "용어집 적용 중...")
        try:
            from core.glossary import apply_glossary
            out = apply_glossary(out, glossary=glossary, fuzzy_threshold=fuzzy_threshold)
        except Exception as exc:
            # 용어집 로딩/치환 실패는 치명적이지 않음 — 직전 단계 결과 유지
            if debug:
                raise RuntimeError(f"용어집 적용 실패: {exc}") from exc
            if warning_callback:
                warning_callback(f"용어집 적용 실패: {exc}")

    # 3) 한국어 정규화 + 띄어쓰기 교정 + 문단 (optional)
    if use_korean_norm:
        try:
            from core.korean_normalizer import (
                normalize_korean, fix_spacing, split_into_paragraphs,
            )
            _progress(45, "한국어 정리 중...")
            out = normalize_korean(out)
            _progress(55, "띄어쓰기 교정 중...")
            out = fix_spacing(
                out,
                progress_callback=lambda pct: _progress(55 + pct * 0.30, "띄어쓰기 교정 중..."),
                debug=debug,
            )
            if paragraphs:
                _progress(90, "문단 정리 중...")
                out = split_into_paragraphs(out, sentences_per_paragraph)
        except Exception as exc:
            # KSS 실패는 graceful — 직전 단계 결과 유지
            if debug:
                raise RuntimeError(f"한국어 정리 실패: {exc}") from exc
            if warning_callback:
                warning_callback(f"한국어 정리 실패: {exc}")

    _progress(100, "텍스트 정리 완료")
    return out


def postprocess_segments(
    segments: List[dict],
    use_glossary: bool = True,
    use_korean_norm: bool = True,
    glossary: "dict | None" = None,
    fuzzy_threshold: int = 85,
    progress_callback: Optional[Callable[[float], None]] = None,
    warning_callback: Optional[Callable[[str], None]] = None,
    debug: bool = False,
) -> List[dict]:
    """세그먼트별 후처리를 적용한다.

    타임스탬프 정합성 유지를 위해 문단 분리는 하지 않고, 각 세그먼트 텍스트에만
    용어집/한국어 정규화를 적용한다.
    """
    processed: List[dict] = []
    total = len(segments) or 1
    for idx, seg in enumerate(segments, 1):
        new_seg = dict(seg)
        text = new_seg.get("text")
        if isinstance(text, str):
            new_seg["text"] = full_postprocess(
                text,
                use_glossary=use_glossary,
                use_korean_norm=use_korean_norm,
                glossary=glossary,
                fuzzy_threshold=fuzzy_threshold,
                paragraphs=False,
                warning_callback=warning_callback,
                debug=debug,
            )
        processed.append(new_seg)
        if progress_callback:
            progress_callback(idx / total * 100.0)
    return processed
