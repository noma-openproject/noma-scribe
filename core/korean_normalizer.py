"""한국어 정규화 모듈.

KSS(Korean Sentence Splitter) 를 활용해:
  - 띄어쓰기 교정
  - 문장 분리
  - 3~5 문장 단위 자연스러운 문단 구성

KSS 설치 실패(mecab 등) 시 graceful fallback — 원본 텍스트를 그대로 반환하거나
정규식 기반의 간단한 처리로 대체.
"""

from __future__ import annotations

import re
from typing import Callable, List, Optional

# ──────────────────────────────────────────────
# KSS 동적 import (graceful)
# ──────────────────────────────────────────────

_KSS_AVAILABLE = False
_KSS_SPLIT = None
_KSS_SPACING = None

try:
    import kss  # type: ignore
    if hasattr(kss, "split_sentences"):
        _KSS_SPLIT = kss.split_sentences
        _KSS_AVAILABLE = True
    if hasattr(kss, "correct_spacing"):
        _KSS_SPACING = kss.correct_spacing
except ImportError:
    _KSS_AVAILABLE = False
except Exception:
    # pecab/mecab 초기화 실패 등 — 무시하고 fallback
    _KSS_AVAILABLE = False


def is_available() -> bool:
    """KSS 실제 사용 가능 여부."""
    return _KSS_AVAILABLE


# ──────────────────────────────────────────────
# 정규화 함수들
# ──────────────────────────────────────────────

_RE_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_RE_MULTI_NEWLINE = re.compile(r"\n{3,}")
_RE_REPEATED_CHAR = re.compile(r"(.)\1{3,}")  # 같은 문자 4회 이상 반복
_RE_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?。！？])\s+")


def _regex_split_sentences(text: str) -> List[str]:
    parts = _RE_SENTENCE_BOUNDARY.split(text)
    return [p.strip() for p in parts if p and p.strip()]


def _chunk_sentences(sentences: List[str], max_chars: int = 240, max_sentences: int = 6) -> List[str]:
    """띄어쓰기 교정을 위해 문장을 작은 청크로 묶는다.

    pecab backend에서는 긴 전체 문자열보다 짧은 청크를 여러 번 처리하는 편이
    훨씬 빠르다.
    """
    if not sentences:
        return []

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for sentence in sentences:
        add_len = len(sentence) + (1 if current else 0)
        if current and (current_len + add_len > max_chars or len(current) >= max_sentences):
            chunks.append(" ".join(current))
            current = [sentence]
            current_len = len(sentence)
        else:
            current.append(sentence)
            current_len += add_len
    if current:
        chunks.append(" ".join(current))
    return chunks


def normalize_korean(text: str) -> str:
    """반복 문자 축소 + 공백 정리.

    - "ㅋㅋㅋㅋㅋㅋ" → "ㅋㅋㅋ"
    - "???????" → "???"
    - 중복 공백 → 1개
    - 3개 이상 연속 개행 → 2개
    """
    if not text:
        return text

    out = _RE_REPEATED_CHAR.sub(lambda m: m.group(1) * 3, text)
    out = _RE_MULTI_SPACE.sub(" ", out)
    out = _RE_MULTI_NEWLINE.sub("\n\n", out)
    out = re.sub(r"[ \t]+(\n|$)", r"\1", out)
    return out.strip()


def fix_spacing(
    text: str,
    progress_callback: Optional[Callable[[float], None]] = None,
    debug: bool = False,
) -> str:
    """KSS 기반 띄어쓰기 교정.

    KSS 가 없거나 실패하면 원본 반환.
    """
    if not text or not _KSS_SPACING:
        return text

    try:
        # 긴 회의록에서 KSS sentence split까지 전체 텍스트에 적용하면 pecab
        # backend에서 지나치게 느려질 수 있어, 띄어쓰기용 청크 분할은 가벼운
        # 정규식 문장 경계를 우선 사용한다.
        sentences = _regex_split_sentences(text)
        if not sentences:
            sentences = split_sentences(text)
        if len(text) < 200 or len(sentences) <= 1:
            out = _KSS_SPACING(text)
            if progress_callback:
                progress_callback(100.0)
            return out

        chunks = _chunk_sentences(sentences)
        corrected: List[str] = []
        total = len(chunks)
        for idx, chunk in enumerate(chunks, 1):
            corrected.append(_KSS_SPACING(chunk))
            if progress_callback:
                progress_callback(idx / total * 100.0)
        return " ".join(part.strip() for part in corrected if part and part.strip())
    except Exception:
        if debug:
            raise
        return text


def split_sentences(text: str) -> List[str]:
    """KSS 로 문장 분리. KSS 가 없으면 간단한 정규식으로 fallback."""
    if not text:
        return []

    if len(text) > 1200:
        return _regex_split_sentences(text)

    if _KSS_AVAILABLE and _KSS_SPLIT:
        try:
            result = _KSS_SPLIT(text)
            if isinstance(result, list):
                return [s.strip() for s in result if s and s.strip()]
        except Exception:
            pass

    # Fallback: 간단한 문장 경계 기반
    return _regex_split_sentences(text)


def split_into_paragraphs(
    text: str,
    sentences_per_paragraph: int = 4,
) -> str:
    """문장 분리 후 N 문장 단위로 문단을 구성한다.

    Args:
        text: 원본 텍스트
        sentences_per_paragraph: 문단당 문장 수 (3~5 권장)

    Returns:
        문단이 \\n\\n 으로 구분된 텍스트
    """
    if not text:
        return text

    sentences = split_sentences(text)
    if not sentences:
        return text

    paragraphs: List[str] = []
    for i in range(0, len(sentences), sentences_per_paragraph):
        chunk = sentences[i:i + sentences_per_paragraph]
        paragraphs.append(" ".join(chunk))

    return "\n\n".join(paragraphs)
