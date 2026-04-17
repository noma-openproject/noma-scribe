"""tests/test_korean_normalizer.py — KSS 연동 테스트 (미설치 시 skip)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from core.korean_normalizer import (
    fix_spacing,
    is_available,
    normalize_korean,
    split_into_paragraphs,
    split_sentences,
)


# ──────────────────────────────────────────────
# normalize_korean (KSS 없이도 동작)
# ──────────────────────────────────────────────

def test_normalize_repeated_chars():
    assert normalize_korean("ㅋㅋㅋㅋㅋㅋ") == "ㅋㅋㅋ"
    assert normalize_korean("????????") == "???"
    assert normalize_korean("!!!!!!") == "!!!"


def test_normalize_multiple_spaces():
    assert normalize_korean("hello   world") == "hello world"
    assert normalize_korean("a    b     c") == "a b c"


def test_normalize_multiple_newlines():
    assert normalize_korean("line1\n\n\n\nline2") == "line1\n\nline2"


def test_normalize_empty():
    assert normalize_korean("") == ""


def test_normalize_preserves_normal_text():
    txt = "안녕하세요. 반갑습니다."
    assert normalize_korean(txt) == txt


# ──────────────────────────────────────────────
# fix_spacing
# ──────────────────────────────────────────────

@pytest.mark.skipif(not is_available(), reason="KSS not installed (mecab/pecab missing)")
def test_fix_spacing_produces_string():
    result = fix_spacing("안녕하세요저는테스트입니다")
    assert isinstance(result, str)
    assert len(result) >= len("안녕하세요저는테스트입니다")


def test_fix_spacing_graceful_when_kss_missing():
    """KSS 없거나 실패해도 예외 없이 문자열 반환."""
    result = fix_spacing("아무거나")
    assert isinstance(result, str)


def test_fix_spacing_empty():
    assert fix_spacing("") == ""


# ──────────────────────────────────────────────
# split_sentences
# ──────────────────────────────────────────────

def test_split_sentences_basic():
    sents = split_sentences("안녕하세요. 반갑습니다. 오늘 날씨가 좋네요.")
    assert len(sents) >= 2


def test_split_sentences_empty():
    assert split_sentences("") == []


def test_split_sentences_single():
    sents = split_sentences("단일 문장입니다.")
    assert len(sents) >= 1


# ──────────────────────────────────────────────
# split_into_paragraphs
# ──────────────────────────────────────────────

def test_split_into_paragraphs_breaks_long_text():
    txt = (
        "안녕하세요. 반갑습니다. "
        "오늘은 날씨가 좋네요. 어제는 비가 왔습니다. "
        "내일도 맑을 예정입니다. 주말에는 여행을 가려고 합니다."
    )
    result = split_into_paragraphs(txt, sentences_per_paragraph=3)
    assert "\n\n" in result


def test_split_into_paragraphs_short_text_no_break():
    result = split_into_paragraphs("한 문장.", sentences_per_paragraph=5)
    assert "\n\n" not in result


def test_split_into_paragraphs_empty():
    assert split_into_paragraphs("") == ""
