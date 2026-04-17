"""tests/test_postprocess_full.py — 통합 후처리 파이프라인 테스트."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from core.postprocess import (
    clean_hallucinations,
    clean_segments,
    clean_text,
    full_postprocess,
)
from core.glossary import add_term, load_glossary
from core.korean_normalizer import is_available as kss_available


# ──────────────────────────────────────────────
# 하위 호환
# ──────────────────────────────────────────────

def test_clean_hallucinations_is_alias_of_clean_text():
    assert clean_hallucinations is clean_text


def test_clean_text_existing_behavior():
    assert clean_text("이,이,이,이,이") == "이"
    assert clean_text("네 네 네 네 네") == "네"
    assert clean_text("") == ""


def test_clean_segments_preserves_structure():
    segs = [
        {"start": 0.0, "end": 1.0, "text": "네 네 네 네 네"},
        {"start": 1.0, "end": 2.0, "text": "정상 문장."},
    ]
    out = clean_segments(segs)
    assert len(out) == 2
    assert out[0]["start"] == 0.0
    assert out[0]["end"] == 1.0
    assert "네 네 네 네" not in out[0]["text"]
    assert out[1]["text"] == "정상 문장."


# ──────────────────────────────────────────────
# full_postprocess
# ──────────────────────────────────────────────

def test_full_postprocess_empty():
    assert full_postprocess("") == ""


def test_full_postprocess_all_off_keeps_hallucination_cleanup():
    result = full_postprocess(
        "네 네 네 네 네 안녕하세요",
        use_glossary=False, use_korean_norm=False,
    )
    assert "네 네 네 네" not in result
    assert "안녕하세요" in result


def test_full_postprocess_glossary_only(tmp_path):
    gp = tmp_path / "glossary.json"
    add_term("Premo", ["프레모"], path=gp)
    gloss = load_glossary(gp)

    result = full_postprocess(
        "프레모가 좋다.",
        use_glossary=True, use_korean_norm=False,
        glossary=gloss,
    )
    assert "Premo가" in result


def test_full_postprocess_korean_norm_only():
    result = full_postprocess(
        "ㅋㅋㅋㅋㅋㅋㅋ 안녕",
        use_glossary=False, use_korean_norm=True,
        paragraphs=False,
    )
    assert isinstance(result, str)


@pytest.mark.skipif(not kss_available(), reason="KSS not installed")
def test_full_postprocess_paragraphs_when_kss_available():
    text = (
        "안녕하세요. 반갑습니다. "
        "오늘은 회의가 있습니다. 내일 다시 만나요. "
        "감사합니다. 좋은 하루 보내세요."
    )
    result = full_postprocess(
        text,
        use_glossary=False, use_korean_norm=True,
        paragraphs=True, sentences_per_paragraph=3,
    )
    assert "\n\n" in result


def test_full_postprocess_integrates_all_steps(tmp_path):
    gp = tmp_path / "glossary.json"
    add_term("Premo", ["프레모"], path=gp)
    gloss = load_glossary(gp)

    text = (
        "네 네 네 네 네 프레모가 좋아요. "
        "프레모는 훌륭합니다. 프레모를 사용합니다. "
        "회의를 시작하겠습니다. 안건이 있습니다."
    )
    result = full_postprocess(
        text,
        use_glossary=True, use_korean_norm=True,
        glossary=gloss, paragraphs=True,
    )
    assert "네 네 네 네" not in result
    assert "Premo가" in result
    assert "Premo는" in result
    assert "Premo를" in result


def test_full_postprocess_empty_glossary_no_crash():
    result = full_postprocess(
        "안녕하세요",
        use_glossary=True, use_korean_norm=False,
        glossary={"terms": []},
    )
    assert "안녕하세요" in result
