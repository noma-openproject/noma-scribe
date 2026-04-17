"""tests/test_glossary.py — 용어집 로딩/저장/치환 테스트."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from core.glossary import (
    _HAS_RAPIDFUZZ,
    _split_particle,
    add_term,
    apply_glossary,
    load_glossary,
    remove_term,
    save_glossary,
)


@pytest.fixture
def tmp_glossary_path(tmp_path: Path) -> Path:
    return tmp_path / "glossary.json"


# ──────────────────────────────────────────────
# I/O
# ──────────────────────────────────────────────

def test_load_missing_returns_empty(tmp_glossary_path):
    assert load_glossary(tmp_glossary_path) == {"terms": []}


def test_load_corrupted_returns_empty(tmp_glossary_path):
    tmp_glossary_path.write_text("not valid json {]")
    assert load_glossary(tmp_glossary_path) == {"terms": []}


def test_load_wrong_schema_returns_empty(tmp_glossary_path):
    tmp_glossary_path.write_text(json.dumps({"wrong": "schema"}))
    assert load_glossary(tmp_glossary_path) == {"terms": []}


def test_save_and_reload_roundtrip(tmp_glossary_path):
    data = {
        "terms": [
            {"canonical": "Premo", "aliases": ["프레모", "프리모"]},
            {"canonical": "Noma", "aliases": ["노마"]},
        ]
    }
    save_glossary(data, tmp_glossary_path)
    assert load_glossary(tmp_glossary_path) == data


def test_save_creates_parent_dir(tmp_path):
    p = tmp_path / "nested" / "dir" / "glossary.json"
    save_glossary({"terms": [{"canonical": "X", "aliases": []}]}, p)
    assert p.exists()


# ──────────────────────────────────────────────
# add_term / remove_term
# ──────────────────────────────────────────────

def test_add_term_basic(tmp_glossary_path):
    data = add_term("Premo", ["프레모"], path=tmp_glossary_path)
    assert len(data["terms"]) == 1
    assert data["terms"][0]["canonical"] == "Premo"
    assert data["terms"][0]["aliases"] == ["프레모"]


def test_add_term_overwrites_existing(tmp_glossary_path):
    add_term("Premo", ["old"], path=tmp_glossary_path)
    data = add_term("Premo", ["new1", "new2"], path=tmp_glossary_path)
    assert len(data["terms"]) == 1
    assert data["terms"][0]["aliases"] == ["new1", "new2"]


def test_add_term_none_aliases(tmp_glossary_path):
    data = add_term("Solo", None, path=tmp_glossary_path)
    assert data["terms"][0]["aliases"] == []


def test_remove_term(tmp_glossary_path):
    add_term("Premo", ["프레모"], path=tmp_glossary_path)
    add_term("Noma", ["노마"], path=tmp_glossary_path)
    data = remove_term("Premo", path=tmp_glossary_path)
    assert len(data["terms"]) == 1
    assert data["terms"][0]["canonical"] == "Noma"


def test_remove_nonexistent_no_error(tmp_glossary_path):
    add_term("Premo", [], path=tmp_glossary_path)
    data = remove_term("Nope", path=tmp_glossary_path)
    assert len(data["terms"]) == 1


# ──────────────────────────────────────────────
# 조사 분리
# ──────────────────────────────────────────────

def test_split_particle_basic():
    assert _split_particle("Premo가") == ("Premo", "가")
    assert _split_particle("노마를") == ("노마", "를")
    assert _split_particle("회의에서") == ("회의", "에서")
    assert _split_particle("파일부터") == ("파일", "부터")


def test_split_particle_no_particle():
    assert _split_particle("Premo") == ("Premo", "")
    assert _split_particle("hello") == ("hello", "")


# ──────────────────────────────────────────────
# apply_glossary — 정확 매칭 + 조사 보존
# ──────────────────────────────────────────────

def test_apply_exact_alias_match(tmp_glossary_path):
    add_term("Premo", ["프레모", "프리모"], path=tmp_glossary_path)
    glossary = load_glossary(tmp_glossary_path)
    result = apply_glossary("오늘 프레모 회의가 있습니다.", glossary)
    assert "Premo" in result
    assert "프레모" not in result


def test_apply_preserves_korean_particle(tmp_glossary_path):
    add_term("Premo", ["프레모"], path=tmp_glossary_path)
    glossary = load_glossary(tmp_glossary_path)

    assert "Premo가" in apply_glossary("프레모가 최고다.", glossary)
    assert "Premo를" in apply_glossary("프레모를 사용합니다.", glossary)
    assert "Premo의" in apply_glossary("프레모의 기능은", glossary)


def test_apply_canonical_self_match(tmp_glossary_path):
    add_term("Premo", ["프레모"], path=tmp_glossary_path)
    glossary = load_glossary(tmp_glossary_path)
    result = apply_glossary("Premo는 최고다.", glossary)
    assert "Premo는" in result


def test_apply_empty_glossary(tmp_glossary_path):
    glossary = load_glossary(tmp_glossary_path)
    text = "아무것도 바뀌지 않아야 한다."
    assert apply_glossary(text, glossary) == text


def test_apply_empty_text():
    assert apply_glossary("", {"terms": []}) == ""


def test_apply_multiple_aliases_in_text(tmp_glossary_path):
    add_term("Premo", ["프레모", "프리모"], path=tmp_glossary_path)
    add_term("Noma", ["노마"], path=tmp_glossary_path)
    glossary = load_glossary(tmp_glossary_path)
    result = apply_glossary("프레모와 노마를 비교", glossary)
    assert "Premo와" in result
    assert "Noma를" in result


# ──────────────────────────────────────────────
# apply_glossary — RapidFuzz
# ──────────────────────────────────────────────

@pytest.mark.skipif(not _HAS_RAPIDFUZZ, reason="rapidfuzz not installed")
def test_apply_fuzzy_threshold_strict(tmp_glossary_path):
    """threshold 100 이면 canonical 정확 매칭만 (alias 없는 다른 단어는 치환 X)."""
    add_term("Premo", [], path=tmp_glossary_path)
    glossary = load_glossary(tmp_glossary_path)
    result = apply_glossary("Prem는 다른 단어", glossary, fuzzy_threshold=100)
    assert "Prem는" in result


def test_apply_short_words_not_fuzzy_matched(tmp_glossary_path):
    """한 글자 단어는 fuzzy 매칭에서 제외 (오매칭 방지)."""
    add_term("A", [], path=tmp_glossary_path)
    glossary = load_glossary(tmp_glossary_path)
    result = apply_glossary("B 혼자", glossary)
    assert "B" in result
