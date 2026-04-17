"""tests/test_auto_glossary.py — 자동 용어 변형 감지 테스트."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from core.auto_glossary import (
    _HAS_RAPIDFUZZ,
    STOPWORDS,
    _extract_words,
    _is_similar,
    _UnionFind,
    detect_term_variants,
    format_cluster_preview,
)


# ──────────────────────────────────────────────
# Union-Find
# ──────────────────────────────────────────────

def test_unionfind_basic():
    uf = _UnionFind(["a", "b", "c", "d"])
    uf.union("a", "b")
    uf.union("c", "d")
    assert uf.find("a") == uf.find("b")
    assert uf.find("c") == uf.find("d")
    assert uf.find("a") != uf.find("c")


def test_unionfind_transitive():
    uf = _UnionFind(["a", "b", "c"])
    uf.union("a", "b")
    uf.union("b", "c")
    assert uf.find("a") == uf.find("c")


# ──────────────────────────────────────────────
# _is_similar
# ──────────────────────────────────────────────

@pytest.mark.skipif(not _HAS_RAPIDFUZZ, reason="rapidfuzz not installed")
def test_is_similar_short_korean_one_char_diff():
    """한글 3자 이내 단어 lev=1 → 유사."""
    assert _is_similar("프레모", "프리모", 80)
    assert _is_similar("프레모", "프레머", 80)
    assert _is_similar("노마", "노머", 80)
    assert _is_similar("클로드", "크로드", 80)


@pytest.mark.skipif(not _HAS_RAPIDFUZZ, reason="rapidfuzz not installed")
def test_is_similar_dissimilar():
    assert not _is_similar("프레모", "회의", 80)
    assert not _is_similar("안녕", "반갑다", 80)


@pytest.mark.skipif(not _HAS_RAPIDFUZZ, reason="rapidfuzz not installed")
def test_is_similar_long_vs_short_not_boosted():
    """짧은/긴 단어 혼합은 보정 대상 아님."""
    assert not _is_similar("안녕", "안녕하세요", 80)


# ──────────────────────────────────────────────
# 단어 추출
# ──────────────────────────────────────────────

def test_extract_words_korean():
    words = _extract_words("오늘 회의는 Premo 프로젝트 입니다.")
    assert "오늘" in words
    assert "Premo" in words
    assert "프로젝트" in words


def test_extract_words_ignores_single_char():
    words = _extract_words("나 는 너 다 ok ab")
    assert "나" not in words
    assert "ok" in words
    assert "ab" in words


# ──────────────────────────────────────────────
# detect_term_variants — 핵심 시나리오
# ──────────────────────────────────────────────

@pytest.mark.skipif(not _HAS_RAPIDFUZZ, reason="rapidfuzz not installed")
def test_detect_korean_variants_premo_cluster():
    """스펙 예시: 프레모/프리모/프레머 → 하나의 클러스터."""
    text = (
        "오늘 프레모 회의. 프레모 기능을 설명했다. 프레모 팀. "
        "다른 사람은 프리모 라고 불렀고, 또 한 명은 프레머 라고 했다. "
        "회의 회의 회의."
    )
    clusters = detect_term_variants(text, min_frequency=2, similarity_threshold=80)

    premo_cluster = next(
        (c for c in clusters if "프레모" in [c["canonical"]] + c["aliases"]),
        None,
    )
    assert premo_cluster is not None

    members = [premo_cluster["canonical"]] + premo_cluster["aliases"]
    assert "프레모" in members
    assert "프리모" in members
    assert "프레머" in members

    # canonical = 최빈도 (프레모 3회)
    assert premo_cluster["canonical"] == "프레모"
    # total_count = 3 + 1 + 1 = 5
    assert premo_cluster["total_count"] == 5


@pytest.mark.skipif(not _HAS_RAPIDFUZZ, reason="rapidfuzz not installed")
def test_detect_cluster_minimum_size_two():
    """클러스터 크기 1 (단일 단어) 은 반환하지 않는다."""
    text = "단일단어 별개단어 독립단어 프레모 프레모"
    clusters = detect_term_variants(text, min_frequency=1, similarity_threshold=80)
    for c in clusters:
        assert len(c["aliases"]) >= 1


@pytest.mark.skipif(not _HAS_RAPIDFUZZ, reason="rapidfuzz not installed")
def test_detect_excludes_existing_canonical():
    text = "프레모 프레모 프리모 프레머"
    clusters = detect_term_variants(
        text, min_frequency=2, similarity_threshold=80,
        existing_canonicals={"프레모"},
    )
    for c in clusters:
        assert "프레모" not in [c["canonical"]] + c["aliases"]


@pytest.mark.skipif(not _HAS_RAPIDFUZZ, reason="rapidfuzz not installed")
def test_detect_excludes_existing_alias():
    text = "프레모 프레모 프리모 프레머"
    clusters = detect_term_variants(
        text, min_frequency=2, similarity_threshold=80,
        existing_aliases={"프리모"},
    )
    for c in clusters:
        members = [c["canonical"]] + c["aliases"]
        assert "프리모" not in members


@pytest.mark.skipif(not _HAS_RAPIDFUZZ, reason="rapidfuzz not installed")
def test_detect_total_count_descending():
    """여러 클러스터 → total_count 내림차순."""
    text = ("프레모 " * 5) + ("프리모 " * 3) + ("노마 " * 2) + ("노머 " * 1)
    clusters = detect_term_variants(text, min_frequency=2, similarity_threshold=80)
    for i in range(len(clusters) - 1):
        assert clusters[i]["total_count"] >= clusters[i + 1]["total_count"]


def test_detect_empty_text():
    assert detect_term_variants("") == []


@pytest.mark.skipif(not _HAS_RAPIDFUZZ, reason="rapidfuzz not installed")
def test_detect_min_frequency_filter():
    """total_count < min_frequency 클러스터 제외."""
    text = "프레모 프리모"  # total=2
    clusters = detect_term_variants(text, min_frequency=3, similarity_threshold=80)
    # total=2 < 3 → 제외
    assert all(c["total_count"] >= 3 for c in clusters)


# ──────────────────────────────────────────────
# 불용어
# ──────────────────────────────────────────────

def test_stopwords_has_common_korean():
    assert "있다" in STOPWORDS
    assert "그리고" in STOPWORDS
    assert "합니다" in STOPWORDS


# ──────────────────────────────────────────────
# 포맷 헬퍼
# ──────────────────────────────────────────────

def test_format_cluster_preview():
    c = {
        "canonical": "프레모",
        "aliases": ["프리모", "프레머"],
        "total_count": 5,
        "frequencies": {"프레모": 3, "프리모": 1, "프레머": 1},
    }
    preview = format_cluster_preview(c)
    assert "프레모" in preview
    assert "프리모" in preview
    assert "프레머" in preview
    assert "3" in preview
