"""자동 용어 변형 감지.

전사 결과에서 서로 유사한 단어 집합(예: Premo / 프레모 / 프리모)을
자동 클러스터링하여 용어집 후보로 제안한다.

알고리즘:
    1) 2글자 이상 단어 추출 + 빈도 집계 (불용어 제외)
    2) 모든 단어 쌍에 대해 RapidFuzz 유사도 계산
    3) 유사도 >= similarity_threshold 쌍을 Union-Find 로 클러스터링
    4) 크기 2+ 클러스터만 채택 (최빈도 → canonical, 나머지 → aliases)
    5) 이미 용어집에 등록된 단어 포함 클러스터는 제외
    6) total_count >= min_frequency 만 반환

LLM 불필요, 순수 빈도 + 편집거리 기반.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Set

try:
    from rapidfuzz import fuzz
    from rapidfuzz.distance import Levenshtein
    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False


# ──────────────────────────────────────────────
# 단어 추출
# ──────────────────────────────────────────────

_RE_WORD = re.compile(r"[가-힣A-Za-z][가-힣A-Za-z0-9]{1,}")


# ──────────────────────────────────────────────
# 불용어
# ──────────────────────────────────────────────

STOPWORDS: Set[str] = {
    # 조사 (단독 추출 가능성)
    "에서", "에게", "한테", "으로", "라고", "이라고", "까지", "부터",
    # 접속/지시
    "그리고", "그래서", "그런데", "그러나", "하지만", "그러면", "따라서",
    "그럼", "근데", "그래", "아무튼", "어쨌든",
    "이것", "저것", "그것", "여기", "저기", "거기",
    "우리", "너희", "자기", "당신", "자신",
    # 일반 동사/형용사 활용형
    "있다", "없다", "하다", "되다", "이다", "아니다",
    "하는", "있는", "없는", "되는", "같은", "다른",
    "하고", "해서", "하면", "되고", "되면",
    "했다", "됐다", "있었", "없었",
    "합니다", "입니다", "됩니다", "합시다", "습니다", "하세요",
    "니다", "겠습니다", "있습니다", "없습니다",
    "해야", "되서", "돼서", "돼요", "됐어",
    "알겠습니다", "감사합니다", "수고하셨습니다",
    "그거", "이거", "저거",
    "이런", "저런", "그런", "이제", "아까", "나중",
    # 부사
    "정말", "진짜", "되게", "너무", "아주", "매우", "그냥", "계속",
    "먼저", "다음", "그때", "지금", "오늘", "어제", "내일",
    # 추임새 (2글자)
    "아니", "맞아", "네네",
    # 숫자 표기
    "하나", "둘째", "셋째", "넷째", "다섯", "여섯", "일곱", "여덟",
    "한번", "두번", "세번", "번째", "가지",
}


def _extract_words(text: str) -> List[str]:
    return _RE_WORD.findall(text)


def _build_frequency(words: Iterable[str]) -> Counter:
    return Counter(w for w in words if w not in STOPWORDS and len(w) >= 2)


def _is_similar(a: str, b: str, threshold: int) -> bool:
    """두 단어가 유사한지 판단.

    기본: fuzz.ratio >= threshold
    보정: 짧은 단어(둘 다 3자 이하) 는 Levenshtein 편집거리 1 이내면 유사 처리
          (한글 3자 단어 1글자 차이 = ratio 66.7 이라 놓치는 문제 보정)
    """
    if not _HAS_RAPIDFUZZ:
        return a == b
    score = fuzz.ratio(a, b)
    if score >= threshold:
        return True
    if max(len(a), len(b)) <= 3 and Levenshtein.distance(a, b) <= 1:
        return True
    return False


# ──────────────────────────────────────────────
# Union-Find
# ──────────────────────────────────────────────

class _UnionFind:
    def __init__(self, items: Iterable[str]):
        self.parent: Dict[str, str] = {x: x for x in items}

    def find(self, x: str) -> str:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            next_x = self.parent[x]
            self.parent[x] = root
            x = next_x
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


# ──────────────────────────────────────────────
# 메인 감지 함수
# ──────────────────────────────────────────────

def detect_term_variants(
    text: str,
    min_frequency: int = 2,
    similarity_threshold: int = 80,
    existing_canonicals: Optional[Set[str]] = None,
    existing_aliases: Optional[Set[str]] = None,
) -> List[Dict]:
    """텍스트에서 용어 변형 클러스터를 자동 감지한다.

    Args:
        text: 분석할 텍스트 (전사 결과)
        min_frequency: 클러스터 total_count 최소 임계치
        similarity_threshold: RapidFuzz 유사도 임계치 (0-100)
        existing_canonicals: 이미 용어집에 있는 canonical 집합 (제외)
        existing_aliases: 이미 용어집에 있는 alias 집합 (제외)

    Returns:
        [{"canonical": str, "aliases": [str], "total_count": int,
          "frequencies": {word: count}}] — total_count 내림차순
    """
    if not text or not _HAS_RAPIDFUZZ:
        return []

    existing_canon = set(existing_canonicals or [])
    existing_alias = set(existing_aliases or [])

    words = _extract_words(text)
    freq = _build_frequency(words)

    candidates = [w for w in freq if freq[w] >= 1]
    if len(candidates) < 2:
        return []

    uf = _UnionFind(candidates)
    for i, a in enumerate(candidates):
        if len(a) < 2:
            continue
        for b in candidates[i + 1:]:
            if len(b) < 2:
                continue
            # 길이 차이가 너무 크면 건너뜀 (최적화 + 오매칭 방지)
            if abs(len(a) - len(b)) > max(len(a), len(b)) // 2 + 1:
                continue
            if _is_similar(a, b, similarity_threshold):
                uf.union(a, b)

    clusters: Dict[str, List[str]] = defaultdict(list)
    for word in candidates:
        root = uf.find(word)
        clusters[root].append(word)

    result: List[Dict] = []
    for members in clusters.values():
        if len(members) < 2:
            continue

        # 이미 용어집에 등록된 단어 포함 → 제외
        if any(m in existing_canon or m in existing_alias for m in members):
            continue

        sorted_members = sorted(members, key=lambda w: (-freq[w], w))
        canonical = sorted_members[0]
        aliases = sorted_members[1:]
        total = sum(freq[m] for m in members)

        if total < min_frequency:
            continue

        result.append({
            "canonical": canonical,
            "aliases": aliases,
            "total_count": total,
            "frequencies": {m: freq[m] for m in sorted_members},
        })

    result.sort(key=lambda x: -x["total_count"])
    return result


def format_cluster_preview(cluster: Dict) -> str:
    """클러스터를 UI 용 Markdown 문자열로 포맷.

    예: "**Premo**(3) ← 프레모(2), 프리모(1)"
    """
    canonical = cluster["canonical"]
    freqs = cluster.get("frequencies", {})
    canon_cnt = freqs.get(canonical, 0)
    alias_parts = [f"{a}({freqs.get(a, 0)})" for a in cluster.get("aliases", [])]
    alias_str = ", ".join(alias_parts)
    return f"**{canonical}**({canon_cnt}) ← {alias_str}"
