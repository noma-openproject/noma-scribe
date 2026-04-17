"""사용자 용어집 시스템.

Whisper 전사 결과에서 자주 틀리는 고유명사/브랜드명을 사용자가 정의한
표준형(canonical)으로 치환한다.

저장 위치: ~/.noma-scribe/glossary.json
형식:
    {
      "terms": [
        {"canonical": "Premo", "aliases": ["프레모", "프리모", "프레머"]},
        {"canonical": "Noma", "aliases": ["노마", "노머"]}
      ]
    }

치환 전략:
    1) aliases 에 명시된 단어는 정확 매칭으로 canonical 치환 (우선)
    2) 그 외 단어 중 canonical 과 유사도 >= fuzzy_threshold 인 것을 치환
    3) 한국어 조사(은/는/이/가/을/를/의/도/로/와/과/에/에서/에게/한테/부터/까지/보다 등)는
       분리해서 보존한다 → "Premo가" → "Premo가" 유지
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from rapidfuzz import fuzz, process
    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False


# ──────────────────────────────────────────────
# 경로
# ──────────────────────────────────────────────

DEFAULT_GLOSSARY_PATH = Path.home() / ".noma-scribe" / "glossary.json"


# ──────────────────────────────────────────────
# 한국어 조사 패턴
# ──────────────────────────────────────────────
_KR_PARTICLES = [
    "에게서", "한테서", "으로부터", "에서부터",
    "에게", "한테", "께서", "이라고", "라고",
    "으로", "에서",
    "부터", "까지", "보다", "마저", "조차",
    "입니다", "이다", "이라", "이네",
    "은", "는", "이", "가", "을", "를", "의", "도", "로",
    "와", "과", "에", "야", "여", "께",
    "만", "뿐", "랑",
]
_KR_PARTICLES = sorted(set(_KR_PARTICLES), key=len, reverse=True)
_PARTICLE_RE = re.compile(
    r"(?P<head>\S+?)(?P<particle>(?:" + "|".join(re.escape(p) for p in _KR_PARTICLES) + r"))$"
)


def _split_particle(word: str) -> Tuple[str, str]:
    """단어를 (본체, 조사) 로 분리. 조사가 없으면 (word, '')."""
    m = _PARTICLE_RE.fullmatch(word)
    if m:
        head = m.group("head")
        if len(head) >= 1:
            return head, m.group("particle")
    return word, ""


# ──────────────────────────────────────────────
# 용어집 I/O
# ──────────────────────────────────────────────

def _default_data() -> Dict:
    return {"terms": []}


def load_glossary(path: Optional[Path] = None) -> Dict:
    """용어집을 로딩한다. 파일이 없거나 깨져있으면 빈 용어집 반환."""
    p = Path(path) if path else DEFAULT_GLOSSARY_PATH
    if not p.exists():
        return _default_data()
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "terms" not in data:
            return _default_data()
        return data
    except (json.JSONDecodeError, OSError):
        return _default_data()


def save_glossary(data: Dict, path: Optional[Path] = None) -> None:
    """용어집을 저장한다 (디렉토리 없으면 생성)."""
    p = Path(path) if path else DEFAULT_GLOSSARY_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def add_term(canonical: str, aliases: Optional[List[str]] = None,
             path: Optional[Path] = None) -> Dict:
    """용어집에 새 항목을 추가한다 (중복 canonical 은 덮어씌움)."""
    data = load_glossary(path)
    aliases_list = list(aliases or [])
    data["terms"] = [t for t in data.get("terms", []) if t.get("canonical") != canonical]
    data["terms"].append({"canonical": canonical, "aliases": aliases_list})
    save_glossary(data, path)
    return data


def remove_term(canonical: str, path: Optional[Path] = None) -> Dict:
    """용어집에서 canonical 항목을 제거한다."""
    data = load_glossary(path)
    data["terms"] = [t for t in data.get("terms", []) if t.get("canonical") != canonical]
    save_glossary(data, path)
    return data


# ──────────────────────────────────────────────
# 치환 로직
# ──────────────────────────────────────────────

_WORD_RE = re.compile(r"\S+")


def apply_glossary(
    text: str,
    glossary: Optional[Dict] = None,
    fuzzy_threshold: int = 85,
) -> str:
    """텍스트에 용어집 치환을 적용한다.

    순서:
      1) aliases 정확 매칭 → canonical (조사 보존)
      2) 남은 단어 중 canonical 과 유사도 >= fuzzy_threshold 인 것 → canonical (조사 보존)

    Args:
        text: 원본 텍스트
        glossary: 용어집 dict. None 이면 DEFAULT_GLOSSARY_PATH 에서 로드.
        fuzzy_threshold: RapidFuzz 유사도 임계치 (0-100). rapidfuzz 미설치 시 정확 매칭만.

    Returns:
        치환된 텍스트
    """
    if not text:
        return text

    if glossary is None:
        glossary = load_glossary()

    terms = glossary.get("terms", [])
    if not terms:
        return text

    # alias → canonical 역매핑
    alias_to_canonical: Dict[str, str] = {}
    canonicals: List[str] = []
    for t in terms:
        canonical = t.get("canonical", "")
        if not canonical:
            continue
        canonicals.append(canonical)
        for alias in t.get("aliases", []) or []:
            if alias:
                alias_to_canonical[alias] = canonical
        # canonical 자체도 alias 로 등록 (정확 매칭 fast-path)
        alias_to_canonical[canonical] = canonical

    def _transform(match: re.Match) -> str:
        word = match.group(0)
        # 구두점 제거한 head 로 치환 판단 (문장 끝 마침표 등은 보존)
        trailing_punct = ""
        core = word
        while core and core[-1] in ".,!?;:·、。":
            trailing_punct = core[-1] + trailing_punct
            core = core[:-1]
        if not core:
            return word

        head, particle = _split_particle(core)

        # 1) 정확 매칭 (alias 포함)
        if head in alias_to_canonical:
            return alias_to_canonical[head] + particle + trailing_punct

        # 2) RapidFuzz 유사도 매칭
        if _HAS_RAPIDFUZZ and head and len(head) >= 2:
            best = process.extractOne(
                head, canonicals, scorer=fuzz.ratio,
                score_cutoff=fuzzy_threshold,
            )
            if best is not None:
                canonical, score, _idx = best
                if canonical != head:
                    return canonical + particle + trailing_punct

        return word

    return _WORD_RE.sub(_transform, text)
