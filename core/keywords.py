"""전사 결과 키워드 빈도 분석.

LLM 없이 순수 텍스트 빈도 분석으로 Top N 명사/고유명사를 추출한다.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import List, Set, Tuple


# 불용어 (조사, 어미, 대명사, 접속사, 부사 등)
STOPWORDS: Set[str] = {
    # 조사
    "은", "는", "이", "가", "을", "를", "에", "의", "도", "로", "와", "과",
    "에서", "까지", "부터", "에게", "한테", "보다", "처럼", "같이",
    # 대명사/지시어
    "한", "그", "저", "것", "이것", "저것", "그것", "여기", "거기", "저기",
    "우리", "제가", "내가", "너가", "걔가", "자기",
    # 부사/접속사
    "수", "등", "더", "안", "좀", "다", "또", "막", "잘", "못", "매우", "정말",
    "진짜", "사실", "아마", "혹시", "그냥", "이미", "아직", "계속", "다시",
    "지금", "이제", "아까", "나중", "먼저", "다음", "항상", "가끔",
    "그래서", "그러면", "그런데", "근데", "하지만", "그리고", "또한",
    "왜냐면", "때문에", "대해서", "통해서", "위해서",
    # 추임새
    "요", "네", "응", "음", "어", "아", "예", "그래", "맞아",
    # 동사/형용사 활용형 (고빈도)
    "하다", "있다", "없다", "되다", "보다", "가다", "오다", "주다", "알다",
    "하는", "있는", "없는", "되는", "하고", "해서", "하면", "했는데", "하는데",
    "있는데", "없는데", "같은데", "인데", "있어서", "없어서", "해야",
    "합니다", "됩니다", "합니다", "입니다", "습니다", "니다",
    "만들다", "나오다", "들어가다", "생각하다", "말하다",
    "해야", "돼요", "할게", "하겠습니다", "알겠습니다",
    "감사합니다", "수고하셨습니다",
    # 숫자/단위 (단독)
    "개", "번", "건", "가지",
}

# 한글 단어 (2글자 이상)
_RE_KOREAN_WORD = re.compile(r"[가-힣]{2,}")

# 영어 단어 (2글자 이상)
_RE_ENGLISH_WORD = re.compile(r"[A-Za-z][A-Za-z0-9._-]{1,}")


def extract_keywords(
    text: str,
    top_n: int = 15,
    min_count: int = 2,
) -> List[Tuple[str, int]]:
    """텍스트에서 고빈도 명사/고유명사를 추출한다.

    Args:
        text: 전사 결과 텍스트
        top_n: 반환할 최대 키워드 수
        min_count: 최소 출현 횟수

    Returns:
        (단어, 빈도) 튜플의 리스트, 빈도 내림차순
    """
    counter: Counter = Counter()

    # 한글 단어
    for m in _RE_KOREAN_WORD.finditer(text):
        word = m.group()
        if word not in STOPWORDS and len(word) >= 2:
            counter[word] += 1

    # 영어 단어
    short_en = {"is", "am", "an", "at", "be", "by", "do", "go", "if", "in",
                "it", "me", "my", "no", "of", "on", "or", "so", "to", "up",
                "us", "we", "the", "and", "for", "not", "but", "are", "was",
                "has", "had", "her", "his", "its", "let", "may", "new", "now",
                "old", "see", "way", "who", "did", "get", "how", "man", "our",
                "out", "say", "she", "too", "use", "yes", "yet", "can", "all"}
    for m in _RE_ENGLISH_WORD.finditer(text):
        word = m.group()
        if word.lower() not in short_en:
            counter[word] += 1

    # min_count 필터 + top_n
    filtered = [(w, c) for w, c in counter.most_common() if c >= min_count]
    return filtered[:top_n]


def format_keywords(keywords: List[Tuple[str, int]]) -> str:
    """키워드 리스트를 사람이 읽기 좋은 문자열로 포맷한다.

    예: "프로젝트(12) · API(8) · 배포(5) · 테스트(4)"
    """
    if not keywords:
        return ""
    return " · ".join(f"{word}({count})" for word, count in keywords)
