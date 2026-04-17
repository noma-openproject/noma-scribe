"""2-pass 자동 용어 추출 전사 (정밀 모드 전용).

LLM 사용하지 않음.
1차: 프롬프트 없이 전사
2차: 1차에서 추출한 도메인 용어를 initial_prompt에 넣고 재전사

용어 추출 기준:
- 2회 이상 등장하는 2글자 이상 한글 단어 중 불용어가 아닌 것
- 영어/영문 단어 전부 (API, SDK, CRM 등)
- 숫자+단위 조합 (500만원, 2주 등)
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Callable, List, Optional, Set

from core.engine import TranscribeResult, transcribe


# ──────────────────────────────────────────────
# 불용어
# ──────────────────────────────────────────────

STOPWORDS: Set[str] = {
    # 조사/어미
    "은", "는", "이", "가", "을", "를", "에", "의", "도", "로", "와", "과",
    "한", "그", "저", "것", "수", "등", "더", "안", "좀", "다", "또", "막",
    "지금", "여기", "거기", "요", "네", "응", "음", "어", "아", "예",
    "하고", "해서", "하면", "그래서", "그러면", "그런데", "근데", "왜냐면",
    "이거", "저거", "그거", "어떻게", "이렇게", "저렇게", "그렇게",
    "했는데", "하는데", "있는데", "없는데", "같은데", "인데",
    "때문에", "대해서", "통해서", "위해서", "있어서", "없어서",
    "그래", "그럼", "아니", "맞아", "진짜", "정말", "사실",
    "먼저", "다음", "이제", "아까", "나중", "지금",
    "우리", "제가", "내가", "너가", "걔가",
    "하나", "둘", "셋", "넷", "다섯",
    "해야", "돼요", "할게", "합니다", "하겠습니다",
    "알겠습니다", "감사합니다", "수고하셨습니다",
}

# 일반 동사/형용사 어간 (빈도가 높지만 도메인 특화가 아닌 것)
COMMON_VERBS: Set[str] = {
    "하다", "있다", "없다", "되다", "보다", "가다", "오다", "주다",
    "알다", "만들다", "나오다", "들어가다", "생각하다", "말하다",
}


# ──────────────────────────────────────────────
# 용어 추출
# ──────────────────────────────────────────────

# 영어 단어 (2글자 이상, 대소문자 혼합/전부 대문자 포함)
_RE_ENGLISH = re.compile(r"[A-Za-z][A-Za-z0-9._-]{1,}")

# 숫자+단위 조합
_RE_NUM_UNIT = re.compile(r"\d+(?:\.\d+)?[만억천백십]?[원달러개월주일년시간분초%]")

# 한글 단어 (2글자 이상)
_RE_KOREAN = re.compile(r"[가-힣]{2,}")


def extract_terms(text: str, min_count: int = 2) -> List[str]:
    """전사 텍스트에서 도메인 용어 후보를 추출한다.

    Returns:
        빈도 내림차순으로 정렬된 용어 리스트
    """
    terms: Counter = Counter()

    # 1) 영어 단어 전부 (1회만 나와도 포함)
    for m in _RE_ENGLISH.finditer(text):
        word = m.group()
        # 너무 짧은 관사/전치사 제외
        if len(word) >= 2 and word.lower() not in {"is", "am", "an", "at", "be", "by",
                                                     "do", "go", "if", "in", "it", "me",
                                                     "my", "no", "of", "on", "or", "so",
                                                     "to", "up", "us", "we"}:
            terms[word] += 1

    # 2) 숫자+단위 (2회 이상)
    for m in _RE_NUM_UNIT.finditer(text):
        terms[m.group()] += 1

    # 3) 한글 단어 (2회 이상, 불용어 제외)
    for m in _RE_KOREAN.finditer(text):
        word = m.group()
        if word not in STOPWORDS and word not in COMMON_VERBS:
            terms[word] += 1

    # 필터: 영어는 1회 이상, 나머지는 min_count 이상
    filtered = []
    for term, count in terms.most_common():
        is_english = _RE_ENGLISH.fullmatch(term) is not None
        if is_english or count >= min_count:
            filtered.append(term)

    return filtered


def build_prompt_from_terms(terms: List[str], max_length: int = 400) -> str:
    """용어 리스트를 공백 구분 프롬프트 문자열로 변환한다.

    Whisper의 initial_prompt 권장 길이(~224 토큰)를 고려하여 max_length로 제한.
    """
    parts = []
    current_len = 0
    for term in terms:
        if current_len + len(term) + 1 > max_length:
            break
        parts.append(term)
        current_len += len(term) + 1
    return " ".join(parts)


# ──────────────────────────────────────────────
# 2-pass 전사
# ──────────────────────────────────────────────

def two_pass_transcribe(
    audio_path: str,
    language: str = "ko",
    model: str = "mlx-community/whisper-large-v3-turbo",
    progress_callback: Optional[Callable[[float], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
    use_glossary: bool = True,
    use_korean_norm: bool = True,
    glossary: Optional[dict] = None,
) -> TranscribeResult:
    """2-pass 자동 용어 추출 전사.

    Args:
        audio_path: 오디오 파일 경로
        language: 언어 코드
        model: Whisper 모델
        progress_callback: 0~100 진행률 콜백
        status_callback: 상태 메시지 콜백 ("1차 전사 중..." 등)
        use_glossary: 사용자 용어집 치환 여부
        use_korean_norm: KSS 기반 한국어 정규화 여부
        glossary: 용어집 dict (None 이면 기본 경로에서 로드)

    Returns:
        2차 전사 결과 (TranscribeResult)
    """
    def _status(msg: str):
        if status_callback:
            status_callback(msg)

    def _progress(pct: float):
        if progress_callback:
            progress_callback(pct)

    # ── 1차 전사: 프롬프트 없이 ──
    _status("1차 전사 중...")

    def _pass1_progress(pct: float):
        # 1차 = 0~40%
        _progress(pct * 0.4)

    pass1_result = transcribe(
        audio_path=audio_path,
        language=language,
        model=model,
        initial_prompt=None,
        progress_callback=_pass1_progress,
        use_glossary=use_glossary,
        use_korean_norm=use_korean_norm,
        glossary=glossary,
    )

    # ── 용어 추출 ──
    _status("용어 추출 중...")
    _progress(42)

    terms = extract_terms(pass1_result.text)
    prompt = build_prompt_from_terms(terms) if terms else None

    _progress(45)

    if not prompt:
        # 추출할 용어가 없으면 1차 결과 그대로 반환
        _status("용어 없음 — 1차 결과 사용")
        _progress(100)
        return pass1_result

    # ── 2차 전사: 추출 용어를 프롬프트에 ──
    _status(f"2차 전사 중... (용어 {len(terms)}개 적용)")

    def _pass2_progress(pct: float):
        # 2차 = 45~95%
        _progress(45 + pct * 0.50)

    pass2_result = transcribe(
        audio_path=audio_path,
        language=language,
        model=model,
        initial_prompt=prompt,
        progress_callback=_pass2_progress,
        use_glossary=use_glossary,
        use_korean_norm=use_korean_norm,
        glossary=glossary,
    )

    _status("후처리 중...")
    _progress(97)

    _progress(100)
    _status("완료")

    return pass2_result
