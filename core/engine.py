"""mlx-whisper 기반 전사 엔진."""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import mlx_whisper


# 기본 모델: Apple Silicon에서 속도/정확도 최적 밸런스
DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"


@dataclass
class TranscribeResult:
    """전사 결과를 담는 데이터 클래스."""
    text: str
    language: str
    duration_seconds: float  # 전사에 걸린 시간
    segments: List[dict] = field(default_factory=list)
    audio_path: str = ""


def transcribe(
    audio_path: str,
    language: str = "ko",
    model: str = DEFAULT_MODEL,
    initial_prompt: Optional[str] = None,
    word_timestamps: bool = False,
    verbose: bool = False,
) -> TranscribeResult:
    """오디오 파일을 전사한다.
    
    Args:
        audio_path: 오디오 파일 경로
        language: 전사 언어 코드 (기본: "ko")
        model: Whisper 모델 경로 또는 HuggingFace 리포
        initial_prompt: 도메인 용어 힌트 (전문용어 인식률 향상)
        word_timestamps: 단어 단위 타임스탬프 생성 여부
        verbose: 세그먼트별 실시간 출력 여부
    
    Returns:
        TranscribeResult 객체
    """
    # 전사 옵션 구성
    options = {
        "path_or_hf_repo": model,
        "language": language,
        "word_timestamps": word_timestamps,
        "verbose": verbose,
    }
    
    if initial_prompt:
        options["initial_prompt"] = initial_prompt
    
    # 전사 실행 및 시간 측정
    start = time.time()
    result = mlx_whisper.transcribe(str(audio_path), **options)
    elapsed = time.time() - start
    
    return TranscribeResult(
        text=result.get("text", "").strip(),
        language=result.get("language", language),
        duration_seconds=elapsed,
        segments=result.get("segments", []),
        audio_path=str(audio_path),
    )


def save_result(result: TranscribeResult, output_path: Path, include_timestamps: bool = False) -> None:
    """전사 결과를 텍스트 파일로 저장한다.
    
    Args:
        result: 전사 결과
        output_path: 저장할 파일 경로
        include_timestamps: True이면 세그먼트별 타임스탬프 포함
    """
    with open(output_path, "w", encoding="utf-8") as f:
        if include_timestamps and result.segments:
            for seg in result.segments:
                start = _format_timestamp(seg["start"])
                end = _format_timestamp(seg["end"])
                text = seg["text"].strip()
                f.write(f"[{start} → {end}] {text}\n")
        else:
            f.write(result.text + "\n")


def _format_timestamp(seconds: float) -> str:
    """초를 HH:MM:SS 형태로 변환한다."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"
