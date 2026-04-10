"""whispermlx 기반 전사 엔진.

whispermlx는 WhisperX의 Apple Silicon 포크로, VAD + 청크 기반 전사를 제공한다.
내부에서 mlx-whisper를 호출하지만 whispermlx의 load_model이 받는 asr_options
중에서는 `initial_prompt`만 실제로 사용된다
(whispermlx/asr.py 참고: "asr_options: Dict of ASR options; only 'initial_prompt' is used.")

사용자가 요청한 나머지 파라미터 (compression_ratio_threshold, no_speech_threshold,
temperature, condition_on_previous_text=False)는 `mlx_whisper.transcribe`를
모듈 수준에서 monkey-patch하여 whispermlx 내부 호출 시점에 주입한다.

후처리: core.postprocess 의 필터가 자동으로 적용되어 반복 환각을 정리한다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import mlx_whisper  # whispermlx 의 백엔드이기도 함
import whispermlx

from core.postprocess import clean_segments, clean_text


# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────

DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"

#: whispermlx 내부에서 호출되는 mlx_whisper.transcribe 에 주입할 기본 ASR 옵션.
#: 사용자가 명시적으로 요청한 반복 환각 방지 파라미터.
DEFAULT_ASR_OPTIONS: Dict[str, Any] = {
    "compression_ratio_threshold": 2.0,   # 기본 2.4 → 2.0 (더 강하게 필터)
    "no_speech_threshold": 0.5,           # 기본 0.6 → 0.5
    "temperature": (0.0, 0.2, 0.4),       # 기본 (0.0,0.2,0.4,0.6,0.8,1.0) → 축소
    "condition_on_previous_text": False,  # 기본 True → False (반복 환각 방지)
}


# ──────────────────────────────────────────────
# Monkey-patch: mlx_whisper.transcribe
# ──────────────────────────────────────────────
#
# whispermlx/asr.py 의 MLXWhisperPipeline.transcribe 는 내부적으로
# `mlx_whisper.transcribe(audio_chunk, path_or_hf_repo=..., language=..., task=...,
#                          verbose=False, initial_prompt=..., word_timestamps=False)`
# 로 호출한다. 우리가 원하는 추가 옵션을 전달할 공식 경로가 없으므로,
# `mlx_whisper.transcribe` 자체를 한 번 교체한다. whispermlx 가 명시적으로 넘기는
# 인자(kwargs)가 우리 기본값을 override 하므로 side effect 는 없다.

_PATCH_ATTR = "_noma_scribe_patched"


def _install_mlx_whisper_patch() -> None:
    if getattr(mlx_whisper, _PATCH_ATTR, False):
        return

    _original_transcribe = mlx_whisper.transcribe

    def _patched_transcribe(audio, **kwargs):
        merged = {**DEFAULT_ASR_OPTIONS, **kwargs}
        return _original_transcribe(audio, **merged)

    mlx_whisper.transcribe = _patched_transcribe  # type: ignore[assignment]
    setattr(mlx_whisper, _PATCH_ATTR, True)


_install_mlx_whisper_patch()


# ──────────────────────────────────────────────
# 결과 데이터 클래스
# ──────────────────────────────────────────────

@dataclass
class TranscribeResult:
    """전사 결과를 담는 데이터 클래스."""
    text: str
    language: str
    duration_seconds: float  # 전사에 걸린 wall-clock 시간
    segments: List[dict] = field(default_factory=list)
    audio_path: str = ""


# ──────────────────────────────────────────────
# 모델 싱글턴 캐시
# ──────────────────────────────────────────────
#
# VAD (silero) 모델 다운로드가 첫 호출에서 수초 걸리므로 (model, language) 쌍으로
# 파이프라인을 캐시한다. 같은 프로세스 내에서 반복 전사 시 로드 비용을 없앤다.

_PIPELINE_CACHE: Dict[tuple, Any] = {}


def _get_pipeline(
    model: str,
    language: Optional[str],
    initial_prompt: Optional[str],
):
    """whispermlx 파이프라인을 캐시 키(모델 + 언어 + 프롬프트)별로 반환."""
    key = (model, language, initial_prompt or "")
    if key in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[key]

    asr_options: Optional[Dict[str, Any]] = None
    if initial_prompt:
        asr_options = {"initial_prompt": initial_prompt}

    pipeline = whispermlx.load_model(
        whisper_arch=model,
        device="cpu",             # VAD 용 디바이스 — MLX 추론은 자동으로 Metal 사용
        language=language,
        asr_options=asr_options,
        vad_method="silero",      # pyannote 는 HF 토큰 필요, silero 는 공개
    )
    _PIPELINE_CACHE[key] = pipeline
    return pipeline


# ──────────────────────────────────────────────
# 전사 엔트리포인트
# ──────────────────────────────────────────────

def transcribe(
    audio_path: str,
    language: str = "ko",
    model: str = DEFAULT_MODEL,
    initial_prompt: Optional[str] = None,
    word_timestamps: bool = False,  # 현재 버전은 세그먼트 단위 타임스탬프만 제공
    verbose: bool = False,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> TranscribeResult:
    """오디오 파일을 whispermlx 파이프라인으로 전사한다.

    Args:
        audio_path: 오디오 파일 경로
        language: 전사 언어 코드 (기본 "ko"). None 이면 자동 감지.
        model: 모델 식별자 (short name 또는 HF repo ID)
        initial_prompt: 도메인 용어 힌트 — 전문용어 인식률 향상
        word_timestamps: 호환성 유지용 플래그. whispermlx 는 내부적으로 False
            로 호출하므로 현재는 효과 없음. UI 의 "타임스탬프 포함" 옵션은
            세그먼트 단위 start/end 를 그대로 사용한다.
        verbose: whispermlx 에 전달할 로깅 플래그
        progress_callback: 0~100 범위의 진행률을 받는 콜백 (whispermlx 가 VAD
            청크 단위로 호출). Gradio Progress 업데이트용.

    Returns:
        후처리된 텍스트와 세그먼트가 담긴 TranscribeResult
    """
    pipeline = _get_pipeline(model, language, initial_prompt)

    start = time.time()
    raw_result = pipeline.transcribe(
        str(audio_path),
        language=language,
        verbose=verbose,
        progress_callback=progress_callback,
    )
    elapsed = time.time() - start

    segments: List[dict] = list(raw_result.get("segments", []))
    detected_language = raw_result.get("language", language or "ko")

    # 원시 텍스트 = 세그먼트 텍스트를 공백으로 join
    raw_text = " ".join(seg.get("text", "").strip() for seg in segments).strip()

    # 후처리 — 반복 환각 필터
    cleaned_text = clean_text(raw_text)
    cleaned_segments = clean_segments(segments)

    return TranscribeResult(
        text=cleaned_text,
        language=detected_language,
        duration_seconds=elapsed,
        segments=cleaned_segments,
        audio_path=str(audio_path),
    )


# ──────────────────────────────────────────────
# 결과 저장
# ──────────────────────────────────────────────

def save_result(
    result: TranscribeResult,
    output_path: Path,
    include_timestamps: bool = False,
) -> None:
    """전사 결과를 텍스트 파일로 저장한다.

    Args:
        result: 전사 결과
        output_path: 저장할 파일 경로
        include_timestamps: True 이면 세그먼트별 [시작 → 끝] 타임스탬프 포함
    """
    output_path = Path(output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        if include_timestamps and result.segments:
            for seg in result.segments:
                start = _format_timestamp(float(seg.get("start", 0.0)))
                end = _format_timestamp(float(seg.get("end", 0.0)))
                text = (seg.get("text") or "").strip()
                f.write(f"[{start} → {end}] {text}\n")
        else:
            f.write((result.text or "") + "\n")


def _format_timestamp(seconds: float) -> str:
    """초를 HH:MM:SS 또는 MM:SS 형태로 변환한다."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"
