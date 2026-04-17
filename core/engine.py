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

from core.postprocess import clean_segments, clean_text, full_postprocess


# ──────────────────────────────────────────────
# 모델 맵
# ──────────────────────────────────────────────

# 프로젝트 루트 (engine.py 기준 한 단계 위)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS = {
    "fast": {
        "path": "mlx-community/whisper-large-v3-turbo",
        "label": "⚡ 빠른 모드",
        "desc": "다국어 범용, 한영 혼용에 적합",
    },
    "korean": {
        "path": str(_PROJECT_ROOT / "models" / "ko-turbo"),
        "label": "🇰🇷 한국어 모드",
        "desc": "한국어 회의에 최적, 빠른 모드보다 빠르고 정확",
    },
    "precise": {
        "path": "mlx-community/whisper-large-v3-mlx",
        "label": "🔬 정밀 모드",
        "desc": "최고 정확도, 2~3배 느림",
    },
}


def resolve_model_path(mode: str) -> str:
    """모드 키 → 실제 모델 경로.

    로컬 경로이면 존재 여부를 확인하고, HF repo ID 이면 그대로 반환.
    """
    info = MODELS.get(mode, MODELS["fast"])
    path = info["path"]
    # 절대 경로 or "models/"로 시작 → 로컬
    if path.startswith("/") or path.startswith("models"):
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = _PROJECT_ROOT / resolved
        if not resolved.exists():
            raise FileNotFoundError(
                f"로컬 모델을 찾을 수 없습니다: {resolved}\n"
                f"한국어 모델을 설치하려면 README를 참조하세요."
            )
        return str(resolved)
    # HF repo ID
    return path


def is_model_available(mode: str) -> bool:
    """모델이 사용 가능한지 확인. 로컬 모델은 경로 존재 확인, HF는 항상 True."""
    info = MODELS.get(mode)
    if not info:
        return False
    path = info["path"]
    if path.startswith("/") or path.startswith("models"):
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = _PROJECT_ROOT / resolved
        return resolved.exists()
    return True


# ──────────────────────────────────────────────
# ffmpeg 구간 슬라이싱
# ──────────────────────────────────────────────

def slice_audio(
    audio_path: str,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
) -> str:
    """ffmpeg 로 오디오의 특정 구간을 잘라낸다.

    Args:
        audio_path: 원본 오디오 파일 경로
        start_time: 시작 시간 (MM:SS 또는 HH:MM:SS). None 이면 처음부터.
        end_time: 끝 시간 (MM:SS 또는 HH:MM:SS). None 이면 끝까지.

    Returns:
        잘려진 임시 파일 경로. 호출자가 정리 책임.
        start_time 과 end_time 모두 None 이면 원본 경로 그대로 반환.
    """
    if not start_time and not end_time:
        return audio_path

    import shutil
    import subprocess
    import tempfile

    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg 가 설치되어 있지 않습니다.")

    suffix = Path(audio_path).suffix or ".m4a"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.close()

    cmd = ["ffmpeg", "-y", "-i", str(audio_path)]
    if start_time:
        cmd += ["-ss", start_time]
    if end_time:
        cmd += ["-to", end_time]
    cmd += ["-c", "copy", "-loglevel", "error", tmp.name]

    subprocess.run(cmd, check=True, timeout=60)
    return tmp.name


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
    # v0.6 통합 후처리 옵션
    use_glossary: bool = True,
    use_korean_norm: bool = True,
    glossary: Optional[dict] = None,
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
        use_glossary: 사용자 용어집 치환 여부 (v0.6). 기본 True.
        use_korean_norm: KSS 기반 한국어 정규화 여부 (v0.6). 기본 True.
        glossary: 용어집 dict (None 이면 DEFAULT_GLOSSARY_PATH 에서 로드)

    Returns:
        후처리된 텍스트와 세그먼트가 담긴 TranscribeResult.
        text 필드는 full_postprocess 파이프라인을 거치고,
        segments 는 반복 환각만 정리된 세그먼트별 원본에 가까운 텍스트.
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

    # v0.6 통합 후처리: clean_hallucinations → glossary → KSS normalize/spacing → paragraphs
    cleaned_text = full_postprocess(
        raw_text,
        use_glossary=use_glossary,
        use_korean_norm=use_korean_norm,
        glossary=glossary,
    )
    # 세그먼트는 타임스탬프 정합성을 위해 반복 환각만 정리 (용어집/문단 분리는 하지 않음)
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


# ──────────────────────────────────────────────
# 단어 단위 정렬 (word_timestamps)
# ──────────────────────────────────────────────
#
# whispermlx.align() 은 wav2vec2 기반 forced alignment 로
# 세그먼트 → 단어별 start/end 타임스탬프를 생성한다.
# 한국어: "kresnik/wav2vec2-large-xlsr-korean" 자동 선택.

_ALIGN_MODEL_CACHE: Dict[str, tuple] = {}


def align_words(
    result: TranscribeResult,
    device: str = "cpu",
) -> dict:
    """전사 결과에 단어 단위 타임스탬프를 부여한다.

    Args:
        result: transcribe() 가 반환한 TranscribeResult
        device: torch 디바이스 ("cpu" 권장, MPS 는 wav2vec2 호환 이슈)

    Returns:
        whispermlx AlignedTranscriptionResult dict
        {"segments": [...], "word_segments": [...]}
    """
    language = result.language or "ko"

    # align 모델 캐시
    if language not in _ALIGN_MODEL_CACHE:
        model, metadata = whispermlx.load_align_model(
            language_code=language,
            device=device,
        )
        _ALIGN_MODEL_CACHE[language] = (model, metadata)
    else:
        model, metadata = _ALIGN_MODEL_CACHE[language]

    aligned = whispermlx.align(
        transcript=result.segments,
        model=model,
        align_model_metadata=metadata,
        audio=result.audio_path,
        device=device,
    )

    return aligned


# ──────────────────────────────────────────────
# 화자 분리 (diarize) — 조건부 활성화
# ──────────────────────────────────────────────
#
# pyannote/speaker-diarization-community-1 은 gated model 이라
# HuggingFace 토큰이 필요하다. 토큰이 없으면 graceful skip.
# 현재 단계에서는 비활성화가 기본이다.

def diarize(
    audio_path: str,
    hf_token: Optional[str] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    device: str = "mps",
) -> Optional["pd.DataFrame"]:
    """화자 분리를 수행한다. HF 토큰이 없으면 None 반환.

    Args:
        audio_path: 오디오 파일 경로
        hf_token: HuggingFace 토큰 (pyannote gated model 접근용)
        min_speakers: 최소 화자 수 (None = 자동)
        max_speakers: 최대 화자 수 (None = 자동)
        device: torch 디바이스 ("mps" 권장 on Apple Silicon)

    Returns:
        화자 분리 DataFrame 또는 None (토큰 없거나 실패 시)
    """
    import os as _os

    token = hf_token or _os.environ.get("HF_TOKEN") or _os.environ.get("HUGGINGFACE_HUB_TOKEN")

    if not token:
        return None

    try:
        from whispermlx.diarize import DiarizationPipeline

        pipeline = DiarizationPipeline(
            token=token,
            device=device,
        )
        diarize_df = pipeline(
            audio_path,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        return diarize_df
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"화자 분리 실패 (무시): {e}")
        return None


def assign_speakers(
    transcript_result: dict,
    diarize_df,
) -> dict:
    """화자 분리 결과를 전사 결과에 매핑한다.

    Args:
        transcript_result: align_words() 또는 transcribe() 결과 dict
        diarize_df: diarize() 가 반환한 DataFrame

    Returns:
        화자 라벨이 붙은 transcript_result
    """
    if diarize_df is None:
        return transcript_result

    return whispermlx.assign_word_speakers(diarize_df, transcript_result)
