#!/usr/bin/env python3
"""noma-scribe 벤치마크 스크립트.

v0.1 (raw mlx-whisper) vs v0.2 (whispermlx + VAD + postprocess) 비교.

사용법:
    source .venv/bin/activate
    python bench/benchmark.py 개발회의4.m4a
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── v0.2 엔진 import (monkey-patch 적용됨) ──
from core.engine import (
    DEFAULT_ASR_OPTIONS,
    DEFAULT_MODEL,
    _install_mlx_whisper_patch,
    transcribe as v2_transcribe,
)
from core.postprocess import clean_text

import mlx_whisper


# ──────────────────────────────────────────────
# 유틸리티
# ──────────────────────────────────────────────

def _count_repeated_words(text: str, threshold: int = 3) -> dict:
    """연속 반복 단어 패턴을 찾아 개수와 예시를 반환."""
    pattern = re.compile(r"(\S+?)(?:\s+\1){" + str(threshold - 1) + r",}")
    matches = pattern.findall(text)
    return {
        "count": len(matches),
        "examples": matches[:5],
    }


def _count_fillers(text: str) -> dict:
    """추임새(응/음/어/아/네/예) 출현 횟수."""
    fillers = ["응", "음", "어", "아", "네", "예"]
    counts = {}
    for f in fillers:
        c = len(re.findall(rf"\b{f}\b", text))
        if c > 0:
            counts[f] = c
    total = sum(counts.values())
    return {"total": total, "breakdown": counts}


def _text_stats(text: str) -> dict:
    words = text.split()
    chars = len(text)
    return {
        "char_count": chars,
        "word_count": len(words),
        "line_count": text.count("\n") + 1,
        "repeated_words": _count_repeated_words(text),
        "fillers": _count_fillers(text),
    }


# ──────────────────────────────────────────────
# v0.1 재현: raw mlx_whisper (monkey-patch 우회)
# ──────────────────────────────────────────────

def run_v01_baseline(audio_path: str, language: str = "ko") -> dict:
    """v0.1 스타일: mlx_whisper.transcribe 직접 호출 (VAD 없음, 기본 옵션)."""
    print("  [v0.1] mlx_whisper.transcribe (raw, no VAD, default options)...")

    # monkey-patch된 함수에서 원본을 추출
    patched_fn = mlx_whisper.transcribe
    original_fn = None
    if hasattr(patched_fn, "__closure__") and patched_fn.__closure__:
        for cell in patched_fn.__closure__:
            try:
                candidate = cell.cell_contents
                if callable(candidate) and candidate is not patched_fn:
                    original_fn = candidate
                    break
            except ValueError:
                pass

    if original_fn is None:
        print("  ⚠️  원본 mlx_whisper.transcribe를 추출할 수 없음. patched 버전으로 대체.")
        # 기본값을 명시적으로 넘겨서 v0.1 스타일 재현
        start = time.time()
        result = patched_fn(
            str(audio_path),
            path_or_hf_repo=DEFAULT_MODEL,
            language=language,
            verbose=False,
            condition_on_previous_text=True,
            compression_ratio_threshold=2.4,
            no_speech_threshold=0.6,
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        )
        elapsed = time.time() - start
    else:
        start = time.time()
        result = original_fn(
            str(audio_path),
            path_or_hf_repo=DEFAULT_MODEL,
            language=language,
            verbose=False,
        )
        elapsed = time.time() - start

    raw_text = result.get("text", "").strip()
    segments = result.get("segments", [])

    return {
        "mode": "v0.1_baseline",
        "description": "raw mlx_whisper.transcribe (no VAD, default options)",
        "elapsed_seconds": round(elapsed, 2),
        "text": raw_text,
        "segment_count": len(segments),
        "stats": _text_stats(raw_text),
    }


# ──────────────────────────────────────────────
# v0.2: whispermlx + VAD + postprocess
# ──────────────────────────────────────────────

def run_v02_engine(audio_path: str, language: str = "ko") -> dict:
    """v0.2 스타일: whispermlx pipeline + postprocess."""
    print("  [v0.2] whispermlx pipeline (VAD + monkey-patch ASR + postprocess)...")

    result = v2_transcribe(
        audio_path=str(audio_path),
        language=language,
        model=DEFAULT_MODEL,
    )

    return {
        "mode": "v0.2_whispermlx",
        "description": "whispermlx pipeline + silero VAD + monkey-patch ASR + postprocess",
        "elapsed_seconds": round(result.duration_seconds, 2),
        "text": result.text,
        "segment_count": len(result.segments),
        "stats": _text_stats(result.text),
        "asr_options": DEFAULT_ASR_OPTIONS,
    }


# ──────────────────────────────────────────────
# 비교 분석
# ──────────────────────────────────────────────

def compare(v01: dict, v02: dict) -> dict:
    """두 결과를 비교하여 요약 생성."""
    s1, s2 = v01["stats"], v02["stats"]

    time_ratio = v01["elapsed_seconds"] / max(v02["elapsed_seconds"], 0.01)
    repeat_diff = s1["repeated_words"]["count"] - s2["repeated_words"]["count"]
    filler_diff = s1["fillers"]["total"] - s2["fillers"]["total"]
    word_diff = s1["word_count"] - s2["word_count"]

    return {
        "time": {
            "v01_seconds": v01["elapsed_seconds"],
            "v02_seconds": v02["elapsed_seconds"],
            "ratio": round(time_ratio, 2),
            "winner": "v0.1" if v01["elapsed_seconds"] < v02["elapsed_seconds"] else "v0.2",
        },
        "hallucination": {
            "v01_repeated_words": s1["repeated_words"]["count"],
            "v02_repeated_words": s2["repeated_words"]["count"],
            "improvement": repeat_diff,
            "v01_examples": s1["repeated_words"]["examples"][:3],
            "v02_examples": s2["repeated_words"]["examples"][:3],
        },
        "fillers": {
            "v01_total": s1["fillers"]["total"],
            "v02_total": s2["fillers"]["total"],
            "improvement": filler_diff,
        },
        "length": {
            "v01_words": s1["word_count"],
            "v02_words": s2["word_count"],
            "v01_chars": s1["char_count"],
            "v02_chars": s2["char_count"],
            "word_difference": word_diff,
        },
        "segments": {
            "v01_segments": v01["segment_count"],
            "v02_segments": v02["segment_count"],
        },
    }


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python bench/benchmark.py <audio_file>")
        sys.exit(1)

    audio_path = Path(sys.argv[1])
    if not audio_path.exists():
        print(f"File not found: {audio_path}")
        sys.exit(1)

    print()
    print(f"  🎙️  noma-scribe 벤치마크")
    print(f"  ──────────────────────────")
    print(f"  파일: {audio_path.name} ({audio_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"  모델: {DEFAULT_MODEL}")
    print()

    # v0.1 baseline
    v01 = run_v01_baseline(str(audio_path))
    print(f"  [v0.1] 완료: {v01['elapsed_seconds']}초, {v01['stats']['word_count']}단어, "
          f"반복패턴 {v01['stats']['repeated_words']['count']}개")
    print()

    # v0.2 engine
    v02 = run_v02_engine(str(audio_path))
    print(f"  [v0.2] 완료: {v02['elapsed_seconds']}초, {v02['stats']['word_count']}단어, "
          f"반복패턴 {v02['stats']['repeated_words']['count']}개")
    print()

    # 비교
    comp = compare(v01, v02)

    # 결과 저장
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_file = results_dir / f"bench_{audio_path.stem}_{timestamp}.json"

    output = {
        "file": audio_path.name,
        "model": DEFAULT_MODEL,
        "timestamp": timestamp,
        "v01": {k: v for k, v in v01.items() if k != "text"},
        "v02": {k: v for k, v in v02.items() if k != "text"},
        "comparison": comp,
    }

    # 텍스트 샘플 (전체가 아닌 처음/끝 500자)
    for key, data in [("v01", v01), ("v02", v02)]:
        text = data["text"]
        output[key]["text_preview"] = {
            "first_500": text[:500],
            "last_500": text[-500:] if len(text) > 500 else "",
        }

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # 전체 텍스트도 별도 파일로 저장
    for key, data in [("v01", v01), ("v02", v02)]:
        txt_file = results_dir / f"bench_{audio_path.stem}_{timestamp}_{key}.txt"
        with open(txt_file, "w", encoding="utf-8") as f:
            f.write(data["text"])

    # 요약 출력
    print("  ══════════════════════════════")
    print("  📊 벤치마크 결과")
    print("  ══════════════════════════════")
    print()
    print(f"  ⏱️  시간:  v0.1 = {comp['time']['v01_seconds']}초  |  v0.2 = {comp['time']['v02_seconds']}초  ({comp['time']['ratio']}x)")
    print(f"  🔁 반복:  v0.1 = {comp['hallucination']['v01_repeated_words']}개  |  v0.2 = {comp['hallucination']['v02_repeated_words']}개  (▼{comp['hallucination']['improvement']})")
    if comp['hallucination']['v01_examples']:
        print(f"          v0.1 예시: {comp['hallucination']['v01_examples']}")
    if comp['hallucination']['v02_examples']:
        print(f"          v0.2 예시: {comp['hallucination']['v02_examples']}")
    print(f"  💬 추임새: v0.1 = {comp['fillers']['v01_total']}개  |  v0.2 = {comp['fillers']['v02_total']}개")
    print(f"  📝 분량:  v0.1 = {comp['length']['v01_words']}단어  |  v0.2 = {comp['length']['v02_words']}단어  (차이 {comp['length']['word_difference']})")
    print(f"  🧩 세그먼트: v0.1 = {comp['segments']['v01_segments']}개  |  v0.2 = {comp['segments']['v02_segments']}개")
    print()
    print(f"  📁 결과 저장: {result_file.name}")
    print()


if __name__ == "__main__":
    main()
