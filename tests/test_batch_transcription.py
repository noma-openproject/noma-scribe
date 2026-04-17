"""tests/test_batch_transcription.py — 배치 전사 경계 유틸 테스트."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.batch_transcription import (
    BatchTranscriptionOptions,
    mode_label_to_key,
    parse_time,
    transcribe_batch,
)


def test_parse_time_variants():
    assert parse_time("") is None
    assert parse_time("01:23") == "00:01:23"
    assert parse_time("01:02:03") == "01:02:03"
    assert parse_time("abc") is None


def test_mode_label_to_key():
    assert mode_label_to_key("🇰🇷 한국어 모드 — 설명") == "korean"
    assert mode_label_to_key("🔬 정밀 모드 — 설명") == "precise"
    assert mode_label_to_key("⚡ 빠른 모드 — 설명") == "fast"


def test_transcribe_batch_without_files_returns_warning():
    options = BatchTranscriptionOptions(
        mode_label="⚡ 빠른 모드 — 설명",
        two_pass_enabled=False,
        output_format="텍스트 (.txt)",
    )

    result = transcribe_batch(None, options, lambda text: [])

    assert result.text == ""
    assert result.download_path is None
    assert result.warnings == ["오디오 파일을 먼저 업로드해주세요."]
