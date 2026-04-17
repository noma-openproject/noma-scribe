"""tests/test_two_pass.py — 2-pass 전사 옵션 전달 테스트."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.engine import TranscribeResult
import core.two_pass as two_pass


def test_two_pass_transcribe_passes_postprocess_options(monkeypatch):
    calls = []

    def fake_transcribe(*args, **kwargs):
        calls.append(kwargs)
        return TranscribeResult(
            text="프레모가 좋다.",
            language="ko",
            duration_seconds=0.1,
            segments=[{"start": 0.0, "end": 1.0, "text": "프레모가 좋다."}],
            audio_path="sample.m4a",
        )

    monkeypatch.setattr(two_pass, "transcribe", fake_transcribe)
    monkeypatch.setattr(two_pass, "extract_terms", lambda text: ["Premo"])
    monkeypatch.setattr(two_pass, "build_prompt_from_terms", lambda terms, max_length=400: "Premo")

    glossary = {"terms": [{"canonical": "Premo", "aliases": ["프레모"]}]}

    two_pass.two_pass_transcribe(
        audio_path="sample.m4a",
        use_glossary=False,
        use_korean_norm=False,
        glossary=glossary,
    )

    assert len(calls) == 2
    assert calls[0]["use_glossary"] is False
    assert calls[0]["use_korean_norm"] is False
    assert calls[0]["glossary"] == glossary
    assert calls[1]["use_glossary"] is False
    assert calls[1]["use_korean_norm"] is False
    assert calls[1]["glossary"] == glossary
