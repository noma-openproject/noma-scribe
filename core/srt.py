"""SRT 자막 포맷 생성기."""

from __future__ import annotations

from pathlib import Path
from typing import List


def _format_srt_timestamp(seconds: float) -> str:
    """초를 SRT 타임스탬프 형식으로 변환한다.

    예: 3661.5 → "01:01:01,500"
    """
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = round((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def segments_to_srt(segments: List[dict]) -> str:
    """세그먼트 리스트를 SRT 문자열로 변환한다.

    segment 스키마: {"start": float, "end": float, "text": str, ...}
    """
    lines = []
    for idx, seg in enumerate(segments, 1):
        start = _format_srt_timestamp(float(seg.get("start", 0.0)))
        end = _format_srt_timestamp(float(seg.get("end", 0.0)))
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        lines.append(f"{idx}")
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")  # 빈 줄 구분
    return "\n".join(lines)


def save_srt(segments: List[dict], output_path: Path) -> None:
    """세그먼트 리스트를 SRT 파일로 저장한다."""
    output_path = Path(output_path)
    content = segments_to_srt(segments)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
