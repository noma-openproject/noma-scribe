"""런타임 진단 로그와 단계별 소요 시간 포맷터."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Mapping, Optional


LOG_DIR = Path.home() / ".noma-scribe" / "logs"


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_seconds(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    if minutes < 60:
        return f"{minutes}m {secs:.1f}s"
    hours = int(minutes // 60)
    mins = minutes % 60
    return f"{hours}h {mins}m {secs:.0f}s"


def format_stage_timings(timings: Mapping[str, float]) -> str:
    lines = []
    for name, seconds in timings.items():
        if seconds <= 0:
            continue
        lines.append(f"- {name}: {format_seconds(seconds)}")
    return "\n".join(lines)


@dataclass
class RunDiagnostics:
    """단일 UI/CLI 실행 단위의 로그 파일 작성기."""

    origin: str
    debug: bool = False
    log_path: Path = field(init=False)

    def __post_init__(self) -> None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        self.log_path = LOG_DIR / f"{stamp}-{self.origin}.log"
        self.note(f"run.start origin={self.origin} debug={self.debug}")

    def _write(self, line: str) -> None:
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{_now()}] {line}\n")

    def note(self, message: str) -> None:
        self._write(message)

    def stage(self, label: str, seconds: float) -> None:
        self._write(f"stage {label} = {format_seconds(seconds)}")

    def stage_map(self, prefix: str, timings: Mapping[str, float]) -> None:
        for name, seconds in timings.items():
            if seconds <= 0:
                continue
            self.stage(f"{prefix}.{name}" if prefix else name, seconds)

    def exception(self, label: str, message: str, traceback_text: Optional[str] = None) -> None:
        self._write(f"error {label}: {message}")
        if traceback_text:
            for line in traceback_text.rstrip().splitlines():
                self._write(f"trace {line}")
