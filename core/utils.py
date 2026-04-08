"""파일 탐지, 경로 생성, 시간 포맷 등 유틸리티."""

import os
from pathlib import Path
from typing import List

SUPPORTED_EXTENSIONS = {".mp3", ".m4a", ".wav", ".webm", ".ogg", ".flac", ".mp4", ".wma", ".aac"}


def find_audio_files(input_path: str) -> List[Path]:
    """입력 경로에서 오디오 파일 리스트를 반환한다.
    
    - 파일이면 해당 파일만 반환 (확장자 검증)
    - 디렉토리면 지원 확장자를 가진 파일들을 탐색하여 반환
    """
    path = Path(input_path)
    
    if path.is_file():
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            return [path]
        else:
            raise ValueError(
                f"지원하지 않는 파일 형식입니다: {path.suffix}\n"
                f"지원 형식: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )
    
    if path.is_dir():
        files = []
        for f in sorted(path.iterdir()):
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(f)
        if not files:
            raise FileNotFoundError(f"디렉토리에 오디오 파일이 없습니다: {path}")
        return files
    
    raise FileNotFoundError(f"경로를 찾을 수 없습니다: {input_path}")


def build_output_path(audio_path: Path, output_dir: str = None) -> Path:
    """전사 결과 저장 경로를 생성한다.
    
    output_dir이 지정되면 해당 디렉토리에, 아니면 원본 파일과 같은 디렉토리에 저장.
    """
    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / f"{audio_path.stem}.txt"
    return audio_path.with_suffix(".txt")


def format_duration(seconds: float) -> str:
    """초를 사람이 읽기 좋은 형태로 포맷한다."""
    if seconds < 60:
        return f"{seconds:.1f}초"
    minutes = int(seconds // 60)
    secs = seconds % 60
    if minutes < 60:
        return f"{minutes}분 {secs:.1f}초"
    hours = int(minutes // 60)
    mins = minutes % 60
    return f"{hours}시간 {mins}분 {secs:.0f}초"


def format_file_size(path: Path) -> str:
    """파일 크기를 사람이 읽기 좋은 형태로 포맷한다."""
    size = path.stat().st_size
    if size < 1024:
        return f"{size}B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f}KB"
    if size < 1024 * 1024 * 1024:
        return f"{size / (1024 * 1024):.1f}MB"
    return f"{size / (1024 * 1024 * 1024):.1f}GB"
