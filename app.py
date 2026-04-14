#!/usr/bin/env python3
"""noma-scribe v0.4: Apple Silicon 로컬 음성 전사 앱.

주요 변경 (v0.4):
- 프리셋 제거 → 빠른/정밀 모드 선택
- 정밀 모드: 2-pass 자동 용어 추출
- 출력 포맷: txt / 타임스탬프 txt / SRT 자막
- 구간 전사: 시작/끝 시간 지정
- 키워드 빈도 표시
"""

import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import List, Optional

import gradio as gr

from core.engine import (
    MODELS,
    DEFAULT_MODEL,
    align_words,
    is_model_available,
    resolve_model_path,
    save_result,
    slice_audio,
    transcribe,
)
from core.keywords import extract_keywords, format_keywords
from core.srt import save_srt, segments_to_srt
from core.two_pass import two_pass_transcribe
from core.utils import format_duration, SUPPORTED_EXTENSIONS


# ──────────────────────────────────────────────
# 헬퍼
# ──────────────────────────────────────────────

def _fmt_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _normalize_files(audio_files) -> List[str]:
    if audio_files is None:
        return []
    if isinstance(audio_files, (str, Path)):
        return [str(audio_files)]
    if isinstance(audio_files, (list, tuple)):
        out = []
        for f in audio_files:
            if isinstance(f, dict):
                out.append(f.get("path") or f.get("name", ""))
            elif hasattr(f, "name"):
                out.append(f.name)
            else:
                out.append(str(f))
        return [p for p in out if p]
    if hasattr(audio_files, "name"):
        return [audio_files.name]
    return [str(audio_files)]


def _probe_audio_duration(path: str) -> Optional[float]:
    if not shutil.which("ffprobe"):
        return None
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            stderr=subprocess.DEVNULL, text=True, timeout=10,
        )
        return float(out.strip())
    except Exception:
        return None


def _format_audio_length(seconds: Optional[float]) -> str:
    if seconds is None:
        return "?"
    if seconds < 60:
        return f"{seconds:.0f}초"
    m = int(seconds // 60)
    s = int(seconds % 60)
    if m < 60:
        return f"{m}분 {s}초"
    h = int(m // 60)
    return f"{h}시간 {m % 60}분"


def _format_body(result, output_format: str, aligned_result: dict = None) -> str:
    """전사 결과를 화면 표시용 텍스트로 변환."""
    if output_format == "자막 (.srt)":
        return segments_to_srt(result.segments)
    elif output_format == "타임스탬프 포함 (.txt)":
        if aligned_result and aligned_result.get("segments"):
            lines = []
            for seg in aligned_result["segments"]:
                seg_start = _fmt_ts(float(seg.get("start", 0.0)))
                seg_end = _fmt_ts(float(seg.get("end", 0.0)))
                words = seg.get("words", [])
                if words:
                    word_parts = []
                    for w in words:
                        word_text = w.get("word", "")
                        if "start" in w:
                            word_parts.append(f"{word_text}({_fmt_ts(float(w['start']))})")
                        else:
                            word_parts.append(word_text)
                    lines.append(f"[{seg_start} → {seg_end}] {' '.join(word_parts)}")
                else:
                    text = (seg.get("text") or "").strip()
                    lines.append(f"[{seg_start} → {seg_end}] {text}")
            return "\n".join(lines)
        # fallback: segment-level
        lines = []
        for seg in result.segments:
            start = _fmt_ts(float(seg.get("start", 0.0)))
            end = _fmt_ts(float(seg.get("end", 0.0)))
            text = (seg.get("text") or "").strip()
            lines.append(f"[{start} → {end}] {text}")
        return "\n".join(lines)
    else:
        return result.text


def _parse_time(time_str: str) -> Optional[str]:
    """사용자 입력 MM:SS 또는 HH:MM:SS를 검증. 빈 문자열이면 None."""
    if not time_str or not time_str.strip():
        return None
    time_str = time_str.strip()
    parts = time_str.split(":")
    if len(parts) == 2:
        return f"00:{time_str}"
    elif len(parts) == 3:
        return time_str
    return None


# ──────────────────────────────────────────────
# 핵심 로직 — 배치 전사
# ──────────────────────────────────────────────

def run_transcription(
    audio_files,
    mode: str,
    two_pass_enabled: bool,
    output_format: str,
    start_time: str,
    end_time: str,
    progress=gr.Progress(),
):
    files = _normalize_files(audio_files)
    if not files:
        gr.Warning("오디오 파일을 먼저 업로드해주세요.")
        return "", None, "", ""

    valid_files = [fp for fp in files if Path(fp).suffix.lower() in SUPPORTED_EXTENSIONS]
    skipped = [Path(fp).name for fp in files if Path(fp).suffix.lower() not in SUPPORTED_EXTENSIONS]
    if skipped:
        gr.Warning(f"지원하지 않는 형식 {len(skipped)}개 무시")
    if not valid_files:
        return "", None, "❌ 지원하는 오디오 파일이 없습니다.", ""

    # 모드 → 모델 키
    if "한국어" in mode:
        mode_key = "korean"
    elif "정밀" in mode:
        mode_key = "precise"
    else:
        mode_key = "fast"

    try:
        model = resolve_model_path(mode_key)
    except FileNotFoundError as e:
        return "", None, f"❌ {e}", ""

    use_two_pass = two_pass_enabled

    # 언어
    lang_code = "ko"

    # 시간 범위
    t_start = _parse_time(start_time)
    t_end = _parse_time(end_time)

    # 타임스탬프/SRT 여부
    needs_timestamps = output_format in ("타임스탬프 포함 (.txt)", "자막 (.srt)")

    n = len(valid_files)
    output_dir = Path(tempfile.mkdtemp(prefix="noma-scribe-"))
    result_paths: List[Path] = []
    display_blocks: List[str] = []
    all_keywords_text = ""
    success_count = 0
    total_elapsed = 0.0

    progress(0.0, desc="준비 중...")

    for idx, fp in enumerate(valid_files, 1):
        audio_path = Path(fp)
        file_dur = _probe_audio_duration(str(audio_path))
        length_str = _format_audio_length(file_dur)
        base_pct = (idx - 1) / n

        mode_label = MODELS[mode_key]["label"]
        progress(base_pct, desc=f"{idx}/{n} [{mode_label}] 전사 중... ({audio_path.name}, {length_str})")

        # 구간 슬라이싱
        actual_path = str(audio_path)
        sliced = False
        try:
            if t_start or t_end:
                actual_path = slice_audio(str(audio_path), t_start, t_end)
                sliced = actual_path != str(audio_path)
        except Exception as e:
            gr.Warning(f"구간 자르기 실패: {e}")

        def _make_cb(file_idx: int):
            def _cb(pct: float):
                try:
                    overall = (file_idx - 1 + max(0, min(1, pct / 100))) / n
                    progress(overall, desc=f"{file_idx}/{n} 전사 중... ({audio_path.name}) — {pct:.0f}%")
                except Exception:
                    pass
            return _cb

        try:
            if use_two_pass:
                # 2-pass 전사 (어떤 모드에서든)
                def _status_cb(msg: str):
                    try:
                        progress(base_pct, desc=f"{idx}/{n} [{mode_label}] {msg} ({audio_path.name})")
                    except Exception:
                        pass

                result = two_pass_transcribe(
                    audio_path=actual_path,
                    language=lang_code,
                    model=model,
                    progress_callback=_make_cb(idx),
                    status_callback=_status_cb,
                )
            else:
                # 1-pass
                result = transcribe(
                    audio_path=actual_path,
                    language=lang_code,
                    model=model,
                    progress_callback=_make_cb(idx),
                )

            total_elapsed += result.duration_seconds
            success_count += 1

            # word alignment (타임스탬프 포맷 선택 시)
            aligned_result = None
            if needs_timestamps and output_format != "자막 (.srt)":
                try:
                    aligned_result = align_words(result)
                except Exception:
                    pass  # fallback to segment-level

            progress(base_pct + 0.9 / n, desc=f"{idx}/{n} 후처리 중...")

            # 결과 저장
            if output_format == "자막 (.srt)":
                out_filename = audio_path.stem + ".srt"
                out_path = output_dir / out_filename
                save_srt(result.segments, out_path)
            else:
                out_filename = audio_path.stem + ".txt"
                out_path = output_dir / out_filename
                include_ts = output_format == "타임스탬프 포함 (.txt)"
                save_result(result, out_path, include_timestamps=include_ts)
            result_paths.append(out_path)

            # 화면 표시
            body = _format_body(result, output_format, aligned_result=aligned_result)
            if n == 1:
                display_blocks.append(body)
            else:
                header = f"━━━ [{idx}/{n}] {audio_path.name} ━━━"
                display_blocks.append(f"{header}\n{body}")

            # 키워드 빈도 (마지막 파일 또는 전체 텍스트)
            if idx == n:
                full_text = " ".join(
                    block.split("\n", 1)[-1] if "━━━" in block else block
                    for block in display_blocks
                )
                kw = extract_keywords(full_text, top_n=15, min_count=2)
                if kw:
                    all_keywords_text = format_keywords(kw)

        except Exception as e:
            if n == 1:
                progress(1.0, desc="실패")
                return "", None, f"❌ {audio_path.name}: {e}", ""
            header = f"━━━ [{idx}/{n}] {audio_path.name} ━━━"
            display_blocks.append(f"{header}\n❌ 오류: {e}")
        finally:
            # 임시 슬라이스 파일 정리
            if sliced and Path(actual_path).exists():
                try:
                    Path(actual_path).unlink()
                except Exception:
                    pass

    progress(1.0, desc="완료")

    if not result_paths:
        return "\n\n".join(display_blocks), None, f"❌ 모두 실패 (0/{n})", all_keywords_text

    # 다운로드 파일
    if len(result_paths) == 1:
        download_path = str(result_paths[0])
        download_label = result_paths[0].name
    else:
        ext = result_paths[0].suffix
        zip_name = f"transcripts{ext}.zip" if ext != ".txt" else "transcripts.zip"
        zip_path = output_dir / zip_name
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in result_paths:
                zf.write(f, arcname=f.name)
        download_path = str(zip_path)
        download_label = f"{zip_name} ({len(result_paths)}개)"

    elapsed_str = format_duration(total_elapsed)
    status_mode = MODELS[mode_key]["label"]
    two_pass_tag = " +2pass" if use_two_pass else ""
    if n == 1:
        status = f"✅ 완료! [{status_mode}{two_pass_tag}] ({elapsed_str}) — {download_label}"
    else:
        status = f"✅ 완료! [{status_mode}{two_pass_tag}] {success_count}/{n}개 ({elapsed_str}) — {download_label}"

    return "\n\n".join(display_blocks), download_path, status, all_keywords_text


# ──────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────

def create_app():
    # 모드 드롭다운 선택지 구성 (모델 존재 여부 반영)
    mode_choices = []
    default_mode = None
    for key, info in MODELS.items():
        available = is_model_available(key)
        if available:
            label = f"{info['label']} — {info['desc']}"
            mode_choices.append(label)
            if default_mode is None:
                default_mode = label
        else:
            label = f"{info['label']} (모델 미설치)"
            mode_choices.append(label)

    if default_mode is None:
        default_mode = mode_choices[0] if mode_choices else ""

    with gr.Blocks(title="noma-scribe") as app:

        gr.Markdown("# 🎙️ noma-scribe", elem_classes=["main-title"])
        gr.Markdown(
            "오디오 파일을 올리고 버튼만 누르면 텍스트로 전사합니다. "
            "완전 로컬, 무료. (엔진: whispermlx)",
            elem_classes=["sub-title"],
        )

        with gr.Row():
            # ── 왼쪽: 입력 ──
            with gr.Column(scale=1):
                audio_input = gr.File(
                    label="오디오 파일 (여러 개 동시 업로드 가능)",
                    file_count="multiple",
                    file_types=[".mp3", ".m4a", ".wav", ".webm", ".ogg",
                                ".flac", ".mp4", ".wma", ".aac"],
                    type="filepath",
                )

                with gr.Accordion("옵션", open=False):
                    mode_select = gr.Dropdown(
                        choices=mode_choices,
                        value=default_mode,
                        label="전사 모드",
                    )
                    two_pass_check = gr.Checkbox(
                        label="2-pass 용어 추출 (1차 전사 → 용어 자동 추출 → 2차 전사)",
                        value=False,
                    )
                    format_select = gr.Dropdown(
                        choices=["텍스트 (.txt)", "타임스탬프 포함 (.txt)", "자막 (.srt)"],
                        value="텍스트 (.txt)",
                        label="출력 포맷",
                    )
                    with gr.Row():
                        start_time = gr.Textbox(
                            label="시작 시간 (선택)",
                            placeholder="MM:SS",
                            max_lines=1,
                        )
                        end_time = gr.Textbox(
                            label="끝 시간 (선택)",
                            placeholder="MM:SS",
                            max_lines=1,
                        )

                transcribe_btn = gr.Button(
                    "🎙️ 전사 시작",
                    variant="primary",
                    size="lg",
                )

            # ── 오른쪽: 출력 ──
            with gr.Column(scale=1):
                status_text = gr.Textbox(
                    label="상태",
                    interactive=False,
                    elem_classes=["status-box"],
                )
                output_text = gr.Textbox(
                    label="전사 결과",
                    lines=18,
                    max_lines=40,
                    interactive=False,
                )
                download_file = gr.File(
                    label="다운로드",
                    visible=True,
                )
                keywords_text = gr.Textbox(
                    label="📊 자주 등장한 단어 (Top 15)",
                    interactive=False,
                    lines=2,
                )

        transcribe_btn.click(
            fn=run_transcription,
            inputs=[audio_input, mode_select, two_pass_check, format_select, start_time, end_time],
            outputs=[output_text, download_file, status_text, keywords_text],
        )

    return app


# ──────────────────────────────────────────────
# 실행
# ──────────────────────────────────────────────

if __name__ == "__main__":
    app = create_app()
    theme = gr.themes.Soft(
        primary_hue="blue",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    )
    css = """
    .main-title { text-align: center; margin-bottom: 0.5em; }
    .sub-title { text-align: center; color: #64748b; margin-top: 0; font-size: 0.95em; }
    .status-box { font-size: 1.1em; padding: 12px; border-radius: 8px; }
    footer { display: none !important; }
    """
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        theme=theme,
        css=css,
    )
