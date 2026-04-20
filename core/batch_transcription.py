"""배치 전사 orchestration.

UI/CLI에서 공통으로 쓸 수 있는 배치 전사 흐름을 app.py 밖으로 분리한다.
Gradio 의존성 없이 경고/진행률 콜백만 주입받는다.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
import traceback
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from core.diagnostics import RunDiagnostics, format_stage_timings
from core.engine import MODELS, align_words, resolve_model_path, save_result, slice_audio, transcribe
from core.keywords import extract_keywords
from core.srt import save_srt, segments_to_srt
from core.two_pass import two_pass_transcribe
from core.utils import SUPPORTED_EXTENSIONS, format_duration


@dataclass
class BatchTranscriptionOptions:
    mode_label: str
    two_pass_enabled: bool
    output_format: str
    start_time: str = ""
    end_time: str = ""
    use_glossary: bool = True
    use_korean_norm: bool = True
    debug_mode: bool = False


@dataclass
class BatchTranscriptionResult:
    text: str
    download_path: Optional[str]
    status: str
    keyword_rows: List[List]
    clusters: List[dict]
    warnings: List[str] = field(default_factory=list)
    log_path: Optional[str] = None


def _fmt_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def normalize_files(audio_files) -> List[str]:
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
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
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


def _format_body(result, output_format: str, aligned_result: dict | None = None) -> str:
    segments = result.processed_segments or result.segments
    if output_format == "자막 (.srt)":
        return segments_to_srt(segments)
    if output_format == "타임스탬프 포함 (.txt)":
        if aligned_result and aligned_result.get("segments"):
            lines = []
            for seg in aligned_result["segments"]:
                seg_start = _fmt_ts(float(seg.get("start", 0.0)))
                seg_end = _fmt_ts(float(seg.get("end", 0.0)))
                words = seg.get("words", [])
                if words:
                    word_parts = []
                    for word in words:
                        word_text = word.get("word", "")
                        if "start" in word:
                            word_parts.append(f"{word_text}({_fmt_ts(float(word['start']))})")
                        else:
                            word_parts.append(word_text)
                    lines.append(f"[{seg_start} → {seg_end}] {' '.join(word_parts)}")
                else:
                    text = (seg.get("text") or "").strip()
                    lines.append(f"[{seg_start} → {seg_end}] {text}")
            return "\n".join(lines)

        lines = []
        for seg in segments:
            start = _fmt_ts(float(seg.get("start", 0.0)))
            end = _fmt_ts(float(seg.get("end", 0.0)))
            text = (seg.get("text") or "").strip()
            lines.append(f"[{start} → {end}] {text}")
        return "\n".join(lines)
    return result.text


def parse_time(time_str: str) -> Optional[str]:
    if not time_str or not time_str.strip():
        return None
    normalized = time_str.strip()
    parts = normalized.split(":")
    if len(parts) == 2:
        return f"00:{normalized}"
    if len(parts) == 3:
        return normalized
    return None


def mode_label_to_key(mode_label: str) -> str:
    if "한국어" in mode_label:
        return "korean"
    if "정밀" in mode_label:
        return "precise"
    return "fast"


def transcribe_batch(
    audio_files,
    options: BatchTranscriptionOptions,
    detect_clusters: Callable[[str], List[dict]],
    *,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    warning_callback: Optional[Callable[[str], None]] = None,
) -> BatchTranscriptionResult:
    warnings: List[str] = []
    diagnostics = RunDiagnostics(origin="ui", debug=options.debug_mode)

    def warn(message: str) -> None:
        warnings.append(message)
        diagnostics.note(f"warn {message}")
        if warning_callback:
            warning_callback(message)

    def update_progress(value: float, desc: str) -> None:
        if progress_callback:
            progress_callback(value, desc)

    files = normalize_files(audio_files)
    if not files:
        warn("오디오 파일을 먼저 업로드해주세요.")
        return BatchTranscriptionResult("", None, "", [], [], warnings=warnings, log_path=str(diagnostics.log_path))

    valid_files = [fp for fp in files if Path(fp).suffix.lower() in SUPPORTED_EXTENSIONS]
    skipped = [Path(fp).name for fp in files if Path(fp).suffix.lower() not in SUPPORTED_EXTENSIONS]
    if skipped:
        warn(f"지원하지 않는 형식 {len(skipped)}개 무시")
    if not valid_files:
        return BatchTranscriptionResult(
            "",
            None,
            "❌ 지원하는 오디오 파일이 없습니다.",
            [],
            [],
            warnings=warnings,
            log_path=str(diagnostics.log_path),
        )

    mode_key = mode_label_to_key(options.mode_label)
    try:
        model = resolve_model_path(mode_key)
    except FileNotFoundError as e:
        diagnostics.exception("resolve_model", str(e))
        return BatchTranscriptionResult("", None, f"❌ {e}", [], [], warnings=warnings, log_path=str(diagnostics.log_path))

    t_start = parse_time(options.start_time)
    t_end = parse_time(options.end_time)
    needs_timestamps = options.output_format in ("타임스탬프 포함 (.txt)", "자막 (.srt)")
    build_processed_segments = options.output_format != "텍스트 (.txt)"

    n = len(valid_files)
    output_dir = Path(tempfile.mkdtemp(prefix="noma-scribe-"))
    result_paths: List[Path] = []
    display_blocks: List[str] = []
    keyword_rows: List[List] = []
    success_count = 0
    total_elapsed = 0.0

    update_progress(0.0, "준비 중...")
    diagnostics.note(
        f"run.config files={n} mode={mode_key} two_pass={options.two_pass_enabled} "
        f"format={options.output_format} glossary={options.use_glossary} "
        f"korean_norm={options.use_korean_norm}"
    )

    for idx, fp in enumerate(valid_files, 1):
        audio_path = Path(fp)
        file_dur = _probe_audio_duration(str(audio_path))
        length_str = _format_audio_length(file_dur)
        base_pct = (idx - 1) / n
        mode_ui_label = MODELS[mode_key]["label"]
        diagnostics.note(f"file.start name={audio_path.name} size={audio_path.stat().st_size}")

        update_progress(base_pct, f"{idx}/{n} [{mode_ui_label}] 전사 중... ({audio_path.name}, {length_str})")

        actual_path = str(audio_path)
        sliced = False
        try:
            if t_start or t_end:
                slice_start = time.time()
                actual_path = slice_audio(str(audio_path), t_start, t_end)
                sliced = actual_path != str(audio_path)
                diagnostics.stage(f"{audio_path.name}.slice_audio", time.time() - slice_start)
        except Exception as e:
            warn(f"구간 자르기 실패: {e}")

        progress_state = {"pct": 0.0, "stage": "전사 중..."}

        def _status_cb(message: str) -> None:
            progress_state["stage"] = message
            overall = (idx - 1 + min(0.96, progress_state["pct"] / 100 * 0.96)) / n
            update_progress(
                overall,
                f"{idx}/{n} [{mode_ui_label}] {message} ({audio_path.name}) — {progress_state['pct']:.0f}%",
            )

        def _progress_cb(pct: float) -> None:
            progress_state["pct"] = max(0.0, min(100.0, pct))
            overall = (idx - 1 + min(0.96, progress_state["pct"] / 100 * 0.96)) / n
            update_progress(
                overall,
                f"{idx}/{n} [{mode_ui_label}] {progress_state['stage']} ({audio_path.name}) — {progress_state['pct']:.0f}%",
            )

        try:
            if options.two_pass_enabled:
                result = two_pass_transcribe(
                    audio_path=actual_path,
                    language="ko",
                    model=model,
                    progress_callback=_progress_cb,
                    status_callback=_status_cb,
                    warning_callback=warn,
                    use_glossary=options.use_glossary,
                    use_korean_norm=options.use_korean_norm,
                    build_processed_segments=build_processed_segments,
                    debug=options.debug_mode,
                )
            else:
                result = transcribe(
                    audio_path=actual_path,
                    language="ko",
                    model=model,
                    progress_callback=_progress_cb,
                    status_callback=_status_cb,
                    warning_callback=warn,
                    use_glossary=options.use_glossary,
                    use_korean_norm=options.use_korean_norm,
                    build_processed_segments=build_processed_segments,
                    debug=options.debug_mode,
                )

            total_elapsed += result.duration_seconds
            success_count += 1
            diagnostics.stage_map(audio_path.name, result.stage_timings)

            aligned_result = None
            if needs_timestamps and options.output_format != "자막 (.srt)" and not (
                options.use_glossary or options.use_korean_norm
            ):
                try:
                    aligned_result = align_words(result)
                except Exception:
                    aligned_result = None

            update_progress((idx - 1 + 0.97) / n, f"{idx}/{n} [{mode_ui_label}] 결과 저장 중... ({audio_path.name})")

            save_start = time.time()
            if options.output_format == "자막 (.srt)":
                out_path = output_dir / f"{audio_path.stem}.srt"
                save_srt(result.processed_segments or result.segments, out_path)
            else:
                out_path = output_dir / f"{audio_path.stem}.txt"
                save_result(
                    result,
                    out_path,
                    include_timestamps=options.output_format == "타임스탬프 포함 (.txt)",
                )
            diagnostics.stage(f"{audio_path.name}.save_result", time.time() - save_start)
            result_paths.append(out_path)

            body = _format_body(result, options.output_format, aligned_result=aligned_result)
            if n == 1:
                display_blocks.append(body)
            else:
                display_blocks.append(f"━━━ [{idx}/{n}] {audio_path.name} ━━━\n{body}")

            if idx == n:
                update_progress((idx - 1 + 0.99) / n, f"{idx}/{n} [{mode_ui_label}] 키워드 정리 중... ({audio_path.name})")
                keyword_start = time.time()
                full_text = " ".join(
                    block.split("\n", 1)[-1] if "━━━" in block else block
                    for block in display_blocks
                )
                kw = extract_keywords(full_text, top_n=15, min_count=2)
                keyword_rows = [[word, count] for word, count in kw]
                diagnostics.stage(f"{audio_path.name}.extract_keywords", time.time() - keyword_start)
        except Exception as e:
            tb = traceback.format_exc()
            diagnostics.exception(audio_path.name, str(e), tb)
            if n == 1:
                update_progress(1.0, "실패")
                detail = f"❌ {audio_path.name}: {e}"
                if options.debug_mode:
                    detail = f"{detail}\n\n{tb.rstrip()}"
                return BatchTranscriptionResult(
                    "",
                    None,
                    detail,
                    [],
                    [],
                    warnings=warnings,
                    log_path=str(diagnostics.log_path),
                )
            display_blocks.append(f"━━━ [{idx}/{n}] {audio_path.name} ━━━\n❌ 오류: {e}")
        finally:
            if sliced and Path(actual_path).exists():
                try:
                    Path(actual_path).unlink()
                except Exception:
                    pass

    update_progress(1.0, "완료")

    if not result_paths:
        return BatchTranscriptionResult(
            "\n\n".join(display_blocks),
            None,
            f"❌ 모두 실패 (0/{n})",
            keyword_rows,
            [],
            warnings=warnings,
            log_path=str(diagnostics.log_path),
        )

    if len(result_paths) == 1:
        download_path = str(result_paths[0])
        download_label = result_paths[0].name
    else:
        ext = result_paths[0].suffix
        zip_name = f"transcripts{ext}.zip" if ext != ".txt" else "transcripts.zip"
        zip_path = output_dir / zip_name
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in result_paths:
                zf.write(file_path, arcname=file_path.name)
        download_path = str(zip_path)
        download_label = f"{zip_name} ({len(result_paths)}개)"

    elapsed_str = format_duration(total_elapsed)
    status_mode = MODELS[mode_key]["label"]
    two_pass_tag = " +2pass" if options.two_pass_enabled else ""
    extras = []
    if options.use_glossary:
        extras.append("용어집")
    if options.use_korean_norm:
        extras.append("한글정리")
    extras_tag = f" [{'/'.join(extras)}]" if extras else ""

    stage_summary = ""
    if n == 1 and success_count == 1:
        stage_summary = format_stage_timings(result.stage_timings)

    if n == 1:
        status = f"✅ 완료! [{status_mode}{two_pass_tag}]{extras_tag} ({elapsed_str}) — {download_label}"
        if stage_summary:
            status += f"\n\n단계별 소요:\n{stage_summary}"
    else:
        status = f"✅ 완료! [{status_mode}{two_pass_tag}]{extras_tag} {success_count}/{n}개 ({elapsed_str}) — {download_label}"
    status += f"\n\n로그: {diagnostics.log_path}"

    final_text = "\n\n".join(display_blocks)
    flat_text = " ".join(
        block.split("\n", 1)[-1] if "━━━" in block else block
        for block in display_blocks
    )
    clusters = detect_clusters(flat_text)

    return BatchTranscriptionResult(
        text=final_text,
        download_path=download_path,
        status=status,
        keyword_rows=keyword_rows,
        clusters=clusters,
        warnings=warnings,
        log_path=str(diagnostics.log_path),
    )
