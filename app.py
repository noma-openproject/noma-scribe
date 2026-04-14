#!/usr/bin/env python3
"""noma-scribe: Apple Silicon 로컬 음성 전사 앱.

더블클릭으로 실행하면 브라우저에서 깔끔한 UI가 열립니다.
오디오 파일을 1개 또는 여러 개 드래그앤드롭하고 '전사 시작' 버튼만 누르면 끝.

엔진: whispermlx (WhisperX 의 Apple Silicon 포크) + mlx-whisper 백엔드
"""

import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import List, Optional

import gradio as gr

from core.engine import transcribe, save_result, align_words, DEFAULT_MODEL
from core.presets import load_prompt, list_presets
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


def _format_body(result, include_timestamps: bool, aligned_result: dict = None) -> str:
    """전사 결과를 화면 표시용 텍스트로 변환.

    aligned_result 가 있으면 단어 단위 타임스탬프를 사용한다.
    """
    if include_timestamps and aligned_result and aligned_result.get("segments"):
        # word-level aligned 결과 사용
        lines = []
        for seg in aligned_result["segments"]:
            seg_start = _fmt_ts(float(seg.get("start", 0.0)))
            seg_end = _fmt_ts(float(seg.get("end", 0.0)))
            words = seg.get("words", [])
            if words:
                word_parts = []
                for w in words:
                    word_text = w.get("word", "")
                    if "start" in w and "end" in w:
                        ws = _fmt_ts(float(w["start"]))
                        we = _fmt_ts(float(w["end"]))
                        word_parts.append(f"{word_text}({ws})")
                    else:
                        word_parts.append(word_text)
                lines.append(f"[{seg_start} → {seg_end}] {' '.join(word_parts)}")
            else:
                text = (seg.get("text") or "").strip()
                lines.append(f"[{seg_start} → {seg_end}] {text}")
        return "\n".join(lines)
    elif include_timestamps and result.segments:
        # segment-level fallback
        lines = []
        for seg in result.segments:
            start = _fmt_ts(float(seg.get("start", 0.0)))
            end = _fmt_ts(float(seg.get("end", 0.0)))
            text = (seg.get("text") or "").strip()
            lines.append(f"[{start} → {end}] {text}")
        return "\n".join(lines)
    return result.text


def _normalize_files(audio_files) -> List[str]:
    """gr.File 의 입력을 항상 파일 경로 리스트로 정규화한다."""
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
    """ffprobe 로 오디오 길이(초)를 조회. 실패 시 None."""
    if not shutil.which("ffprobe"):
        return None
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        )
        return float(out.strip())
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError, FileNotFoundError):
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
    m = m % 60
    return f"{h}시간 {m}분"


# ──────────────────────────────────────────────
# 핵심 로직 — 배치 전사
# ──────────────────────────────────────────────

def run_transcription(
    audio_files,
    preset: str,
    custom_prompt: str,
    language: str,
    include_timestamps: bool,
    progress=gr.Progress(),
):
    """오디오 파일 1개 또는 N개를 순차 전사한다.

    - 1개  → 단일 .txt 파일을 다운로드 슬롯에 제공
    - N개  → 각 파일별 .txt 를 만들어 transcripts.zip 으로 묶어 제공
    """

    files = _normalize_files(audio_files)
    if not files:
        gr.Warning("오디오 파일을 먼저 업로드해주세요.")
        return "", None, ""

    # 지원 확장자 필터링
    valid_files = []
    skipped = []
    for fp in files:
        if Path(fp).suffix.lower() in SUPPORTED_EXTENSIONS:
            valid_files.append(fp)
        else:
            skipped.append(Path(fp).name)

    if skipped:
        gr.Warning(
            f"지원하지 않는 형식 {len(skipped)}개 무시: "
            f"{', '.join(skipped[:3])}{'...' if len(skipped) > 3 else ''}"
        )

    if not valid_files:
        return "", None, "❌ 지원하는 오디오 파일이 없습니다."

    # ── 프롬프트 결정: 커스텀 > 프리셋 ──
    prompt: Optional[str] = None
    if custom_prompt and custom_prompt.strip():
        prompt = custom_prompt.strip()
    elif preset and preset != "없음":
        try:
            prompt = load_prompt(preset=preset)
        except Exception:
            pass

    # ── 언어 코드 매핑 ──
    lang_map = {
        "한국어": "ko",
        "English": "en",
        "日本語": "ja",
        "中文": "zh",
        "자동 감지": None,
    }
    lang_code = lang_map.get(language, "ko")

    # ── 전체 오디오 길이 사전 조회 (ffprobe) ──
    # 상태 메시지에 "X분 오디오" 로 노출하여 대기 예측을 돕는다.
    per_file_durations = [_probe_audio_duration(fp) for fp in valid_files]
    total_known_duration = sum(d for d in per_file_durations if d is not None)

    # ── 배치 처리 ──
    n = len(valid_files)
    output_dir = Path(tempfile.mkdtemp(prefix="noma-scribe-"))
    txt_paths: List[Path] = []
    display_blocks: List[str] = []
    success_count = 0
    total_elapsed = 0.0

    progress(
        0.0,
        desc=(
            f"준비 중... ({n}개 파일"
            + (f", 총 {_format_audio_length(total_known_duration)}" if total_known_duration else "")
            + ")"
        ),
    )

    for idx, fp in enumerate(valid_files, 1):
        audio_path = Path(fp)
        file_dur = per_file_durations[idx - 1]
        length_str = _format_audio_length(file_dur)
        base_pct = (idx - 1) / n

        # 파일 시작 시 progress 업데이트 (길이 힌트 포함)
        progress(
            base_pct,
            desc=f"{idx}/{n} 전사 중... ({audio_path.name}, {length_str})",
        )

        # whispermlx 가 VAD 청크 단위로 호출해주는 progress_callback.
        # 청크 진행률(0~100)을 전체 진행률로 환산해서 Gradio Progress 에 반영한다.
        def _make_cb(file_idx: int):
            def _cb(pct: float):
                try:
                    chunk_fraction = max(0.0, min(1.0, pct / 100.0))
                    overall = (file_idx - 1 + chunk_fraction) / n
                    progress(
                        overall,
                        desc=(
                            f"{file_idx}/{n} 전사 중... "
                            f"({audio_path.name}, {length_str}) — {pct:.0f}%"
                        ),
                    )
                except Exception:
                    # progress 업데이트 실패는 전사를 멈추지 않는다
                    pass
            return _cb

        try:
            result = transcribe(
                audio_path=str(audio_path),
                language=lang_code,
                model=DEFAULT_MODEL,
                initial_prompt=prompt,
                word_timestamps=include_timestamps,
                progress_callback=_make_cb(idx),
            )
            total_elapsed += result.duration_seconds
            success_count += 1

            # word-level alignment (타임스탬프 옵션 활성화 시)
            aligned_result = None
            if include_timestamps:
                try:
                    aligned_result = align_words(result)
                except Exception as align_err:
                    # alignment 실패해도 세그먼트 단위 타임스탬프로 fallback
                    import logging
                    logging.getLogger(__name__).warning(
                        f"word alignment 실패 (세그먼트 단위로 대체): {align_err}"
                    )

            # 결과 저장
            txt_filename = audio_path.stem + ".txt"
            txt_path = output_dir / txt_filename
            save_result(result, txt_path, include_timestamps=include_timestamps)
            txt_paths.append(txt_path)

            # 화면 표시 텍스트 누적
            body = _format_body(result, include_timestamps, aligned_result=aligned_result)
            if n == 1:
                display_blocks.append(body)
            else:
                header = f"━━━ [{idx}/{n}] {audio_path.name} ━━━"
                display_blocks.append(f"{header}\n{body}")

        except Exception as e:
            err = f"❌ 오류: {e}"
            if n == 1:
                progress(1.0, desc="실패")
                return "", None, f"❌ {audio_path.name}: {e}"
            header = f"━━━ [{idx}/{n}] {audio_path.name} ━━━"
            display_blocks.append(f"{header}\n{err}")

    progress(1.0, desc="완료")

    if not txt_paths:
        return "\n\n".join(display_blocks), None, f"❌ 모두 실패 (0/{n})"

    # ── 다운로드 파일 결정 ──
    if len(txt_paths) == 1:
        download_path = str(txt_paths[0])
        download_label = txt_paths[0].name
    else:
        zip_path = output_dir / "transcripts.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in txt_paths:
                zf.write(f, arcname=f.name)
        download_path = str(zip_path)
        download_label = f"transcripts.zip ({len(txt_paths)}개)"

    elapsed_str = format_duration(total_elapsed)
    if n == 1:
        status = f"✅ 완료! ({elapsed_str}) — {download_label}"
    else:
        status = f"✅ 완료! {success_count}/{n}개 ({elapsed_str}) — {download_label}"

    return "\n\n".join(display_blocks), download_path, status


# ──────────────────────────────────────────────
# UI 구성
# ──────────────────────────────────────────────

def create_app():
    presets = list_presets()
    preset_choices = ["없음"] + list(presets.keys())

    with gr.Blocks(title="noma-scribe") as app:

        # 헤더
        gr.Markdown("# 🎙️ noma-scribe", elem_classes=["main-title"])
        gr.Markdown(
            "오디오 파일을 1개 또는 여러 개 올리고 버튼만 누르면 텍스트로 전사합니다. "
            "완전 로컬, 무료. (엔진: whispermlx + mlx-whisper)",
            elem_classes=["sub-title"],
        )

        with gr.Row():
            # ── 왼쪽: 입력 ──
            with gr.Column(scale=1):
                audio_input = gr.File(
                    label="오디오 파일 (여러 개 동시 업로드 가능)",
                    file_count="multiple",
                    file_types=[
                        ".mp3", ".m4a", ".wav", ".webm", ".ogg",
                        ".flac", ".mp4", ".wma", ".aac",
                    ],
                    type="filepath",
                )

                with gr.Accordion("옵션", open=False):
                    preset_dropdown = gr.Dropdown(
                        choices=preset_choices,
                        value="없음",
                        label="프리셋",
                        info="도메인 전문용어 힌트",
                    )
                    custom_prompt = gr.Textbox(
                        label="커스텀 프롬프트 (선택)",
                        placeholder="전문용어를 공백으로 구분하여 입력... (입력 시 프리셋보다 우선)",
                        lines=2,
                    )
                    language = gr.Dropdown(
                        choices=["한국어", "English", "日本語", "中文", "자동 감지"],
                        value="한국어",
                        label="언어",
                    )
                    timestamps = gr.Checkbox(
                        label="타임스탬프 포함 (단어 단위 정렬)",
                        value=False,
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
                    label="전사 결과 (여러 파일은 구분선으로 분리)",
                    lines=18,
                    max_lines=40,
                    interactive=False,
                )
                download_file = gr.File(
                    label="다운로드 (1개=txt, 여러 개=zip)",
                    visible=True,
                )

        # 이벤트 연결
        transcribe_btn.click(
            fn=run_transcription,
            inputs=[audio_input, preset_dropdown, custom_prompt, language, timestamps],
            outputs=[output_text, download_file, status_text],
        )

        # 프리셋 설명
        gr.Markdown(
            "**프리셋 목록**: "
            + " · ".join([f"`{k}` ({v})" for k, v in presets.items()])
            + " · `presets/` 폴더에 txt 파일 추가로 확장 가능"
        )

    return app


# ──────────────────────────────────────────────
# 실행
# ──────────────────────────────────────────────

if __name__ == "__main__":
    app = create_app()

    # Gradio 6.0+: theme/css 는 launch() 로 전달
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
