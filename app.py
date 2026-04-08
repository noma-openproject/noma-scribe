#!/usr/bin/env python3
"""noma-scribe: Apple Silicon 로컬 음성 전사 앱.

더블클릭으로 실행하면 브라우저에서 깔끔한 UI가 열립니다.
오디오 파일을 드래그앤드롭하고 '전사 시작' 버튼만 누르면 끝.
"""

import os
import tempfile
import time
from pathlib import Path
from typing import Optional

import gradio as gr

from core.engine import transcribe, save_result, DEFAULT_MODEL
from core.presets import load_prompt, list_presets
from core.utils import format_duration, format_file_size, SUPPORTED_EXTENSIONS


# ──────────────────────────────────────────────
# 핵심 로직
# ──────────────────────────────────────────────

def run_transcription(
    audio_file,
    preset: str,
    custom_prompt: str,
    language: str,
    include_timestamps: bool,
):
    """전사를 실행하고 결과를 반환한다."""
    
    if audio_file is None:
        gr.Warning("오디오 파일을 먼저 업로드해주세요.")
        return "", None, ""
    
    # 프롬프트 결정: 커스텀 > 프리셋
    prompt = None
    if custom_prompt and custom_prompt.strip():
        prompt = custom_prompt.strip()
    elif preset and preset != "없음":
        try:
            prompt = load_prompt(preset=preset)
        except Exception:
            pass
    
    # 언어 코드 매핑
    lang_map = {
        "한국어": "ko",
        "English": "en",
        "日本語": "ja",
        "中文": "zh",
        "자동 감지": None,
    }
    lang_code = lang_map.get(language, "ko")
    
    # 파일 정보
    audio_path = Path(audio_file)
    file_size = format_file_size(audio_path)
    file_name = audio_path.name
    
    # 상태 메시지
    status = f"🎙️ 전사 중... ({file_name}, {file_size})"
    
    try:
        # 전사 옵션
        options = {
            "audio_path": str(audio_path),
            "model": DEFAULT_MODEL,
            "word_timestamps": include_timestamps,
        }
        if lang_code:
            options["language"] = lang_code
        if prompt:
            options["initial_prompt"] = prompt
        
        result = transcribe(**options)
        
        elapsed = format_duration(result.duration_seconds)
        
        # txt 파일 생성
        output_dir = tempfile.mkdtemp()
        txt_filename = audio_path.stem + ".txt"
        txt_path = Path(output_dir) / txt_filename
        save_result(result, txt_path, include_timestamps=include_timestamps)
        
        # 결과 텍스트 구성
        if include_timestamps and result.segments:
            display_lines = []
            for seg in result.segments:
                start = _fmt_ts(seg["start"])
                end = _fmt_ts(seg["end"])
                text = seg["text"].strip()
                display_lines.append(f"[{start} → {end}] {text}")
            display_text = "\n".join(display_lines)
        else:
            display_text = result.text
        
        status = f"✅ 완료! ({elapsed}) — {txt_filename}"
        
        return display_text, str(txt_path), status
        
    except Exception as e:
        status = f"❌ 오류: {str(e)}"
        return "", None, status


def _fmt_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


# ──────────────────────────────────────────────
# UI 구성
# ──────────────────────────────────────────────

def create_app():
    # 프리셋 목록
    presets = list_presets()
    preset_choices = ["없음"] + list(presets.keys())

    with gr.Blocks(title="noma-scribe") as app:
        
        # 헤더
        gr.Markdown("# 🎙️ noma-scribe", elem_classes=["main-title"])
        gr.Markdown("오디오 파일을 올리고 버튼만 누르면 텍스트로 전사합니다. 완전 로컬, 무료.", elem_classes=["sub-title"])
        
        with gr.Row():
            # ── 왼쪽: 입력 ──
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    label="오디오 파일",
                    type="filepath",
                    sources=["upload"],
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
                        label="타임스탬프 포함",
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
                    label="전사 결과",
                    lines=18,
                    max_lines=40,
                    interactive=False,
                )
                download_file = gr.File(
                    label="다운로드",
                    visible=True,
                )
        
        # 이벤트 연결
        transcribe_btn.click(
            fn=run_transcription,
            inputs=[audio_input, preset_dropdown, custom_prompt, language, timestamps],
            outputs=[output_text, download_file, status_text],
        )
        
        # 프리셋 설명 표시
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

    # Gradio 6.0+: theme/css를 launch()로 이동
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
        inbrowser=True,  # 자동으로 브라우저 열기
        theme=theme,
        css=css,
    )
