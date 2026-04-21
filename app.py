#!/usr/bin/env python3
"""noma-scribe Gradio 앱."""

import gradio as gr

from core.batch_transcription import (
    BatchTranscriptionOptions,
    transcribe_batch,
)
from core.engine import MODELS, is_model_available
from core.glossary import DEFAULT_GLOSSARY_PATH
from core.korean_normalizer import is_available as kss_available
from ui_glossary import (
    add_glossary_term,
    add_keyword_to_glossary,
    detect_clusters_from_text,
    mount_cluster_render,
    refresh_glossary_display,
    remove_glossary_term,
)


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
    use_glossary: bool,
    use_korean_norm: bool,
    debug_mode: bool,
    progress=gr.Progress(),
):
    options = BatchTranscriptionOptions(
        mode_label=mode,
        two_pass_enabled=two_pass_enabled,
        output_format=output_format,
        start_time=start_time,
        end_time=end_time,
        use_glossary=use_glossary,
        use_korean_norm=use_korean_norm,
        debug_mode=debug_mode,
    )

    result = transcribe_batch(
        audio_files,
        options,
        detect_clusters_from_text,
        progress_callback=lambda value, desc: progress(value, desc=desc),
        warning_callback=gr.Warning,
    )

    return (
        result.text,
        result.download_path,
        result.status,
        result.keyword_rows,
        refresh_glossary_display(),
        result.clusters,
    )


# ──────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────

def create_app():
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
            f"완전 로컬 음성 전사 · whispermlx + 용어집 + 한국어 정리 "
            f"(KSS: {'✅' if kss_available() else '❌ 미설치'})",
            elem_classes=["sub-title"],
        )

        with gr.Tabs():
            # ════════════════════════════════════
            # Tab 1: 전사
            # ════════════════════════════════════
            with gr.TabItem("🎙️ 전사"):
                with gr.Row():
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
                                start_time_in = gr.Textbox(
                                    label="시작 시간 (선택)",
                                    placeholder="MM:SS",
                                    max_lines=1,
                                )
                                end_time_in = gr.Textbox(
                                    label="끝 시간 (선택)",
                                    placeholder="MM:SS",
                                    max_lines=1,
                                )

                        with gr.Accordion("고급 옵션", open=False):
                            use_glossary_check = gr.Checkbox(
                                label="용어집 적용 (고유명사 자동 치환)",
                                value=True,
                            )
                            use_korean_norm_check = gr.Checkbox(
                                label="한국어 문장 정리 (KSS 띄어쓰기 + 문단 분리)",
                                value=True,
                            )
                            debug_mode_check = gr.Checkbox(
                                label="디버그 모드 (상세 로그/예외 표시)",
                                value=False,
                            )

                        transcribe_btn = gr.Button(
                            "🎙️ 전사 시작",
                            variant="primary",
                            size="lg",
                        )

                    with gr.Column(scale=1):
                        status_text = gr.Textbox(
                            label="상태",
                            lines=12,
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

                        gr.Markdown("### 📊 자주 등장한 단어 (Top 15)")
                        keywords_table = gr.Dataframe(
                            headers=["단어", "빈도"],
                            datatype=["str", "number"],
                            column_count=(2, "fixed"),
                            row_count=5,
                            interactive=False,
                        )
                        with gr.Row():
                            selected_keyword = gr.Textbox(
                                label="용어집에 추가할 단어",
                                placeholder="위 표에서 단어를 복사해 입력",
                                max_lines=1,
                                scale=3,
                            )
                            add_kw_btn = gr.Button("➕ 용어집 추가", scale=1)

                        # ── v0.6.1: 자동 감지된 용어 변형 섹션 ──
                        gr.Markdown("### 🔍 자동 감지된 용어 변형")
                        gr.Markdown(
                            "_전사 결과에서 서로 비슷한 단어(예: 프레모/프리모/프레머)를 자동으로 묶어 "
                            "용어집 후보로 제안합니다. [등록] 누르면 즉시 반영되어 위 전사 결과가 업데이트됩니다._",
                            elem_classes=["sub-title"],
                        )
                        clusters_state = gr.State(value=[])

                        mount_cluster_render(clusters_state, output_text)

            # ════════════════════════════════════
            # Tab 2: 용어집 관리
            # ════════════════════════════════════
            with gr.TabItem("📚 용어집 관리"):
                gr.Markdown(
                    f"**저장 위치**: `{DEFAULT_GLOSSARY_PATH}`\n\n"
                    "고유명사, 브랜드명, 전문용어의 **표준형(canonical)** 과 자주 오인식되는 "
                    "**별칭(aliases)** 을 등록하면, 전사 결과가 자동으로 표준형으로 치환됩니다. "
                    "별칭 외에도 RapidFuzz 유사도 85% 이상인 단어는 자동 치환됩니다."
                )

                glossary_table = gr.Dataframe(
                    headers=["표준형 (canonical)", "별칭 (aliases, 쉼표 구분)"],
                    datatype=["str", "str"],
                    column_count=(2, "fixed"),
                    row_count=5,
                    interactive=False,
                    value=refresh_glossary_display(),
                )

                with gr.Row():
                    refresh_btn = gr.Button("🔄 새로고침", scale=1)

                gr.Markdown("### ➕ 용어 추가")
                with gr.Row():
                    new_canonical = gr.Textbox(
                        label="표준형",
                        placeholder="예: Premo",
                        max_lines=1,
                        scale=2,
                    )
                    new_aliases = gr.Textbox(
                        label="별칭 (쉼표 구분)",
                        placeholder="예: 프레모, 프리모, 프레머",
                        max_lines=1,
                        scale=3,
                    )
                    add_term_btn = gr.Button("추가", variant="primary", scale=1)

                gr.Markdown("### 🗑️ 용어 삭제")
                with gr.Row():
                    remove_canonical = gr.Textbox(
                        label="삭제할 표준형",
                        placeholder="예: Premo",
                        max_lines=1,
                        scale=3,
                    )
                    remove_term_btn = gr.Button("삭제", variant="stop", scale=1)

        # 이벤트
        transcribe_btn.click(
            fn=run_transcription,
            inputs=[
                audio_input, mode_select, two_pass_check, format_select,
                start_time_in, end_time_in, use_glossary_check, use_korean_norm_check,
                debug_mode_check,
            ],
            outputs=[output_text, download_file, status_text, keywords_table, glossary_table, clusters_state],
        )

        add_kw_btn.click(
            fn=add_keyword_to_glossary,
            inputs=[selected_keyword],
            outputs=[glossary_table],
        )

        add_term_btn.click(
            fn=add_glossary_term,
            inputs=[new_canonical, new_aliases],
            outputs=[glossary_table, new_canonical, new_aliases],
        )

        remove_term_btn.click(
            fn=remove_glossary_term,
            inputs=[remove_canonical],
            outputs=[glossary_table, remove_canonical],
        )

        refresh_btn.click(
            fn=refresh_glossary_display,
            inputs=[],
            outputs=[glossary_table],
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
