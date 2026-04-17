#!/usr/bin/env python3
"""noma-scribe v0.6: Apple Silicon 로컬 음성 전사 앱.

주요 변경 (v0.6):
- 고급 옵션: 용어집 적용 / 한국어 문장 정리 체크박스
- 용어집 관리 탭 (~/.noma-scribe/glossary.json)
- 전사 후 키워드별 "용어집 추가" 기능 (Dataframe UI)
"""

import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

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
from core.auto_glossary import detect_term_variants, format_cluster_preview
from core.glossary import (
    DEFAULT_GLOSSARY_PATH,
    add_term,
    apply_glossary,
    load_glossary,
    remove_term,
    save_glossary,
)
from core.keywords import extract_keywords, format_keywords
from core.korean_normalizer import is_available as kss_available
from core.srt import save_srt, segments_to_srt
from core.two_pass import two_pass_transcribe
from core.utils import format_duration, SUPPORTED_EXTENSIONS


# ──────────────────────────────────────────────
# 헬퍼 (기존)
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
# 용어집 UI 헬퍼 (v0.6)
# ──────────────────────────────────────────────

def _glossary_to_rows(data: dict) -> List[List[str]]:
    """용어집 dict → Dataframe 행 [[canonical, aliases(쉼표)]]"""
    rows = []
    for t in data.get("terms", []):
        canonical = t.get("canonical", "")
        aliases = ", ".join(t.get("aliases", []) or [])
        rows.append([canonical, aliases])
    return rows


def _refresh_glossary_display() -> List[List[str]]:
    return _glossary_to_rows(load_glossary())


def ui_add_glossary_term(canonical: str, aliases_str: str) -> Tuple[List[List[str]], str, str]:
    canonical = (canonical or "").strip()
    if not canonical:
        gr.Warning("표준형(canonical)을 입력해주세요.")
        return _refresh_glossary_display(), canonical, aliases_str

    aliases = [a.strip() for a in (aliases_str or "").split(",") if a.strip()]
    try:
        add_term(canonical, aliases)
        gr.Info(f"용어 추가: {canonical} ({len(aliases)}개 별칭)")
    except Exception as e:
        gr.Warning(f"용어 추가 실패: {e}")
    return _refresh_glossary_display(), "", ""


def ui_remove_glossary_term(canonical: str) -> Tuple[List[List[str]], str]:
    canonical = (canonical or "").strip()
    if not canonical:
        gr.Warning("삭제할 표준형을 입력해주세요.")
        return _refresh_glossary_display(), canonical
    try:
        remove_term(canonical)
        gr.Info(f"용어 삭제: {canonical}")
    except Exception as e:
        gr.Warning(f"용어 삭제 실패: {e}")
    return _refresh_glossary_display(), ""


def ui_add_keyword_to_glossary(keyword: str) -> List[List[str]]:
    keyword = (keyword or "").strip()
    if not keyword:
        gr.Warning("추가할 키워드를 입력해주세요.")
    else:
        try:
            add_term(keyword, [])
            gr.Info(f"'{keyword}' 용어집에 추가됨 (별칭은 용어집 관리 탭에서 추가)")
        except Exception as e:
            gr.Warning(f"용어집 추가 실패: {e}")
    return _refresh_glossary_display()


# ──────────────────────────────────────────────
# v0.6.1: 자동 감지 클러스터 헬퍼
# ──────────────────────────────────────────────

def _detect_clusters(text: str) -> List[dict]:
    """현재 용어집을 참고하여 용어 변형 클러스터 감지."""
    if not text:
        return []
    current = load_glossary()
    existing_canon = {t.get("canonical", "") for t in current.get("terms", [])}
    existing_alias = set()
    for t in current.get("terms", []):
        for a in t.get("aliases", []) or []:
            existing_alias.add(a)
    try:
        return detect_term_variants(
            text,
            min_frequency=2,
            similarity_threshold=80,
            existing_canonicals=existing_canon,
            existing_aliases=existing_alias,
        )
    except Exception:
        return []


def ui_register_cluster(
    idx: int,
    clusters: List[dict],
    current_text: str,
) -> Tuple[List[dict], str]:
    """클러스터 한 개를 용어집에 등록 + 현재 전사 텍스트에 즉시 재적용.

    Returns: (새 클러스터 리스트, 업데이트된 전사 텍스트)
    """
    if idx < 0 or idx >= len(clusters):
        gr.Warning("유효하지 않은 클러스터 인덱스")
        return clusters, current_text

    cluster = clusters[idx]
    canonical = cluster["canonical"]
    aliases = list(cluster.get("aliases", []))

    try:
        add_term(canonical, aliases)
        gr.Info(f"'{canonical}' ← {aliases} 용어집 등록 완료")
    except Exception as e:
        gr.Warning(f"등록 실패: {e}")
        return clusters, current_text

    # 전사 결과에 즉시 재적용
    updated_text = current_text
    if current_text:
        try:
            updated_text = apply_glossary(current_text, load_glossary())
        except Exception:
            pass

    # 해당 클러스터 제거
    new_clusters = clusters[:idx] + clusters[idx + 1:]
    return new_clusters, updated_text


def ui_ignore_cluster(idx: int, clusters: List[dict]) -> List[dict]:
    """클러스터를 숨김 (용어집에 추가 안 함, UI 에서만 제거)."""
    if idx < 0 or idx >= len(clusters):
        return clusters
    new_clusters = clusters[:idx] + clusters[idx + 1:]
    gr.Info(f"'{clusters[idx]['canonical']}' 무시됨")
    return new_clusters


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
    progress=gr.Progress(),
):
    files = _normalize_files(audio_files)
    if not files:
        gr.Warning("오디오 파일을 먼저 업로드해주세요.")
        return "", None, "", [], _refresh_glossary_display(), []

    valid_files = [fp for fp in files if Path(fp).suffix.lower() in SUPPORTED_EXTENSIONS]
    skipped = [Path(fp).name for fp in files if Path(fp).suffix.lower() not in SUPPORTED_EXTENSIONS]
    if skipped:
        gr.Warning(f"지원하지 않는 형식 {len(skipped)}개 무시")
    if not valid_files:
        return "", None, "❌ 지원하는 오디오 파일이 없습니다.", [], _refresh_glossary_display(), []

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
        return "", None, f"❌ {e}", [], _refresh_glossary_display(), []

    use_two_pass = two_pass_enabled
    lang_code = "ko"
    t_start = _parse_time(start_time)
    t_end = _parse_time(end_time)
    needs_timestamps = output_format in ("타임스탬프 포함 (.txt)", "자막 (.srt)")

    n = len(valid_files)
    output_dir = Path(tempfile.mkdtemp(prefix="noma-scribe-"))
    result_paths: List[Path] = []
    display_blocks: List[str] = []
    keyword_rows: List[List] = []
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
                result = transcribe(
                    audio_path=actual_path,
                    language=lang_code,
                    model=model,
                    progress_callback=_make_cb(idx),
                    use_glossary=use_glossary,
                    use_korean_norm=use_korean_norm,
                )

            total_elapsed += result.duration_seconds
            success_count += 1

            aligned_result = None
            if needs_timestamps and output_format != "자막 (.srt)":
                try:
                    aligned_result = align_words(result)
                except Exception:
                    pass

            progress(base_pct + 0.9 / n, desc=f"{idx}/{n} 후처리 중...")

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

            body = _format_body(result, output_format, aligned_result=aligned_result)
            if n == 1:
                display_blocks.append(body)
            else:
                header = f"━━━ [{idx}/{n}] {audio_path.name} ━━━"
                display_blocks.append(f"{header}\n{body}")

            if idx == n:
                full_text = " ".join(
                    block.split("\n", 1)[-1] if "━━━" in block else block
                    for block in display_blocks
                )
                kw = extract_keywords(full_text, top_n=15, min_count=2)
                keyword_rows = [[w, c] for w, c in kw]

        except Exception as e:
            if n == 1:
                progress(1.0, desc="실패")
                return "", None, f"❌ {audio_path.name}: {e}", [], _refresh_glossary_display(), []
            header = f"━━━ [{idx}/{n}] {audio_path.name} ━━━"
            display_blocks.append(f"{header}\n❌ 오류: {e}")
        finally:
            if sliced and Path(actual_path).exists():
                try:
                    Path(actual_path).unlink()
                except Exception:
                    pass

    progress(1.0, desc="완료")

    if not result_paths:
        return "\n\n".join(display_blocks), None, f"❌ 모두 실패 (0/{n})", keyword_rows, _refresh_glossary_display(), []

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
    extras = []
    if use_glossary:
        extras.append("용어집")
    if use_korean_norm:
        extras.append("한글정리")
    extras_tag = f" [{'/'.join(extras)}]" if extras else ""

    if n == 1:
        status = f"✅ 완료! [{status_mode}{two_pass_tag}]{extras_tag} ({elapsed_str}) — {download_label}"
    else:
        status = f"✅ 완료! [{status_mode}{two_pass_tag}]{extras_tag} {success_count}/{n}개 ({elapsed_str}) — {download_label}"

    # v0.6.1: 자동 감지 클러스터 계산
    final_text = "\n\n".join(display_blocks)
    flat_text = " ".join(
        block.split("\n", 1)[-1] if "━━━" in block else block
        for block in display_blocks
    )
    clusters = _detect_clusters(flat_text)

    return final_text, download_path, status, keyword_rows, _refresh_glossary_display(), clusters


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

                        transcribe_btn = gr.Button(
                            "🎙️ 전사 시작",
                            variant="primary",
                            size="lg",
                        )

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

                        gr.Markdown("### 📊 자주 등장한 단어 (Top 15)")
                        keywords_table = gr.Dataframe(
                            headers=["단어", "빈도"],
                            datatype=["str", "number"],
                            col_count=(2, "fixed"),
                            row_count=(5, "dynamic"),
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

                        @gr.render(inputs=[clusters_state, output_text])
                        def _render_clusters(clusters_value, current_text):
                            if not clusters_value:
                                gr.Markdown("_(자동 감지된 변형 없음 — 전사를 실행하면 여기에 후보가 나타납니다.)_")
                                return
                            for idx, cluster in enumerate(clusters_value):
                                preview = format_cluster_preview(cluster)
                                with gr.Row():
                                    gr.Markdown(preview)
                                    register_btn = gr.Button(
                                        f"[등록]",
                                        variant="primary",
                                        size="sm",
                                        scale=0,
                                    )
                                    ignore_btn = gr.Button(
                                        f"[무시]",
                                        size="sm",
                                        scale=0,
                                    )
                                # closure 로 idx 고정
                                register_btn.click(
                                    fn=lambda clusters, text, i=idx: ui_register_cluster(i, clusters, text),
                                    inputs=[clusters_state, output_text],
                                    outputs=[clusters_state, output_text],
                                )
                                ignore_btn.click(
                                    fn=lambda clusters, i=idx: ui_ignore_cluster(i, clusters),
                                    inputs=[clusters_state],
                                    outputs=[clusters_state],
                                )

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
                    col_count=(2, "fixed"),
                    row_count=(5, "dynamic"),
                    interactive=False,
                    value=_refresh_glossary_display(),
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
            ],
            outputs=[output_text, download_file, status_text, keywords_table, glossary_table, clusters_state],
        )

        add_kw_btn.click(
            fn=ui_add_keyword_to_glossary,
            inputs=[selected_keyword],
            outputs=[glossary_table],
        )

        add_term_btn.click(
            fn=ui_add_glossary_term,
            inputs=[new_canonical, new_aliases],
            outputs=[glossary_table, new_canonical, new_aliases],
        )

        remove_term_btn.click(
            fn=ui_remove_glossary_term,
            inputs=[remove_canonical],
            outputs=[glossary_table, remove_canonical],
        )

        refresh_btn.click(
            fn=_refresh_glossary_display,
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
