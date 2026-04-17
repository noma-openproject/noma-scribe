"""용어집/클러스터 관련 UI 헬퍼."""

from __future__ import annotations

from typing import List, Tuple

import gradio as gr

from core.auto_glossary import detect_term_variants, format_cluster_preview
from core.glossary import add_term, apply_glossary, load_glossary, remove_term


def glossary_to_rows(data: dict) -> List[List[str]]:
    """용어집 dict → Dataframe 행 [[canonical, aliases(쉼표)]]"""
    rows = []
    for term in data.get("terms", []):
        canonical = term.get("canonical", "")
        aliases = ", ".join(term.get("aliases", []) or [])
        rows.append([canonical, aliases])
    return rows


def refresh_glossary_display() -> List[List[str]]:
    return glossary_to_rows(load_glossary())


def add_glossary_term(canonical: str, aliases_str: str) -> Tuple[List[List[str]], str, str]:
    canonical = (canonical or "").strip()
    if not canonical:
        gr.Warning("표준형(canonical)을 입력해주세요.")
        return refresh_glossary_display(), canonical, aliases_str

    aliases = [alias.strip() for alias in (aliases_str or "").split(",") if alias.strip()]
    try:
        add_term(canonical, aliases)
        gr.Info(f"용어 추가: {canonical} ({len(aliases)}개 별칭)")
    except Exception as e:
        gr.Warning(f"용어 추가 실패: {e}")
    return refresh_glossary_display(), "", ""


def remove_glossary_term(canonical: str) -> Tuple[List[List[str]], str]:
    canonical = (canonical or "").strip()
    if not canonical:
        gr.Warning("삭제할 표준형을 입력해주세요.")
        return refresh_glossary_display(), canonical
    try:
        remove_term(canonical)
        gr.Info(f"용어 삭제: {canonical}")
    except Exception as e:
        gr.Warning(f"용어 삭제 실패: {e}")
    return refresh_glossary_display(), ""


def add_keyword_to_glossary(keyword: str) -> List[List[str]]:
    keyword = (keyword or "").strip()
    if not keyword:
        gr.Warning("추가할 키워드를 입력해주세요.")
    else:
        try:
            add_term(keyword, [])
            gr.Info(f"'{keyword}' 용어집에 추가됨 (별칭은 용어집 관리 탭에서 추가)")
        except Exception as e:
            gr.Warning(f"용어집 추가 실패: {e}")
    return refresh_glossary_display()


def detect_clusters_from_text(text: str) -> List[dict]:
    """현재 용어집을 참고하여 용어 변형 클러스터 감지."""
    if not text:
        return []

    current = load_glossary()
    existing_canon = {term.get("canonical", "") for term in current.get("terms", [])}
    existing_alias = set()
    for term in current.get("terms", []):
        for alias in term.get("aliases", []) or []:
            existing_alias.add(alias)

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


def register_cluster(idx: int, clusters: List[dict], current_text: str) -> Tuple[List[dict], str]:
    """클러스터 한 개를 용어집에 등록 + 현재 전사 텍스트에 즉시 재적용."""
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

    updated_text = current_text
    if current_text:
        try:
            updated_text = apply_glossary(current_text, load_glossary())
        except Exception:
            pass

    new_clusters = clusters[:idx] + clusters[idx + 1 :]
    return new_clusters, updated_text


def ignore_cluster(idx: int, clusters: List[dict]) -> List[dict]:
    """클러스터를 숨김 (용어집에 추가 안 함, UI 에서만 제거)."""
    if idx < 0 or idx >= len(clusters):
        return clusters
    new_clusters = clusters[:idx] + clusters[idx + 1 :]
    gr.Info(f"'{clusters[idx]['canonical']}' 무시됨")
    return new_clusters


def mount_cluster_render(clusters_state, output_text) -> None:
    """자동 감지된 클러스터 UI를 렌더링한다."""

    @gr.render(inputs=[clusters_state, output_text])
    def _render_clusters(clusters_value, current_text):
        if not clusters_value:
            gr.Markdown("_(자동 감지된 변형 없음 — 전사를 실행하면 여기에 후보가 나타납니다.)_")
            return

        for idx, cluster in enumerate(clusters_value):
            preview = format_cluster_preview(cluster)
            with gr.Row():
                gr.Markdown(preview)
                register_btn = gr.Button("[등록]", variant="primary", size="sm", scale=0)
                ignore_btn = gr.Button("[무시]", size="sm", scale=0)

            register_btn.click(
                fn=lambda clusters, text, i=idx: register_cluster(i, clusters, text),
                inputs=[clusters_state, output_text],
                outputs=[clusters_state, output_text],
            )
            ignore_btn.click(
                fn=lambda clusters, i=idx: ignore_cluster(i, clusters),
                inputs=[clusters_state],
                outputs=[clusters_state],
            )
