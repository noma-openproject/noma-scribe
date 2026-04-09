"""도메인별 프리셋 프롬프트 관리."""

import os
from pathlib import Path
from typing import Optional

# 프리셋 디렉토리: 이 파일 기준으로 상위의 presets/ 폴더
PRESETS_DIR = Path(__file__).parent.parent / "presets"

# 내장 프리셋 목록
BUILTIN_PRESETS = {
    "dev": "개발회의 (프레임워크, DevOps, AI/클라우드 용어)",
    "medical": "의료/성형외과 (진료, 시술, 해부학 용어)",
    "general": "일반 (프롬프트 없음)",
    "meeting": "일반 회의 (아젠다, 액션아이템, 스크럼 용어)",
    "interview": "사용자/채용 인터뷰 (지원자, 역량, 페인포인트 용어)",
    "finance": "금융/회계 (매출, 재무제표, 밸류에이션 용어)",
}


def list_presets() -> dict:
    """사용 가능한 프리셋 목록을 반환한다."""
    presets = dict(BUILTIN_PRESETS)
    
    # 커스텀 프리셋 탐지 (내장 목록에 없는 .txt 파일)
    if PRESETS_DIR.exists():
        for f in sorted(PRESETS_DIR.glob("*.txt")):
            name = f.stem
            if name not in presets:
                presets[name] = f"커스텀 ({f.name})"
    
    return presets


def load_prompt(
    preset: Optional[str] = None,
    prompt_text: Optional[str] = None,
    prompt_file: Optional[str] = None,
) -> Optional[str]:
    """프리셋, 직접 입력, 또는 파일에서 프롬프트를 로딩한다.
    
    우선순위: prompt_text > prompt_file > preset
    """
    # 1) 직접 입력된 프롬프트
    if prompt_text:
        return prompt_text.strip()
    
    # 2) 외부 파일에서 로딩
    if prompt_file:
        path = Path(prompt_file)
        if not path.exists():
            raise FileNotFoundError(f"프롬프트 파일을 찾을 수 없습니다: {prompt_file}")
        text = path.read_text(encoding="utf-8").strip()
        return text if text else None
    
    # 3) 프리셋에서 로딩
    if preset:
        preset_path = PRESETS_DIR / f"{preset}.txt"
        if not preset_path.exists():
            available = ", ".join(list_presets().keys())
            raise ValueError(
                f"프리셋 '{preset}'을(를) 찾을 수 없습니다.\n"
                f"사용 가능: {available}"
            )
        text = preset_path.read_text(encoding="utf-8").strip()
        return text if text else None
    
    # 아무것도 지정 안 됨
    return None
