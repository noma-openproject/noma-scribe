#!/usr/bin/env python3
"""noma-scribe: Apple Silicon 최적화 로컬 음성 전사 도구.

사용법:
    python transcribe.py audio.m4a
    python transcribe.py audio.m4a --preset dev
    python transcribe.py ./recordings/ --preset dev -o ./transcripts/
    python transcribe.py audio.m4a --prompt "API 리팩토링 마이크로서비스"
    python transcribe.py --list-presets
"""

import argparse
import sys
from pathlib import Path

from core.engine import transcribe, save_result, DEFAULT_MODEL
from core.presets import load_prompt, list_presets
from core.utils import find_audio_files, build_output_path, format_duration, format_file_size


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="noma-scribe",
        description="🎙️ noma-scribe — Apple Silicon 로컬 음성 전사 도구",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python transcribe.py meeting.m4a                     # 기본 한국어 전사
  python transcribe.py meeting.m4a --preset dev        # 개발 용어 프리셋
  python transcribe.py meeting.m4a --preset medical    # 의료 용어 프리셋
  python transcribe.py ./recordings/ -o ./out/         # 폴더 일괄 전사
  python transcribe.py meeting.m4a --prompt "Next.js Supabase 리팩토링"
  python transcribe.py meeting.m4a --timestamps        # 타임스탬프 포함
  python transcribe.py meeting.m4a --lang en           # 영어 전사
  python transcribe.py --list-presets                  # 프리셋 목록 조회
        """,
    )
    
    # 위치 인자
    parser.add_argument(
        "input",
        nargs="?",
        help="오디오 파일 또는 디렉토리 경로",
    )
    
    # 프롬프트 옵션 (상호 배타적)
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument(
        "--preset", "-p",
        help="프리셋 이름 (dev, medical, general 등)",
    )
    prompt_group.add_argument(
        "--prompt",
        help="도메인 용어 힌트 (직접 입력)",
    )
    prompt_group.add_argument(
        "--prompt-file",
        help="도메인 용어 힌트 파일 경로",
    )
    
    # 출력 옵션
    parser.add_argument(
        "--output", "-o",
        help="출력 디렉토리 (미지정 시 원본 파일과 같은 위치)",
    )
    parser.add_argument(
        "--timestamps", "-t",
        action="store_true",
        help="세그먼트별 타임스탬프 포함",
    )
    
    # 모델/언어 옵션
    parser.add_argument(
        "--lang",
        default="ko",
        help="전사 언어 코드 (기본: ko)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Whisper 모델 (기본: {DEFAULT_MODEL})",
    )
    
    # 기타
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="세그먼트별 실시간 출력",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="사용 가능한 프리셋 목록 출력",
    )
    
    return parser


def print_header():
    print()
    print("  🎙️  noma-scribe — 로컬 음성 전사")
    print("  ─────────────────────────────────")


def print_presets():
    print_header()
    print()
    presets = list_presets()
    for name, desc in presets.items():
        print(f"  {name:<12} {desc}")
    print()
    print("  사용법: python transcribe.py audio.m4a --preset dev")
    print()


def main():
    parser = create_parser()
    args = parser.parse_args()
    
    # 프리셋 목록 조회
    if args.list_presets:
        print_presets()
        return
    
    # 입력 필수 검증
    if not args.input:
        parser.print_help()
        sys.exit(1)
    
    print_header()
    
    # 1. 오디오 파일 탐지
    try:
        audio_files = find_audio_files(args.input)
    except (FileNotFoundError, ValueError) as e:
        print(f"\n  ❌ {e}")
        sys.exit(1)
    
    # 2. 프롬프트 로딩
    try:
        prompt = load_prompt(
            preset=args.preset,
            prompt_text=args.prompt,
            prompt_file=args.prompt_file,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"\n  ❌ {e}")
        sys.exit(1)
    
    # 3. 설정 요약 출력
    print(f"\n  모델:   {args.model}")
    print(f"  언어:   {args.lang}")
    if prompt:
        prompt_source = args.preset or args.prompt_file or "직접 입력"
        prompt_preview = prompt[:60] + "..." if len(prompt) > 60 else prompt
        print(f"  프롬프트: [{prompt_source}] {prompt_preview}")
    print(f"  파일:   {len(audio_files)}개")
    print()
    
    # 4. 전사 실행
    total_elapsed = 0.0
    success_count = 0
    
    for i, audio_path in enumerate(audio_files, 1):
        prefix = f"  [{i}/{len(audio_files)}]"
        file_size = format_file_size(audio_path)
        print(f"{prefix} {audio_path.name} ({file_size})")
        print(f"         전사 중...", end="", flush=True)
        
        try:
            result = transcribe(
                audio_path=str(audio_path),
                language=args.lang,
                model=args.model,
                initial_prompt=prompt,
                word_timestamps=args.timestamps,
                verbose=args.verbose,
            )
            
            # 결과 저장
            output_path = build_output_path(audio_path, args.output)
            save_result(result, output_path, include_timestamps=args.timestamps)
            
            elapsed_str = format_duration(result.duration_seconds)
            total_elapsed += result.duration_seconds
            success_count += 1
            
            print(f"\r         ✅ 완료 ({elapsed_str}) → {output_path.name}")
            
            # verbose 모드가 아닐 때 텍스트 미리보기 (첫 100자)
            if not args.verbose and result.text:
                preview = result.text[:100]
                if len(result.text) > 100:
                    preview += "..."
                print(f"         📝 {preview}")
            print()
            
        except Exception as e:
            print(f"\r         ❌ 실패: {e}")
            print()
    
    # 5. 요약
    print("  ─────────────────────────────────")
    print(f"  완료: {success_count}/{len(audio_files)}개 | 총 소요: {format_duration(total_elapsed)}")
    print()


if __name__ == "__main__":
    main()
