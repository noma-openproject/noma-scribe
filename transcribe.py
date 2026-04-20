#!/usr/bin/env python3
"""noma-scribe: Apple Silicon 최적화 로컬 음성 전사 도구.

사용법:
    python transcribe.py audio.m4a
    python transcribe.py audio.m4a --precise
    python transcribe.py audio.m4a --format srt
    python transcribe.py audio.m4a --start 01:00 --end 05:30
    python transcribe.py ./recordings/ -o ./transcripts/
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

from core.diagnostics import RunDiagnostics, format_stage_timings
from core.engine import resolve_model_path, save_result, slice_audio, transcribe
from core.keywords import extract_keywords, format_keywords
from core.srt import save_srt
from core.two_pass import two_pass_transcribe
from core.utils import find_audio_files, build_output_path, format_duration, format_file_size


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="noma-scribe",
        description="🎙️ noma-scribe — Apple Silicon 로컬 음성 전사 도구",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python transcribe.py meeting.m4a                     # 빠른 모드 (기본)
  python transcribe.py meeting.m4a --precise           # 정밀 모드 (2-pass)
  python transcribe.py meeting.m4a --format srt        # SRT 자막 출력
  python transcribe.py meeting.m4a --start 01:00 --end 05:30   # 구간 전사
  python transcribe.py ./recordings/ -o ./out/         # 폴더 일괄 전사
  python transcribe.py meeting.m4a --timestamps        # 타임스탬프 포함
  python transcribe.py meeting.m4a --lang en           # 영어 전사
  python transcribe.py meeting.m4a --keywords          # 키워드 빈도 표시
        """,
    )

    parser.add_argument("input", nargs="?", help="오디오 파일 또는 디렉토리 경로")
    parser.add_argument("--output", "-o", help="출력 디렉토리")

    # 모드
    parser.add_argument("--precise", action="store_true",
                        help="정밀 모드 (large-v3 풀모델, 2-pass 용어 추출)")

    # 포맷
    parser.add_argument("--format", "-f", choices=["txt", "timestamps", "srt"],
                        default="txt", help="출력 포맷 (기본: txt)")
    parser.add_argument("--timestamps", "-t", action="store_true",
                        help="--format timestamps 의 단축 옵션")

    # 구간
    parser.add_argument("--start", help="시작 시간 (MM:SS 또는 HH:MM:SS)")
    parser.add_argument("--end", help="끝 시간 (MM:SS 또는 HH:MM:SS)")

    # 언어/모델
    parser.add_argument("--lang", default="ko", help="전사 언어 코드 (기본: ko)")

    # 기타
    parser.add_argument("--keywords", "-k", action="store_true",
                        help="전사 후 키워드 빈도 Top 15 표시")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="세그먼트별 실시간 출력")
    parser.add_argument("--debug", action="store_true",
                        help="상세 로그와 traceback 출력")

    return parser


def print_header():
    print()
    print("  🎙️  noma-scribe — 로컬 음성 전사")
    print("  ─────────────────────────────────")


def main():
    parser = create_parser()
    args = parser.parse_args()

    if not args.input:
        parser.print_help()
        sys.exit(1)

    print_header()
    diagnostics = RunDiagnostics(origin="cli", debug=args.debug)

    # 포맷 결정
    if args.timestamps:
        args.format = "timestamps"

    # 모델 선택
    model_key = "precise" if args.precise else "fast"
    try:
        model = resolve_model_path(model_key)
    except FileNotFoundError as e:
        print(f"\n  ❌ {e}")
        sys.exit(1)
    mode_label = "정밀" if args.precise else "빠른"

    # 오디오 파일 탐지
    try:
        audio_files = find_audio_files(args.input)
    except (FileNotFoundError, ValueError) as e:
        print(f"\n  ❌ {e}")
        sys.exit(1)

    print(f"\n  모드:   {mode_label} ({Path(model).name})")
    print(f"  언어:   {args.lang}")
    print(f"  포맷:   {args.format}")
    if args.start or args.end:
        print(f"  구간:   {args.start or '처음'} ~ {args.end or '끝'}")
    print(f"  파일:   {len(audio_files)}개")
    print(f"  로그:   {diagnostics.log_path}")
    print()

    total_elapsed = 0.0
    success_count = 0
    all_texts = []
    build_processed_segments = args.format != "txt"

    for i, audio_path in enumerate(audio_files, 1):
        prefix = f"  [{i}/{len(audio_files)}]"
        file_size = format_file_size(audio_path)
        print(f"{prefix} {audio_path.name} ({file_size})")

        # 구간 슬라이싱
        actual_path = str(audio_path)
        sliced = False
        if args.start or args.end:
            try:
                slice_start = time.time()
                actual_path = slice_audio(str(audio_path), args.start, args.end)
                sliced = actual_path != str(audio_path)
                diagnostics.stage(f"{audio_path.name}.slice_audio", time.time() - slice_start)
            except Exception as e:
                print(f"         ⚠️ 구간 자르기 실패: {e}")
                diagnostics.note(f"warn slice_audio {audio_path.name}: {e}")

        print(f"         {mode_label} 전사 중...", end="", flush=True)

        try:
            if args.precise:
                result = two_pass_transcribe(
                    audio_path=actual_path,
                    language=args.lang,
                    model=model,
                    build_processed_segments=build_processed_segments,
                    warning_callback=lambda message: diagnostics.note(f"warn {audio_path.name}: {message}"),
                    debug=args.debug,
                )
            else:
                result = transcribe(
                    audio_path=actual_path,
                    language=args.lang,
                    model=model,
                    verbose=args.verbose,
                    build_processed_segments=build_processed_segments,
                    warning_callback=lambda message: diagnostics.note(f"warn {audio_path.name}: {message}"),
                    debug=args.debug,
                )

            diagnostics.stage_map(audio_path.name, result.stage_timings)

            # 결과 저장
            save_start = time.time()
            if args.format == "srt":
                output_path = build_output_path(audio_path, args.output)
                output_path = output_path.with_suffix(".srt")
                save_srt(result.processed_segments or result.segments, output_path)
            else:
                output_path = build_output_path(audio_path, args.output)
                save_result(result, output_path,
                            include_timestamps=(args.format == "timestamps"))
            diagnostics.stage(f"{audio_path.name}.save_result", time.time() - save_start)

            elapsed_str = format_duration(result.duration_seconds)
            total_elapsed += result.duration_seconds
            success_count += 1
            all_texts.append(result.text)

            print(f"\r         ✅ 완료 ({elapsed_str}) → {output_path.name}")
            stage_summary = format_stage_timings(result.stage_timings)
            if stage_summary:
                print("         단계별 소요:")
                for line in stage_summary.splitlines():
                    print(f"         {line}")

            if not args.verbose and result.text:
                preview = result.text[:100] + ("..." if len(result.text) > 100 else "")
                print(f"         📝 {preview}")
            print()

        except Exception as e:
            print(f"\r         ❌ 실패: {e}")
            diagnostics.exception(audio_path.name, str(e), traceback.format_exc())
            if args.debug:
                print(traceback.format_exc().rstrip())
            print()
        finally:
            if sliced and Path(actual_path).exists():
                try:
                    Path(actual_path).unlink()
                except Exception:
                    pass

    # 요약
    print("  ─────────────────────────────────")
    print(f"  완료: {success_count}/{len(audio_files)}개 | 총 소요: {format_duration(total_elapsed)}")

    # 키워드 빈도
    if args.keywords and all_texts:
        full_text = " ".join(all_texts)
        kw = extract_keywords(full_text, top_n=15, min_count=2)
        if kw:
            print(f"\n  📊 자주 등장한 단어:")
            print(f"     {format_keywords(kw)}")

    print()


if __name__ == "__main__":
    main()
