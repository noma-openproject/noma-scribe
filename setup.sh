#!/bin/bash
# noma-scribe 설치 스크립트
# MacBook Pro M5 (Apple Silicon) 기준

set -e

echo ""
echo "  🎙️  noma-scribe 설치"
echo "  ─────────────────────"
echo ""

# 1. ffmpeg 확인 및 설치
if command -v ffmpeg &> /dev/null; then
    echo "  ✅ ffmpeg 이미 설치됨"
else
    echo "  📦 ffmpeg 설치 중..."
    if command -v brew &> /dev/null; then
        brew install ffmpeg
    else
        echo "  ❌ Homebrew가 필요합니다."
        echo "     설치: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        exit 1
    fi
fi

# 2. Python 3.10+ 찾기 (gradio 5.0+ 요구사항)
PYTHON=""
for v in 3.13 3.12 3.11 3.10; do
    if command -v python$v &> /dev/null; then
        PYTHON="python$v"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "  ❌ Python 3.10 이상이 필요합니다."
    echo "     설치: brew install python@3.12"
    exit 1
fi

echo "  ✅ Python 발견: $PYTHON ($($PYTHON --version))"

# 3. Python 가상환경 생성
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "  📦 Python 가상환경 생성 중..."
    $PYTHON -m venv "$VENV_DIR"
else
    # 기존 venv의 Python 버전 확인
    VENV_PY_VER=$("$VENV_DIR/bin/python" --version 2>&1 | awk '{print $2}')
    VENV_MAJOR_MINOR=$(echo "$VENV_PY_VER" | cut -d. -f1,2)
    REQUIRED=$(echo "3.10" | awk -F. '{print $1*100+$2}')
    CURRENT=$(echo "$VENV_MAJOR_MINOR" | awk -F. '{print $1*100+$2}')
    if [ "$CURRENT" -lt "$REQUIRED" ]; then
        echo "  ⚠️  기존 가상환경의 Python이 너무 낮습니다 ($VENV_PY_VER). 재생성합니다..."
        rm -rf "$VENV_DIR"
        $PYTHON -m venv "$VENV_DIR"
    else
        echo "  ✅ 가상환경 이미 존재 (Python $VENV_PY_VER)"
    fi
fi

# 3. 가상환경 활성화 및 의존성 설치
echo "  📦 의존성 설치 중..."
source "$VENV_DIR/bin/activate"
pip install --upgrade pip -q
pip install -r requirements.txt -q

# 4. 실행 파일 권한 설정
chmod +x noma-scribe.command 2>/dev/null

echo ""
echo "  ✅ 설치 완료!"
echo ""
echo "  실행 방법:"
echo ""
echo "    방법 1) 더블클릭: noma-scribe.command 파일을 더블클릭"
echo "    방법 2) 터미널:   source .venv/bin/activate && python app.py"
echo "    방법 3) CLI:      source .venv/bin/activate && python transcribe.py audio.m4a"
echo ""
echo "  첫 실행 시 Whisper 모델(~1.6GB)이 자동 다운로드됩니다."
echo ""
