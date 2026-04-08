#!/bin/bash
# noma-scribe 실행기
# 이 파일을 더블클릭하면 앱이 실행됩니다.

cd "$(dirname "$0")"

# 가상환경이 없으면 설치 안내
if [ ! -d ".venv" ]; then
    echo ""
    echo "  ⚠️  먼저 설치를 실행해주세요:"
    echo "     cd $(pwd) && ./setup.sh"
    echo ""
    read -p "  Enter 키를 누르면 닫힙니다..."
    exit 1
fi

# 가상환경 활성화 후 앱 실행
source .venv/bin/activate

echo ""
echo "  🎙️  noma-scribe 시작 중..."
echo "  브라우저가 자동으로 열립니다."
echo "  종료하려면 이 창을 닫거나 Ctrl+C를 누르세요."
echo ""

python app.py
