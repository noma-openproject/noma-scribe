# 🎙️ noma-scribe

Apple Silicon 로컬 음성 전사 앱.

오디오 파일을 드래그앤드롭하고 버튼 하나 누르면 텍스트로 전사합니다.
mlx-whisper 기반, 완전 로컬, 무료.

## 요구사항

- macOS (Apple Silicon M1 이상)
- Python 3.11+
- ffmpeg

## 설치

```bash
chmod +x setup.sh
./setup.sh
```

## 실행

### 방법 1: 더블클릭 (추천)
`noma-scribe.command` 파일을 더블클릭하면 브라우저에서 앱이 열립니다.

### 방법 2: 터미널
```bash
source .venv/bin/activate
python app.py
```

### 방법 3: CLI (터미널에서 직접)
```bash
source .venv/bin/activate
python transcribe.py meeting.m4a --preset dev
```

## 프리셋

| 이름 | 설명 |
|------|------|
| `dev` | 개발회의 (API, 프레임워크, DevOps 용어) |
| `medical` | 의료/성형외과 (시술명, 해부학 용어) |
| `general` | 일반 (프롬프트 없음) |

`presets/` 폴더에 `.txt` 파일을 추가하면 커스텀 프리셋으로 자동 인식됩니다.

## 모델

기본: `mlx-community/whisper-large-v3-turbo` (~1.6GB, 첫 실행 시 자동 다운로드)
M5 32GB 기준 12분 오디오를 약 10~15초에 전사합니다.
