# 🎙️ noma-scribe

> Apple Silicon 로컬 음성 전사 앱 — 완전 로컬, 무료, 한국어 최적화.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![macOS](https://img.shields.io/badge/macOS-Apple%20Silicon-black?logo=apple&logoColor=white)](https://www.apple.com/mac/)
[![mlx-whisper](https://img.shields.io/badge/mlx--whisper-0.4+-FF6B35)](https://github.com/ml-explore/mlx-examples/tree/main/whisper)
[![Gradio](https://img.shields.io/badge/Gradio-6.0+-FFCD00?logo=gradio)](https://www.gradio.app/)
[![GitHub stars](https://img.shields.io/github/stars/noma-openproject/noma-scribe?style=social)](https://github.com/noma-openproject/noma-scribe)

오디오 파일을 1개 또는 여러 개 드래그앤드롭하면 텍스트로 전사합니다.
mlx-whisper 기반, 클라우드 없이 Mac에서 직접 돌아가며, 비용 없음.

**Repository:** https://github.com/noma-openproject/noma-scribe

---

## ✨ 기능

- 🍎 **Apple Silicon 최적화** — `mlx-whisper`로 M1/M2/M3/M4/M5에서 빠른 추론
- 📦 **단일/일괄 전사** — 파일 1개 또는 여러 개를 한 번에 처리
- 🗂️ **자동 패키징** — 1개는 `.txt`, 여러 개는 `.zip`으로 묶어 다운로드
- 🎯 **도메인 프리셋** — `presets/` 폴더의 `.txt`로 전문용어 인식률 향상
- 🇰🇷 **한국어 기본** — 영어/일본어/중국어/자동 감지도 지원
- ⏱️ **타임스탬프 옵션** — 세그먼트별 `[00:12 → 00:25]` 형태로 출력
- 🌐 **로컬 브라우저 UI** — Gradio 기반, 외부 전송 없음

---

## 🛠️ 요구사항

- macOS (Apple Silicon: M1 이상)
- Python **3.10 이상** (`brew install python@3.12` 권장)
- ffmpeg (`setup.sh`가 자동 설치)
- Homebrew (ffmpeg 설치용)

---

## 📥 설치

```bash
git clone https://github.com/noma-openproject/noma-scribe.git
cd noma-scribe
chmod +x setup.sh
./setup.sh
```

`setup.sh`가 자동으로:
1. ffmpeg 확인 후 없으면 `brew install ffmpeg`
2. 시스템에서 **Python 3.10+** 자동 탐지 (3.13 → 3.12 → 3.11 → 3.10 순)
3. `.venv/` 가상환경 생성
4. `mlx-whisper`, `gradio` 의존성 설치
5. `noma-scribe.command`에 실행 권한 부여

소요 시간: 첫 설치 약 2~5분 (네트워크에 따라 다름).

---

## 🚀 실행

### 방법 1: 더블클릭 (가장 쉬움)
Finder에서 **`noma-scribe.command`** 파일을 더블클릭하면 브라우저가 자동으로 열립니다.

> **macOS Gatekeeper 경고가 뜨면**: `noma-scribe.command`를 **우클릭 → 열기 → 열기**를 한 번만 해주세요. 이후엔 그냥 더블클릭으로 실행됩니다.

### 방법 2: 터미널 (브라우저 UI)
```bash
source .venv/bin/activate
python app.py
```
브라우저가 자동으로 http://127.0.0.1:7860 을 엽니다.

### 방법 3: CLI (스크립트/배치 자동화)
```bash
source .venv/bin/activate

# 단일 파일
python transcribe.py meeting.m4a

# 도메인 프리셋
python transcribe.py meeting.m4a --preset dev

# 폴더 일괄 전사 + 출력 디렉토리 지정
python transcribe.py ./recordings/ -o ./transcripts/

# 직접 프롬프트
python transcribe.py meeting.m4a --prompt "Next.js Supabase 리팩토링"

# 타임스탬프 포함
python transcribe.py meeting.m4a --timestamps

# 영어 전사
python transcribe.py meeting.m4a --lang en

# 프리셋 목록
python transcribe.py --list-presets
```

---

## 📂 일괄 전사 (브라우저 UI)

1. **여러 파일 선택**: 파일 영역에 오디오 파일을 한꺼번에 드래그앤드롭하거나 선택
2. **전사 시작**: 진행 상태가 `2/5 전사 중... (meeting.m4a)`처럼 표시됨
3. **결과 확인**: 각 파일의 결과가 `━━━ [n/N] 파일명 ━━━` 구분선으로 분리되어 표시됨
4. **다운로드**:
   - 파일 **1개** → 단일 `.txt` 다운로드
   - 파일 **여러 개** → `transcripts.zip`으로 묶어서 다운로드

---

## 🎯 프리셋

도메인 용어 힌트를 주면 Whisper의 전문용어 인식률이 크게 올라갑니다.

| 이름 | 설명 |
|---|---|
| `dev` | 개발회의 (프레임워크, DevOps, AI/클라우드 용어) |
| `medical` | 의료/성형외과 (진료, 시술, 해부학 용어) |
| `general` | 일반 (프롬프트 없음) |
| `meeting` | 일반 회의 (아젠다, 액션아이템, 스크럼 용어) |
| `interview` | 사용자/채용 인터뷰 (지원자, 역량, 페인포인트 용어) |
| `finance` | 금융/회계 (매출, 재무제표, 밸류에이션 용어) |

`presets/` 폴더에 새 `.txt` 파일을 추가하면 **자동으로 인식**됩니다.

```bash
# 예: 법률 도메인 프리셋 추가
echo "판례 소송 조항 계약 손해배상 의무 권리 갑 을 NDA MSA SOW" > presets/legal.txt
```

---

## 🤖 모델

- 기본: `mlx-community/whisper-large-v3-turbo` (~1.6GB)
- 첫 실행 시 HuggingFace에서 자동 다운로드 → `~/.cache/huggingface/`
- 캐시 후에는 오프라인에서도 동작

성능 (M5 32GB 기준): **12분 오디오 → 약 10~15초** 전사

다른 모델로 바꾸려면 `core/engine.py`의 `DEFAULT_MODEL` 또는 CLI에서 `--model`로 지정.

---

## 📁 프로젝트 구조

```
noma-scribe/
├── app.py                  # Gradio 브라우저 UI (단일/일괄 전사)
├── transcribe.py           # CLI 진입점
├── noma-scribe.command     # 더블클릭 실행기
├── setup.sh                # 설치 스크립트 (Python 3.10+ 자동 탐지)
├── requirements.txt        # mlx-whisper, gradio
├── core/
│   ├── engine.py           # mlx-whisper 래퍼
│   ├── presets.py          # 프리셋 로딩
│   └── utils.py            # 파일 탐지/포맷 헬퍼
└── presets/
    ├── dev.txt             # 개발 용어
    └── general.txt         # (빈 파일, 기본값)
```

---

## ❓ 트러블슈팅

**`Python 3.10 이상이 필요합니다`**
→ `brew install python@3.12` 후 `rm -rf .venv && ./setup.sh`

**`ffmpeg not found`**
→ Homebrew 설치: `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`, 그다음 `./setup.sh` 재실행

**모델 다운로드가 안 됨 / 너무 느림**
→ 첫 실행은 ~1.6GB 다운로드라 시간이 걸립니다. 캐시는 `~/.cache/huggingface/hub/` 에 저장됩니다.

**Gatekeeper가 `.command` 차단**
→ Finder에서 **우클릭 → 열기 → 열기**를 1회만 수행

**포트 7860이 이미 사용 중**
→ `app.py`의 `server_port=7860`을 다른 포트로 수정

---

## 🙏 크레딧

- [`mlx-whisper`](https://github.com/ml-explore/mlx-examples/tree/main/whisper) — Apple MLX 기반 Whisper
- [`gradio`](https://www.gradio.app/) — 브라우저 UI
- [`whisper-large-v3-turbo`](https://huggingface.co/openai/whisper-large-v3-turbo) (mlx-community 변환)
