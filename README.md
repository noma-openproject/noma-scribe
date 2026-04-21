# 🎙️ noma-scribe

> Apple Silicon 맥에서 완전 로컬로 돌아가는 음성 전사 앱.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![macOS](https://img.shields.io/badge/macOS-Apple%20Silicon-black?logo=apple&logoColor=white)](https://www.apple.com/mac/)
[![Gradio](https://img.shields.io/badge/Gradio-6.0+-FFCD00?logo=gradio)](https://www.gradio.app/)

오디오 파일을 1개 또는 여러 개 넣으면 텍스트로 전사합니다.  
클라우드 업로드 없이 Mac 안에서 직접 처리하고, 한국어 회의록 품질을 높이기 위한 후처리와 용어집 기능이 포함되어 있습니다.

**Repository:** [noma-openproject/noma-scribe](https://github.com/noma-openproject/noma-scribe)

---

## ✨ 기능

- Apple Silicon 최적화 전사: `whispermlx` 기반
- 단일 파일 / 폴더 / 다중 업로드 배치 전사
- 출력 포맷: 일반 텍스트, 타임스탬프 텍스트, SRT 자막
- 구간 전사: 시작/끝 시간 지정
- 2-pass 정밀 모드: 1차 전사 → 용어 추출 → 2차 전사
- 사용자 용어집: `~/.noma-scribe/glossary.json`
- 자동 용어 변형 감지: 전사 결과에서 용어집 후보 제안
- 한국어 문장 정리: KSS 기반 띄어쓰기/문단 정리
- 키워드 빈도 추출
- 단계별 진행률/소요 시간 표시
- 실행 로그 파일 저장: `~/.noma-scribe/logs/`
- 로컬 브라우저 UI + CLI 둘 다 지원

---

## 🛠️ 요구사항

- macOS (Apple Silicon: M1 이상)
- Python 3.10 이상
- `ffmpeg`
- Homebrew (`ffmpeg` 설치용)

---

## 📥 설치

```bash
git clone https://github.com/noma-openproject/noma-scribe.git
cd noma-scribe
chmod +x setup.sh
./setup.sh
```

`setup.sh`가 자동으로:

1. `ffmpeg` 확인 후 없으면 설치
2. `ffmpeg@7` 호환 라이브러리 연결 준비
3. `mecab-ko` / `python-mecab-ko` 설치
4. Python 3.10+ 탐지
5. `.venv/` 생성
6. 의존성 설치
7. `noma-scribe.command` 실행 권한 부여

Homebrew 경로는 `brew --prefix`로 자동 탐지하므로 `/opt/homebrew`가 아닌 환경도
하드코딩 없이 설치됩니다.

첫 실행 시 Whisper 모델이 다운로드될 수 있어서 네트워크 상태에 따라 몇 분 걸릴 수 있습니다.

---

## 🚀 실행

### 방법 1. 더블클릭

Finder에서 [noma-scribe.command](/Users/suyujeo/.codex/worktrees/3969/noma-scribe/noma-scribe.command) 를 더블클릭하면 브라우저 UI가 열립니다.

> Gatekeeper 경고가 뜨면 한 번만 `우클릭 → 열기`로 실행해 주세요.

### 방법 2. 브라우저 UI

```bash
source .venv/bin/activate
python app.py
```

기본 주소: `http://127.0.0.1:7860`

### 방법 3. CLI

```bash
source .venv/bin/activate

# 단일 파일
python transcribe.py meeting.m4a

# 정밀 모드 (2-pass)
python transcribe.py meeting.m4a --precise

# SRT 자막 저장
python transcribe.py meeting.m4a --format srt

# 타임스탬프 텍스트 저장
python transcribe.py meeting.m4a --timestamps

# 특정 구간만 전사
python transcribe.py meeting.m4a --start 01:00 --end 05:30

# 폴더 일괄 전사
python transcribe.py ./recordings/ -o ./transcripts/

# 영어 전사
python transcribe.py meeting.m4a --lang en

# 키워드 빈도 표시
python transcribe.py meeting.m4a --keywords

# 세그먼트 로그 출력
python transcribe.py meeting.m4a --verbose

# 디버그 모드 (상세 traceback + 로그)
python transcribe.py meeting.m4a --debug
```

---

## 🖥️ 브라우저 UI 흐름

1. 오디오 파일을 하나 이상 업로드
2. 필요하면 전사 모드, 출력 포맷, 구간, 고급 옵션 선택
3. 전사 시작
4. 결과 확인
5. 다운로드

배치 전사 시:

- 1개 파일이면 단일 결과 파일 다운로드
- 여러 파일이면 ZIP으로 묶어서 다운로드

고급 옵션:

- 용어집 적용
- 한국어 문장 정리
- 2-pass 용어 추출
- 디버그 모드

전사 후에는:

- 자주 등장한 단어 Top 15 확인
- 키워드를 바로 용어집에 추가
- 자동 감지된 용어 변형을 용어집 후보로 등록

---

## 📚 용어집

저장 위치:

```text
~/.noma-scribe/glossary.json
```

형식 예시:

```json
{
  "terms": [
    { "canonical": "Premo", "aliases": ["프레모", "프리모"] },
    { "canonical": "Noma", "aliases": ["노마"] }
  ]
}
```

용어집을 적용하면:

- alias 정확 매칭 치환
- 한국어 조사 보존
- RapidFuzz 기반 유사 치환

예:

- `프레모가` → `Premo가`
- `프리모를` → `Premo를`

---

## 🤖 모델

현재 코드 기준 모델 모드:

- `fast`: 기본 빠른 모드
- `precise`: 정밀 모드
- `korean`: 로컬 한국어 모델이 설치된 경우만 노출

기본 모델은 Hugging Face에서 자동 다운로드됩니다.  
`korean` 모드는 `models/ko-turbo` 가 있을 때만 앱에 표시됩니다.

---

## 📁 프로젝트 구조

```text
noma-scribe/
├── app.py                      # Gradio 앱 진입점
├── transcribe.py               # CLI 진입점
├── noma-scribe.command         # 더블클릭 실행기
├── setup.sh                    # 설치 스크립트
├── ui_glossary.py              # 용어집/클러스터 UI 헬퍼
├── core/
│   ├── engine.py               # whispermlx 래퍼, 모델/정렬/저장
│   ├── batch_transcription.py  # 배치 전사 orchestration
│   ├── postprocess.py          # 후처리 파이프라인
│   ├── glossary.py             # 사용자 용어집
│   ├── auto_glossary.py        # 용어 변형 자동 감지
│   ├── korean_normalizer.py    # 한국어 정리/KSS 연동
│   ├── keywords.py             # 키워드 추출
│   ├── two_pass.py             # 2-pass 정밀 전사
│   ├── srt.py                  # SRT 저장
│   └── utils.py                # 공용 유틸
└── tests/
    ├── test_batch_transcription.py
    ├── test_postprocess_full.py
    ├── test_two_pass.py
    └── ...
```

---

## ✅ 검증

```bash
source .venv/bin/activate
pytest -q
```

---

## ❓ 트러블슈팅

**Python 3.10 이상이 필요합니다**

```bash
brew install python@3.12
rm -rf .venv
./setup.sh
```

**`ffmpeg not found`**

Homebrew를 설치한 뒤 `./setup.sh`를 다시 실행하세요.

**`torchcodec` / `pyannote` 경고가 보입니다**

최신 `setup.sh`는 `ffmpeg@7` 호환 라이브러리와 KSS용 `mecab-ko`까지 함께 정리합니다.
기존 환경이라면 `./setup.sh`를 한 번 더 실행한 뒤 앱을 다시 시작하세요.

**모델 다운로드가 느립니다**

첫 실행은 모델 다운로드 때문에 시간이 걸릴 수 있습니다.
캐시 후에는 재사용됩니다.

**진행률이 오래 멈춘 것처럼 보입니다**

긴 한국어 회의록에서는 `한국어 문장 정리(KSS)` 단계가 추론보다 오래 걸릴 수 있습니다.
최신 코드에서는 단계별 진행률과 소요 시간을 상태창에 표시하고, 실행 로그를
`~/.noma-scribe/logs/` 에 남깁니다.

**KSS가 설치돼 있어도 정리가 비활성처럼 보입니다**

환경에 따라 `kss` 내부 초기화가 실패할 수 있습니다. 이 경우 앱은 graceful fallback으로 원문에 가까운 결과를 유지합니다.

**디버그가 필요합니다**

UI의 `디버그 모드`를 켜거나 CLI에서 `--debug`를 사용하면 내부 후처리 예외를
숨기지 않고 더 자세히 보여줍니다.

**포트 7860이 이미 사용 중입니다**

[app.py](/Users/suyujeo/.codex/worktrees/3969/noma-scribe/app.py) 의 `server_port` 값을 바꿔 실행하세요.

---

## 🙏 크레딧

- `whispermlx`
- [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper)
- [Gradio](https://www.gradio.app/)
