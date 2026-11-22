속삭여줘 v1.2.2 — 실시간 강의 자막 / 화자 분리 호환 개선판
작성일: 2026-11-23
원본: https://github.com/junseong0906/soksakyeojwo
기준 커밋: 2bb34ccf45e657723ced8b9aff9dfb39d69c0b61
WhisperLiveKit 버전: 0.2.28

실행 파일로 시작하기
1. Windows 배포 ZIP을 모두 풀어 주세요. exe와 _internal 폴더를 함께 유지합니다.
2. 속삭여줘.exe를 실행합니다.
3. “설정 · 모델 다운로드”에서 원하는 모델 행의 “다운로드” 버튼을 누릅니다.
4. 모델마다 별도의 상태와 진행률을 표시합니다. 적용할 모델은 왼쪽 체크박스로 하나만 선택합니다.
5. 실행 장치를 선택합니다. 처음에는 “자동 (CPU 우선)” 또는 CPU를 권장합니다.
6. “선택한 모델 적용”을 누릅니다. 필요한 앱 전용 환경을 준비하고 모델을 불러옵니다.
   “실행환경 준비”로 환경 설치만 먼저 진행할 수도 있습니다.
7. 메인 화면에 모델 이름이 표시되고 엔진 준비가 끝나면 “실시간 자막 시작”을 누릅니다.
8. WAV 녹음은 “녹음 시작”으로 별도로 켭니다.

첫 설치에는 인터넷과 충분한 디스크 공간이 필요합니다.
Python이 없는 PC에서는 공식 Python 3.12 임베디드 런타임을 앱 폴더 아래에 준비합니다.
시스템 Python, 전역 패키지, 시스템 PATH를 수정하지 않습니다.
실행환경은 모델 파일 이외에 Torch 및 기타 패키지 공간을 별도로 사용합니다.
v1.2.2에서 기존 실행환경을 처음 적용하면 인증서 구성요소를 추가로 확인합니다. 이미 설치된 CUDA/Torch 패키지는 재사용합니다.
CUDA Torch 파일은 약 3.46 GB이며, 다운로드 후 압축 해제와 설치에 추가 시간이 필요합니다.
설정 화면에서 [1/5]~[5/5] 단계, 단계별 경과 시간, 준비 중인 파일의 용량을 표시합니다.
실행환경 설치 로그는 data/logs/runtime-install.log에 기록됩니다.
출력이 없어도 CPU·디스크 활동이 있으면 설치를 계속 기다립니다.
출력과 작업 활동이 모두 10분간 없거나 한 설치 단계가 45분을 넘으면 종료하고 로그를 안내합니다.
중지 후 다시 적용하면 완료된 패키지와 캐시를 재사용하며, 누락된 pip도 복구합니다.
다운로드 취소 후 다시 누르면 .part 파일에서 이어받습니다.
네트워크 요청 중에는 중지 완료까지 최대 수십 초 걸릴 수 있습니다.
모델 가중치 파일은 다운로드 크기 및 제공되는 LFS SHA-256으로 확인합니다.
파일 검증이 끝나야 “다운로드됨” 상태가 됩니다.
다운로드 완료와 엔진 준비 완료는 별도 상태입니다.

지원 모델
- Whisper Tiny
- Whisper Small
- Whisper Medium
- Whisper Large-v3
- Whisper Large-v3 Turbo
- Qwen3-ASR-0.6B
- Voxtral Mini 4B Realtime

모델별 실행
- Whisper: faster-whisper / LocalAgreement / CTranslate2 모델
- Qwen: qwen3-streaming / 한국어 지원 windowed 모드
  영어 전용 causal 체크포인트로 바꾸지 않습니다.
- Voxtral: voxtral / Hugging Face Transformers
- 각 계열과 화자 분리 사용 여부에 따라 별도 환경을 만듭니다.
- 기본 자동 장치는 호환성을 우선해 CPU를 사용합니다.
- NVIDIA GPU를 쓰려면 설정에서 CUDA를 선택합니다.
  CUDA 프로필은 Torch 2.8 / CUDA 12.8 휠을 준비하고 GPU 사용 가능 여부를 검사합니다.
  호환 NVIDIA 드라이버와 충분한 메모리가 필요합니다.
- 큰 모델은 PC 성능에 따라 실시간보다 느릴 수 있습니다.
- 표의 초기 용량은 참고치이며 다운로드 후 실제 선택한 파일들의 크기를 표시합니다.

화자 분리
1. 설정에서 “화자 분리 켜기”를 선택합니다.
2. Hugging Face 계정으로 아래 모델 페이지에서 이용 조건에 동의합니다.
   https://huggingface.co/pyannote/segmentation-3.0
   https://huggingface.co/pyannote/embedding
3. 읽기 토큰을 발급받아 설정의 비밀번호 형태 입력란에 입력합니다.
   https://huggingface.co/settings/tokens
4. “실행환경 준비” 또는 “선택한 모델 적용”으로 적용합니다.

Windows 및 Voxtral/Transformers 5 환경과의 호환을 위해 WLK의 Diart 백엔드를 사용합니다.
화자 분리용 환경은 Python 3.12, Torch 2.8, NumPy 1.x 및 pyannote.audio 3.x를 사용합니다.
토큰은 설정 JSON에 저장하지 않으며 이번 앱 실행에서만 사용합니다.
HF_TOKEN 환경 변수가 있다면 사용할 수 있습니다.
화자 모델 자체는 서버 최초 실행 때 추가로 내려받습니다.
화자 이름 설정에는 1 또는 spk_1을 입력할 수 있습니다.
화자 ID는 연결 단위의 익명 번호입니다. 다시 연결하거나 모델을 전환하면 같은 번호가 다른 사람을 가리킬 수 있으므로 표시 이름을 확인해 주세요.
이름을 지정하는 기능은 목소리로 사람의 신원을 자동 확인하는 기능과 다릅니다.

수정된 처리
- v1.2.2: WLK 0.2.28과 Diart 사이의 화자 모델 인자 이름 불일치를 보정했습니다.
  Torch 2.8의 제한된 체크포인트 로딩을 유지하며 공식 pyannote 모델의 메타데이터 타입을 지원합니다.
  임베디드 Python에서 SpeechBrain 지연 모듈 검사 중 발생하는 재귀 오류를 보정했습니다.
  화자 모델 캐시를 앱의 data/hf-cache 아래에 보관합니다.
  외부 음성 인식 실행환경에도 Windows 인증서 저장소를 적용합니다.
- v1.2.1: CUDA 설치 단계·용량·경과 시간과 상세 설치 로그를 추가했습니다.
  응답 없는 설치의 무한 대기를 막고, 앱에 포함된 DLL 경로가 외부 Python 설치를 방해하지 않도록 분리했습니다.
- v1.2.0: 모델마다 다운로드 버튼/진행률을 표시하고 적용 모델을 체크박스로 선택합니다.
- v1.2.0: 표준 Python으로 배포본을 만들고 일치하는 OpenSSL 라이브러리를 포함합니다.
  Windows 인증서 저장소를 사용하며 TLS 인증서 검증은 계속 수행합니다.
  구버전에서 다운로드 전 발생한 Illegal byte sequence SSL 오류를 수정했습니다.
- 화자 분리 설정을 실제 서버 인자에 반영합니다.
- 마이크를 바꾸면 기존 녹음을 확정하고 새 장치의 샘플레이트로 새 WAV를 만듭니다.
  전환 순간의 물리적 입력 공백은 있을 수 있지만 새 녹음 파일을 실수로 닫지 않습니다.
- 새 녹음 파일 이름에 마이크로초를 포함해 같은 초의 파일 덮어쓰기를 방지합니다.
- 임시 자막과 확정 내용/기록을 분리합니다.
- WLK mode=full 스냅샷은 기존 줄 갱신으로 반영하며 화자 수정과 버퍼도 표시합니다.
- 이전 스냅샷에서 서버가 정리한 오래된 구간은 기록에 보존합니다.
- 마이크 음성을 16 kHz PCM으로 직접 전송합니다. 관리형 서버 연결에는 FFmpeg가 필요 없습니다.
- 정지 시 녹음 워커의 오디오 순서를 보존하고 빈 바이너리 EOF를 전송합니다.
- 마지막 자막과 ready_to_stop을 받은 뒤 연결을 닫습니다.
  서버가 응답하지 않는 경우 45초 후 경고하며 종료합니다.
- 모델 전환 전 기존 자막을 마무리합니다. 로컬 녹음은 계속 유지합니다.
- 새 엔진 시작에 실패하면 이전 엔진 복원을 시도합니다.
- 자막 모델을 준비하거나 내려받는 동안 UI는 계속 응답합니다.
- 모델/화자 분리/장치 설정을 저장하고 설치가 완전한 기존 모델만 다음 실행에서 복원합니다.
- 공통 SpeakerLabeler 하나를 UI와 테스트가 함께 사용합니다.

파일 위치
기본 위치는 실행 파일 또는 소스 폴더 안의 data입니다.
- data/models: ASR 모델과 완료 정보, 다운로드 중 .part 파일
- data/runtimes: 계열별 Python 실행환경
- data/hf-cache: 화자 모델 등 보조 모델 캐시
- data/recordings: WAV 녹음
- data/logs/wlk_server.log: 엔진 실행 로그
- data/settings.json: 선택 모델 및 화면의 화자 이름 설정
- data/last_session.json: 정상 종료 시 마지막 자막 기록과 북마크
SOKSAK_DATA_DIR 환경 변수로 저장 위치를 지정할 수 있습니다.
북마크 시각은 현재 시계 기준이고 자막 기록의 시각은 각 연결 내 오디오 시각입니다.
현재 last_session.json은 저장용이며 이전 세션을 자동으로 불러오는 UI는 포함하지 않습니다.

소스로 실행/테스트/빌드
Python 3.12에서 다음 명령을 사용합니다.
  python -m venv .venv
  .venv\Scripts\python.exe -m pip install -r requirements-dev.txt
  .venv\Scripts\python.exe app.py
  .venv\Scripts\python.exe -m pytest tests -q
  .venv\Scripts\python.exe -m PyInstaller --noconfirm build.spec

실행.cmd 또는 run.ps1로 UI 의존성 설치 후 실행할 수도 있습니다.
app.py --demo는 마이크나 서버 없이 자막 화면을 보여줍니다.
SOKSAK_PYTHON 환경 변수로 환경 생성에 사용할 Python 3.12 경로를 지정할 수 있습니다.
외부 WLK에 연결할 경우 --pcm-input과 mode=full 응답이 필요합니다.
이 앱은 localhost의 임의 포트에 직접 실행한 서버만 종료합니다.

검증 범위
- 파서, 줄 갱신, 버퍼, 기록 누락, 화자 이름 정규화, 마이크 전환, 녹음 유지, 다운로드 취소·이어받기, 실제 로컬 WebSocket 종료 순서를 회귀 테스트합니다.
- Whisper Tiny의 실제 모델 다운로드, CPU 환경 준비, 저장소의 약 6초 음성 전사와 정상 EOF 종료를 검증했습니다.
- 공식 Windows 임베디드 Python의 다운로드·pip 부트스트랩을 검증했습니다.
- Qwen/Voxtral 및 화자 분리 환경의 설치/로더 가져오기를 별도로 확인합니다.
- 모든 대용량 모델의 실시간 추론, CUDA 추론, 강의실 다중 화자 정확도는 이 환경에서 측정하지 않았습니다.
  화자 분리의 실제 실행에는 사용자 계정의 모델 접근 동의와 토큰이 필요합니다.

관련 문서
https://github.com/QuentinFuxa/WhisperLiveKit
https://github.com/QwenLM/Qwen3-ASR
https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602
