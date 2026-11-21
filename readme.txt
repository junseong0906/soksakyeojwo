터미널 명령어(VSC에서 작업했음)
웹임

<설치>
python --version
python -m pip --version (버전 확인용)
python -m venv .venv (가상환경 만들기)
.\.venv\Scripts\Activate.ps1 (가상환경 실행)
    (여기서 안 되면,
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
    입력 후 다시 입력하면 될 것임
    성공했으면 터미널 앞에 (.venv)가 나타남)
python -m pip install --upgrade pip (최신으로 업뎃)
python -m pip install whisperlivekit (좀 오래걸림)
wlk --help (설치 확인)

<실행> 
wlk --model tiny --backend faster-whisper --lan ko --host 127.0.0.1 --port 8000 (모델 사용 및 서버 실행)
(브라우저에 http://127.0.0.1:8000/ 입력해서 접속)
(Ctrl+C를 터미널에 입력하면 서버 종료)

<가상환경 종료>
deactivate