from __future__ import annotations
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.request
from app_paths import data_dir
from model_service import check_cancel, hidden_flags

# WLK 0.2.28's LocalAgreement loader hardcodes device='auto'. Apply the user's
# device choice before its local import, without editing the installed package.
SERVER_BOOTSTRAP = '''
import os
import importlib.util
from pathlib import Path
handles = []
import huggingface_hub
hub_download = huggingface_hub.hf_hub_download
def compatible_download(*args, **kwargs):
    # pyannote.audio 3.x still uses the old auth keyword; HF Hub 1/2 removed it.
    if 'use_auth_token' in kwargs:
        kwargs.setdefault('token', kwargs.pop('use_auth_token'))
    import inspect
    supported = inspect.signature(hub_download).parameters
    for legacy in ('resume_download', 'force_filename', 'local_dir_use_symlinks'):
        if legacy not in supported: kwargs.pop(legacy, None)
    return hub_download(*args, **kwargs)
huggingface_hub.hf_hub_download = compatible_download
if os.name == 'nt':
    torch_spec = importlib.util.find_spec('torch')
    if torch_spec:
        lib = Path(torch_spec.origin).parent / 'lib'
        if lib.exists():
            handles.append(os.add_dll_directory(str(lib)))
            os.environ['PATH'] = str(lib) + os.pathsep + os.environ.get('PATH', '')
import faster_whisper
original = faster_whisper.WhisperModel
class SelectedDeviceModel(original):
    def __init__(self, *args, **kwargs):
        device = os.environ.get('SOKSAK_DEVICE', 'cpu')
        kwargs['device'] = device
        kwargs['compute_type'] = 'float16' if device == 'cuda' else 'int8'
        super().__init__(*args, **kwargs)
faster_whisper.WhisperModel = SelectedDeviceModel
from whisperlivekit.basic_server import main
main()
'''

class WlkServerManager:
    """Owns only processes started by this app; readiness requires WLK /health."""
    def __init__(self):
        self.proc = None
        self.log = None
        self.port = None
        self.config = None

    @property
    def url(self):
        return f'ws://127.0.0.1:{self.port}/asr?mode=full'

    def start(self, python, model, model_path, diarization, device, token, progress, cancel):
        previous = self.config
        self.stop_if_started()
        config = (python, model, model_path, diarization, device, token)
        try:
            self._launch(config, progress, cancel)
            self.config = config
        except Exception:
            self.stop_if_started()
            if previous and not cancel.is_set():
                try:
                    progress(-1, '새 엔진 적용 실패 · 이전 모델을 복원하는 중…')
                    self._launch(previous, progress, cancel)
                    self.config = previous
                except Exception:
                    self.stop_if_started()
            raise

    def _launch(self, config, progress, cancel):
        python, model, model_path, diarization, device, token = config
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            self.port = listener.getsockname()[1]
        logs = data_dir() / 'logs'
        logs.mkdir(exist_ok=True)
        path = logs / 'wlk_server.log'
        # Rotate logs without retaining authentication data in command arguments.
        if path.exists() and path.stat().st_size > 5_000_000:
            path.replace(logs / 'wlk_server.previous.log')
        self.log = path.open('ab', buffering=0)
        command = [str(python), '-c', SERVER_BOOTSTRAP,
                   '--host', '127.0.0.1', '--port', str(self.port), *model.arguments(Path(model_path), diarization, device)]
        env = {**os.environ, 'PYTHONUTF8': '1', 'PYTHONUNBUFFERED': '1',
               'HF_HOME': str(data_dir() / 'hf-cache'), 'HF_HUB_DISABLE_XET': '1',
               'SOKSAK_DEVICE': 'cuda' if device == 'cuda' else 'cpu'}
        if device != 'cuda': env['CUDA_VISIBLE_DEVICES'] = ''
        if token: env['HF_TOKEN'] = token
        self.proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=self.log, stderr=subprocess.STDOUT,
                                     creationflags=hidden_flags(), env=env)
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            check_cancel(cancel)
            if self.proc.poll() is not None:
                tail = path.read_text(encoding='utf-8', errors='replace')[-2400:]
                if token: tail = tail.replace(token, '[token]')
                raise RuntimeError('엔진 시작에 실패했습니다. 설정의 실행환경 준비와 모델 권한을 확인해 주세요.\n' + tail)
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{self.port}/health', timeout=.5) as response:
                    health = json.load(response)
                if health.get('ready') is True and health.get('backend') == model.backend:
                    progress(100, f'{model.name} 사용 중 · 한국어 · 화자 분리 ' + ('켜짐' if diarization else '꺼짐'))
                    return
            except (OSError, ValueError): pass
            progress(-1, f'{model.name} 엔진을 불러오는 중… 최초 화자 모델 준비에는 시간이 걸릴 수 있습니다.')
            cancel.wait(.5)
        raise RuntimeError('엔진 준비 제한 시간(10분)을 초과했습니다. data/logs/wlk_server.log를 확인해 주세요.')

    def stop_if_started(self):
        if self.proc is not None:
            if self.proc.poll() is None:
                self.proc.terminate()
                try: self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait(timeout=3)
            self.proc = None
        if self.log:
            self.log.close()
            self.log = None
