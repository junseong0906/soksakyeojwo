from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import zipfile

from app_paths import atomic_json, data_dir
from model_catalog import ModelSpec, WLK_VERSION
from runtime_install import InstallCancelled, run_installer

RUNTIME_REVISION = 2


class Cancelled(Exception):
    pass


def check_cancel(cancel):
    if cancel.is_set(): raise Cancelled('작업을 중지했습니다. 다시 시도하면 이어서 다운로드합니다.')


def hidden_flags():
    return subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0


def human_bytes(value):
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if value < 1000: return f'{value:.1f} {unit}'
        value /= 1000
    return f'{value:.1f} PB'


def wanted_file(name: str, family: str) -> bool:
    # Select one usable weight format; avoid duplicate original/PyTorch weights.
    if family == 'whisper':
        return name in ('model.bin', 'config.json', 'tokenizer.json', 'vocabulary.json', 'vocabulary.txt', 'preprocessor_config.json')
    leaf = name.rsplit('/', 1)[-1]
    if '/' in name or leaf.startswith(('consolidated', 'original')): return False
    return name.endswith(('.json', '.safetensors', '.model', '.tiktoken', '.txt', '.jinja'))


class ModelStore:
    def __init__(self, root: Path | None = None):
        self.root = root or data_dir() / 'models'
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, model): return self.root / model.id

    def installed(self, model):
        target = self.path(model)
        try:
            manifest = json.loads((target / 'complete.json').read_text(encoding='utf-8'))
            return manifest['repo'] == model.repo and bool(manifest['files']) and all(
                (target / name).is_file() and (target / name).stat().st_size == size
                for name, size in manifest['files'].items())
        except (OSError, ValueError, KeyError, TypeError): return False

    def size(self, model):
        try: return sum(json.loads((self.path(model) / 'complete.json').read_text(encoding='utf-8'))['files'].values())
        except (OSError, ValueError, KeyError): return None

    def download(self, model, progress, cancel, token=None):
        from huggingface_hub import HfApi, hf_hub_url
        import requests
        import hashlib
        progress(-1, '모델 파일 목록과 정확한 용량을 확인하는 중…')
        info = HfApi(token=token or None).model_info(model.repo, files_metadata=True)
        files = [f for f in info.siblings if wanted_file(f.rfilename, model.family)]
        if not any(f.rfilename.endswith(('.bin', '.safetensors')) for f in files):
            raise RuntimeError('사용 가능한 모델 가중치 파일을 찾지 못했습니다.')
        total = sum(f.size or 0 for f in files)
        if shutil.disk_usage(self.root).free < total * 1.1:
            raise RuntimeError(f'모델 저장 공간이 부족합니다. 최소 {human_bytes(total * 1.1)}가 필요합니다.')
        done = 0
        target = self.path(model)
        target.mkdir(parents=True, exist_ok=True)
        for item in files:
            check_cancel(cancel)
            size = item.size or 0
            destination = target / item.rfilename
            partial = target / (item.rfilename + '.part')
            if partial.exists() and partial.stat().st_size == size:
                partial.replace(destination)
            elif partial.exists() and partial.stat().st_size > size:
                partial.unlink()
            if not destination.exists() or destination.stat().st_size != size:
                for attempt in range(3):
                    check_cancel(cancel)
                    offset = partial.stat().st_size if partial.exists() else 0
                    headers = {'Authorization': f'Bearer {token}'} if token else {}
                    if offset: headers['Range'] = f'bytes={offset}-'
                    try:
                        url = hf_hub_url(model.repo, item.rfilename, revision=info.sha)
                        with requests.get(url, headers=headers, stream=True, timeout=(10, 30)) as response:
                            response.raise_for_status()
                            if response.status_code != 206: offset = 0
                            elif not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                                raise RuntimeError('서버가 잘못된 이어받기 범위를 반환했습니다.')
                            with partial.open('ab' if offset else 'wb') as handle:
                                for chunk in response.iter_content(1024 * 256):
                                    check_cancel(cancel)
                                    if not chunk: continue
                                    handle.write(chunk)
                                    offset += len(chunk)
                                    progress(int(min(99, 100 * (done + offset) / max(1, total))), f'{item.rfilename} · {human_bytes(done + offset)} / {human_bytes(total)}')
                        if partial.stat().st_size != size: raise IOError('다운로드 크기 불일치')
                        partial.replace(destination)
                        break
                    except requests.RequestException:
                        if attempt == 2: raise
                        cancel.wait(1)
            if destination.stat().st_size != size:
                raise RuntimeError(f'다운로드 크기 검증 실패: {item.rfilename}. 다시 시도해 주세요.')
            expected_hash = getattr(getattr(item, 'lfs', None), 'sha256', None)
            if expected_hash:
                digest = hashlib.sha256()
                with destination.open('rb') as handle:
                    while chunk := handle.read(1024 * 1024):
                        check_cancel(cancel)
                        digest.update(chunk)
                if digest.hexdigest() != expected_hash:
                    destination.unlink()
                    raise RuntimeError(f'모델 파일 검증 실패: {item.rfilename}. 다시 다운로드해 주세요.')
            done += size
            progress(int(100 * done / max(1, total)), f'{human_bytes(done)} / {human_bytes(total)}')
        check_cancel(cancel)
        atomic_json(target / 'complete.json', {'repo': model.repo, 'revision': info.sha, 'files': {f.rfilename: f.size for f in files}})
        progress(100, f'{model.name} 다운로드 완료 · 엔진 적용은 별도로 진행합니다.')
        return str(target)


class RuntimeManager:
    def __init__(self, root=None):
        self.root = root or data_dir() / 'runtimes'
        self.root.mkdir(parents=True, exist_ok=True)

    def profile(self, model, diarization, device='auto'):
        # Do not replace Torch DLLs in an environment serving an active model.
        accelerator = 'cuda' if device == 'cuda' else 'cpu'
        return self.root / (model.family + ('-diart' if diarization else '') + '-' + accelerator)

    def python(self, model, diarization, device='auto'):
        path = self.profile(model, diarization, device)
        return path / ('python.exe' if (path / 'python.exe').exists() else ('Scripts/python.exe' if os.name == 'nt' else 'bin/python'))

    def ready(self, model, diarization, device='auto'):
        try:
            stamp = json.loads((self.profile(model, diarization, device) / 'ready.json').read_text())
            accelerator = 'cuda' if device == 'cuda' else 'cpu'
            return stamp == {'wlk': WLK_VERSION, 'device': accelerator, 'diarization': diarization, 'family': model.family, 'runtime': RUNTIME_REVISION} and self.python(model, diarization, device).exists()
        except (OSError, ValueError): return False

    @staticmethod
    def run(command, progress, cancel, env=None, stage='실행환경 준비'):
        progress(-1, stage + ' · 시작 중…')
        try:
            run_installer(command, progress, cancel, data_dir() / 'logs/runtime-install.log', env=env, stage=stage)
        except InstallCancelled as exc:
            raise Cancelled(str(exc)) from exc

    def _ensure_pip(self, python, target, progress, cancel):
        # A cancelled get-pip leaves python.exe present but no usable pip.
        probe = 'import importlib.util; assert importlib.util.find_spec("pip") is not None, "pip missing"'
        try:
            self.run([str(python), '-c', probe], progress, cancel, stage='[1/5] Python 설치 상태 확인')
            return
        except Cancelled: raise
        except RuntimeError: pass
        get_pip = target / 'get-pip.py'
        with urllib.request.urlopen('https://bootstrap.pypa.io/get-pip.py', timeout=30) as response:
            get_pip.write_bytes(response.read())
        self.run([str(python), str(get_pip), '--no-input', '--no-warn-script-location'], progress, cancel, stage='[1/5] 패키지 설치 도구 복구')

    def _bootstrap(self, target, progress, cancel):
        # Official portable Python: no administrator access or system Python modification.
        if os.name != 'nt': raise RuntimeError('Python 3.12로 앱을 실행해 주세요.')
        target.mkdir(parents=True, exist_ok=True)
        archive = target / 'python.zip'
        progress(-1, '앱 전용 Python 3.12 실행환경 다운로드 중…')
        with urllib.request.urlopen('https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip', timeout=30) as response, archive.open('wb') as handle:
            while chunk := response.read(1024 * 512):
                check_cancel(cancel)
                handle.write(chunk)
        with zipfile.ZipFile(archive) as bundle: bundle.extractall(target)
        archive.unlink()
        (target / 'python312._pth').write_text('python312.zip\n.\nLib\nLib/site-packages\nimport site\n')

    def prepare(self, model, diarization, device, progress, cancel):
        if self.ready(model, diarization, device): return str(self.python(model, diarization, device))
        target = self.profile(model, diarization, device)
        target.mkdir(parents=True, exist_ok=True)
        stamp = target / 'ready.json'
        stamp.unlink(missing_ok=True)
        python = self.python(model, diarization, device)
        if not python.exists():
            source = os.environ.get('SOKSAK_PYTHON')
            if not source and not getattr(sys, 'frozen', False) and sys.version_info[:2] in ((3, 11), (3, 12)):
                source = sys.executable
            if source:
                self.run([source, '-m', 'venv', str(target)], progress, cancel, stage='[1/5] 앱 전용 Python 생성')
            else: self._bootstrap(target, progress, cancel)
        python = self.python(model, diarization, device)
        self._ensure_pip(python, target, progress, cancel)
        extras = []
        if model.family == 'qwen': extras.append('qwen3-streaming')
        if model.family == 'voxtral': extras.append('voxtral-hf')
        if diarization: extras.append('diarization-diart')
        package = 'whisperlivekit' + ('[' + ','.join(extras) + ']' if extras else '') + '==' + WLK_VERSION
        env = {**os.environ, 'PYTHONUTF8': '1', 'PYTHONUNBUFFERED': '1', 'PIP_DISABLE_PIP_VERSION_CHECK': '1',
               'PIP_NO_INPUT': '1', 'PIP_DEFAULT_TIMEOUT': '30', 'PIP_RETRIES': '3'}
        # Matching Torch/TorchAudio wheels are installed together. Diart requires <2.9.
        torch_version = '2.8.0'
        audio_version = torch_version
        index = 'https://download.pytorch.org/whl/cu128' if device == 'cuda' else 'https://download.pytorch.org/whl/cpu'
        progress(-1, '음성 인식 실행환경 준비 중 · 최초 설치에는 수 분이 걸릴 수 있습니다.')
        pip = [str(python), '-m', 'pip', 'install', '--no-input', '--no-warn-script-location', '--prefer-binary', '--progress-bar', 'raw']
        # Upgrade first: old pip versions do not support the raw progress stream.
        self.run([str(python), '-m', 'pip', 'install', '--no-input', '--upgrade', 'pip'], progress, cancel, env, '[1/5] 패키지 설치 도구 준비')
        torch_packages = [f'torch=={torch_version}', f'torchaudio=={audio_version}']
        if diarization: torch_packages.append('torchvision==0.23.0')
        self.run([*pip, *torch_packages, '--index-url', index], progress, cancel, env, '[2/5] ' + ('NVIDIA CUDA' if device == 'cuda' else 'CPU') + ' Torch 설치')
        requirements = [package, 'truststore==0.10.5']
        if model.family != 'voxtral': requirements += ['huggingface-hub==0.36.0']
        if model.family == 'qwen': requirements += ['qwen3-asr-causal==0.1.0', 'qwen-asr==0.0.6']
        if diarization: requirements += ['numpy<2', 'pyannote.audio<4', 'torchvision==0.23.0']
        self.run([*pip, *requirements], progress, cancel, env, '[3/5] 음성 인식 구성요소 설치')
        self.run([str(python), '-m', 'pip', 'check'], progress, cancel, env, '[4/5] 패키지 호환성 검사')
        probe = 'import whisperlivekit, torch; print("CUDA:", torch.cuda.is_available())'
        if device == 'cuda': probe += '; assert torch.cuda.is_available(), "NVIDIA GPU 또는 호환 드라이버를 찾지 못했습니다. CPU를 선택하거나 NVIDIA 드라이버를 확인해 주세요."'
        if model.family == 'qwen': probe += '; from whisperlivekit.qwen3_streaming import Qwen3StreamingASR'
        if model.family == 'voxtral': probe += '; from transformers import VoxtralRealtimeForConditionalGeneration'
        if diarization: probe += '; import diart'
        self.run([str(python), '-c', probe], progress, cancel, env, '[5/5] 모델 로더·실행 장치 검사')
        atomic_json(stamp, {'wlk': WLK_VERSION, 'device': 'cuda' if device == 'cuda' else 'cpu', 'diarization': diarization, 'family': model.family, 'runtime': RUNTIME_REVISION})
        progress(100, '실행환경 준비 완료 · 모델을 적용하면 엔진을 불러옵니다.')
        return str(python)
