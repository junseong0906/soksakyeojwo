from __future__ import annotations
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import time
import urllib.request
from app_paths import data_dir
from app_version import APP_VERSION
from local_engine import HEALTH_PATH, LOCAL_HOST, LocalEngineEndpoint
from model_service import check_cancel, hidden_flags
from external_process import external_popen

# WLK 0.2.28's LocalAgreement loader hardcodes device='auto'. Apply the user's
# device choice before its local import, without editing the installed package.
SERVER_BOOTSTRAP = '''
import os
import sys
import inspect
import importlib.util
from pathlib import Path
handles = []
if os.name == 'nt':
    import truststore
    truststore.inject_into_ssl()
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
if '--diarization' in sys.argv:
    import re
    import torch
    from collections import defaultdict
    from dataclasses import replace
    from typing import Any
    from torch.torch_version import TorchVersion
    from pyannote.audio.core.task import Specifications, Problem, Resolution
    # Embedded Python's inspect.stack scans modules without on-disk stdlib
    # sources. SpeechBrain 1.1's lazy __file__ lookup then recurses into inspect.
    try:
        from speechbrain.utils.importutils import LazyModule
    except ImportError:
        pass
    else:
        original_lazy_getattr = LazyModule.__getattr__
        def compatible_lazy_getattr(self, name):
            if name == '__file__':
                loaded = self.__dict__.get('lazy_module')
                if loaded is None: raise AttributeError(name)
                return getattr(loaded, name)
            return original_lazy_getattr(self, name)
        LazyModule.__getattr__ = compatible_lazy_getattr
    from pyannote.audio.core.model import Introspection
    from omegaconf import DictConfig, ListConfig
    from omegaconf.base import ContainerMetadata, Metadata
    from omegaconf.nodes import AnyNode
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    checkpoint_metadata = [TorchVersion, Specifications, Problem, Resolution, Introspection,
                           DictConfig, ListConfig, ContainerMetadata, Metadata, AnyNode,
                           EarlyStopping, ModelCheckpoint, defaultdict, int, list, dict, Any]
    diart_backend = importlib.import_module('whisperlivekit.diarization.diart_backend')
    OriginalDiart = diart_backend.DiartDiarization
    parameters = inspect.signature(OriginalDiart).parameters
    class CompatibleDiart(OriginalDiart):
        def __init__(self, *args, **kwargs):
            # WLK 0.2.28 core passes *_model; its Diart class expects *_model_name.
            for caller, constructor in (('segmentation_model', 'segmentation_model_name'),
                                        ('embedding_model', 'embedding_model_name')):
                if caller in kwargs and caller not in parameters and constructor in parameters:
                    value = kwargs.pop(caller)
                    if value is not None: kwargs.setdefault(constructor, value)
            # Official pyannote checkpoints contain these metadata types.
            # Keep Torch's restricted loader; scope the allowlist to this load.
            with torch.serialization.safe_globals(checkpoint_metadata):
                super().__init__(*args, **kwargs)
            self._session_started = False

        async def diarize(self):
            # Diart reports zero-based names like speaker0. WLK adds one when
            # aligning tokens, so supply a number without changing the timing.
            segments = []
            for segment in await super().diarize():
                if isinstance(segment.speaker, int):
                    segments.append(segment)
                    continue
                match = re.fullmatch(r'(?:speaker_?)?([0-9]+)', str(segment.speaker), re.IGNORECASE)
                number = int(match[1]) if match else -1
                segments.append(replace(segment, speaker=number))
            return segments

        def new_session(self):
            if not self._session_started:
                self._session_started = True
                return self
            # Cleanup closes the audio source. Reuse the loaded models with a
            # fresh Diart pipeline and observer for the next WebSocket session.
            return CompatibleDiart(config=self.pipeline.config,
                                   sample_rate=self.custom_source.sample_rate,
                                   block_duration=self.custom_source.block_duration)

    diart_backend.DiartDiarization = CompatibleDiart
    wlk_core = importlib.import_module('whisperlivekit.core')
    wlk_processor = importlib.import_module('whisperlivekit.audio_processor')
    original_diarization_factory = wlk_core.online_diarization_factory
    def new_diarization_session(args, backend):
        if isinstance(backend, CompatibleDiart):
            return backend.new_session()
        return original_diarization_factory(args, backend)
    wlk_core.online_diarization_factory = new_diarization_session
    wlk_processor.online_diarization_factory = new_diarization_session
basic_server = importlib.import_module('whisperlivekit.basic_server')
if health_path := os.environ.get('SOKSAK_HEALTH_PATH'):
    import hmac
    import json
    from fastapi import Request, HTTPException

    @basic_server.router.get(health_path, include_in_schema=False)
    async def managed_health(request: Request):
        expected = 'Bearer ' + os.environ['WLK_API_TOKEN']
        if not hmac.compare_digest(request.headers.get('authorization', '').encode(), expected.encode()):
            raise HTTPException(status_code=403)
        health = await basic_server.health(request)
        return {**json.loads(health.body), 'pid': os.getpid()}

basic_server.main()
'''


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class WlkServerManager:
    """Own the local engine and expose an endpoint only while its process is ready."""
    def __init__(self):
        self.proc = None
        self.log = None
        self._endpoint = None
        self.config = None

    @property
    def endpoint(self):
        proc = self.proc
        if proc is not None and proc.poll() is None:
            return self._endpoint
        return None

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

    def restart(self, progress, cancel):
        config = self.config
        if config is None:
            raise RuntimeError('복구할 로컬 엔진이 없습니다. 설정에서 모델을 적용해 주세요.')
        if self.log is not None and self.proc is not None:
            self.log.write(f'\nRestarting local engine (pid={self.proc.pid}, exit={self.proc.poll()})\n'.encode())
        self.stop_if_started()
        try:
            self._launch(config, progress, cancel)
        except Exception:
            self.stop_if_started()
            raise
        return config[1].id

    def _launch(self, config, progress, cancel):
        python, model, model_path, diarization, device, token = config
        check_cancel(cancel)
        with socket.socket() as listener:
            listener.bind((LOCAL_HOST, 0))
            endpoint = LocalEngineEndpoint(listener.getsockname()[1], secrets.token_urlsafe(32))
        logs = data_dir() / 'logs'
        logs.mkdir(exist_ok=True)
        path = logs / 'wlk_server.log'
        # Rotate logs without retaining authentication data in command arguments.
        if path.exists() and path.stat().st_size > 5_000_000:
            path.replace(logs / 'wlk_server.previous.log')
        self.log = path.open('ab', buffering=0)
        started_at = time.strftime('%Y-%m-%d %H:%M:%S')
        self.log.write(f'\n{started_at} app={APP_VERSION} model={model.id} device={device} diarization={diarization}\n'.encode())
        command = [str(python), '-c', SERVER_BOOTSTRAP,
                   '--host', LOCAL_HOST, '--port', str(endpoint.port),
                   '--session-idle-timeout', '0',
                   *model.arguments(Path(model_path), diarization, device)]
        env = {**os.environ, 'PYTHONUTF8': '1', 'PYTHONUNBUFFERED': '1',
               'HF_HOME': str(data_dir() / 'hf-cache'), 'HF_HUB_DISABLE_XET': '1',
               'PYANNOTE_CACHE': str(data_dir() / 'hf-cache/hub'),
               'SOKSAK_DEVICE': 'cuda' if device == 'cuda' else 'cpu',
               'SOKSAK_HEALTH_PATH': HEALTH_PATH, 'WLK_API_TOKEN': endpoint.access_key}
        if device != 'cuda': env['CUDA_VISIBLE_DEVICES'] = ''
        if token: env['HF_TOKEN'] = token
        self.proc = external_popen(command, stdin=subprocess.DEVNULL, stdout=self.log, stderr=subprocess.STDOUT,
                                     creationflags=hidden_flags(), env=env)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
        request = urllib.request.Request(endpoint.health_url, headers=endpoint.headers)
        started = time.monotonic()
        deadline = started + 600
        while time.monotonic() < deadline:
            check_cancel(cancel)
            if self.proc.poll() is not None:
                tail = path.read_text(encoding='utf-8', errors='replace')[-2400:]
                tail = tail.replace(endpoint.access_key, '[key]')
                if token: tail = tail.replace(token, '[token]')
                raise RuntimeError('엔진 시작에 실패했습니다. 설정의 실행환경 준비와 모델 권한을 확인해 주세요.\n' + tail)
            try:
                with opener.open(request, timeout=.5) as response:
                    health = json.load(response)
                if (isinstance(health, dict) and health.get('pid') == self.proc.pid
                        and health.get('ready') is True and health.get('backend') == model.backend):
                    self._endpoint = endpoint
                    progress(100, f'{model.name} 사용 중 · 한국어 · 화자 분리 ' + ('켜짐' if diarization else '꺼짐'))
                    return
            except (OSError, ValueError): pass
            elapsed = int(time.monotonic() - started)
            progress(-1, f'{model.name} 엔진을 불러오는 중 · {elapsed // 60:02}:{elapsed % 60:02} 경과\nCUDA 초기화·모델 로딩 중입니다. 상세 로그: wlk_server.log')
            cancel.wait(.5)
        raise RuntimeError('엔진 준비 제한 시간(10분)을 초과했습니다. data/logs/wlk_server.log를 확인해 주세요.')

    def stop_if_started(self):
        self._endpoint = None
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
