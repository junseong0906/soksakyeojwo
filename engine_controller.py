from __future__ import annotations

import threading
import traceback
from PySide6.QtCore import QObject, QThread, Signal
from app_paths import SettingsStore
from model_catalog import get_model
from model_service import ModelStore, RuntimeManager, Cancelled
from wlk_server_manager import WlkServerManager


class Operation(QThread):
    progress = Signal(int, str)
    result = Signal(object)
    failed = Signal(str)
    def __init__(self, action, parent=None):
        super().__init__(parent)
        self.action = action
        self.cancel = threading.Event()
    def run(self):
        try: self.result.emit(self.action(self.progress.emit, self.cancel))
        except Cancelled as exc: self.failed.emit(str(exc))
        except Exception as exc:
            self.failed.emit(f'{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}')


class EngineController(QObject):
    progress = Signal(int, str)
    changed = Signal()
    applied = Signal()
    failed = Signal(str)
    def __init__(self, parent=None, settings=None):
        super().__init__(parent)
        self.settings = settings or SettingsStore()
        self.models = ModelStore()
        self.runtimes = RuntimeManager()
        self.server = WlkServerManager()
        self.operation = None
        self.status = '설정에서 모델을 다운로드하고 적용해 주세요.'
        self.active_id = None
        self.token = ''  # Session only. Never persisted or printed.
        self.task_kind = ''
        self.task_model_id = None
        self.download_states = {}

    @property
    def busy(self): return self.operation is not None

    def run_task(self, action, on_success=None, *, kind='', model_id=None):
        if self.busy: return False
        self.task_kind = kind
        self.task_model_id = model_id
        operation = Operation(action, self)
        self.operation = operation
        operation.progress.connect(self._progress)
        operation.result.connect(lambda value: self._success(value, on_success))
        operation.failed.connect(self._failure)
        operation.finished.connect(self._finished)
        operation.start()
        self.changed.emit()
        return True

    def _progress(self, percent, message):
        if self.token: message = message.replace(self.token, '[token]')
        self.status = message
        if self.task_kind == 'download' and self.task_model_id:
            self.download_states[self.task_model_id] = {'state': '다운로드 중', 'percent': percent, 'message': message}
        self.progress.emit(percent, message)

    def _success(self, value, callback):
        if self.task_kind == 'download' and self.task_model_id:
            self.download_states[self.task_model_id] = {'state': '다운로드 완료', 'percent': 100, 'message': self.status}
        if callback: callback(value)

    def _failure(self, message):
        if self.token: message = message.replace(self.token, '[token]')
        if self.server.proc is None or self.server.proc.poll() is not None:
            self.active_id = None
        self.status = message
        if self.task_kind == 'download' and self.task_model_id:
            self.download_states[self.task_model_id] = {'state': '다운로드 실패' if 'Traceback' in message else '중지됨', 'percent': 0, 'message': message}
        from app_paths import data_dir
        log = data_dir() / 'logs' / 'model-operation.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        try:
            with log.open('a', encoding='utf-8') as handle:
                handle.write(message + '\n\n')
        except OSError: pass
        self.progress.emit(0, message)
        self.failed.emit(message)

    def _finished(self):
        operation = self.operation
        self.operation = None
        self.task_kind = ''
        self.task_model_id = None
        if operation: operation.deleteLater()
        self.changed.emit()

    def download(self, model_id, token=''):
        model = get_model(model_id)
        self.token = token.strip()
        return self.run_task(lambda progress, cancel: self.models.download(model, progress, cancel, self.token), kind='download', model_id=model.id)

    def prepare(self, model_id, diarization, device):
        model = get_model(model_id)
        return self.run_task(lambda progress, cancel: self.runtimes.prepare(model, diarization, device, progress, cancel))

    def apply(self, model_id, diarization, device, token=''):
        model = get_model(model_id)
        self.token = token.strip()
        if diarization and not (self.token or __import__('os').environ.get('HF_TOKEN')):
            self.failed.emit('화자 분리는 Hugging Face 토큰이 필요합니다. 설정의 안내 링크에서 모델 이용에 동의한 뒤 읽기 토큰을 입력해 주세요.')
            return False
        if not self.models.installed(model):
            self.failed.emit('선택한 모델을 먼저 다운로드해 주세요.')
            return False
        def action(progress, cancel):
            python = self.runtimes.prepare(model, diarization, device, progress, cancel)
            self.server.start(python, model, self.models.path(model), diarization, device, self.token, progress, cancel)
            return model.id
        def success(value):
            self.active_id = value
            self.settings.save(model_id=value, diarization=diarization, device=device)
            self.applied.emit()
        return self.run_task(action, success, kind='apply')

    def restart(self):
        def success(model_id):
            self.active_id = model_id
            self.applied.emit()
        return self.run_task(self.server.restart, success, kind='restart')

    def stop_task(self):
        if self.operation:
            self.operation.cancel.set()
            self._progress(-1, '중지 요청 중… 현재 네트워크 응답이나 설치 작업이 정리되면 종료됩니다.')

    def shutdown(self):
        self.stop_task()
        if self.operation: self.operation.wait(10000)
        self.server.stop_if_started()
