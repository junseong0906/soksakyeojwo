from __future__ import annotations
from datetime import datetime
from multiprocessing import Queue
from queue import Empty
import threading
from pathlib import Path
import numpy as np
import sounddevice as sd
import soundfile as sf
import soxr
from app_paths import data_dir

def query_input_devices():
    devices = sd.query_devices()
    default = sd.default.device[0]
    return [{'index': i, 'name': str(d['name']), 'label': ('[기본] ' if i == default else '') + str(d['name']),
             'is_default': i == default, 'channels': 1, 'samplerate': int(d['default_samplerate'] or 16000)}
            for i, d in enumerate(devices) if d.get('max_input_channels', 0) > 0]

class Recorder:
    def __init__(self, events, asr_queue=None, folder=None):
        self.events, self.asr_queue = events, asr_queue
        self.folder = Path(folder) if folder else data_dir() / 'recordings'
        self.folder.mkdir(parents=True, exist_ok=True)
        self.stream = None
        self.file = None
        self.path = None
        self.device = None
        self.samplerate = 16000
        self.streaming = False
        self.resampler = None
        self.lock = threading.RLock()
        self.last_level = 0

    def _pcm(self, mono):
        if mono.size and self.asr_queue is not None:
            pcm = (np.clip(mono, -1, 1) * 32767).astype('<i2').tobytes()
            self.asr_queue.put({'type': 'audio_chunk', 'pcm_bytes': pcm})

    def callback(self, indata, frames, timing, status):
        import time
        with self.lock:
            mono = np.asarray(indata[:, 0] if indata.ndim > 1 else indata, dtype=np.float32)
            if not mono.size: return
            if status: self.events.put({'type': 'error', 'message': f'오디오 장치 상태: {status}'})
            if self.file is not None:
                try: self.file.write(mono)
                except Exception as exc:
                    self.stop_recording()
                    self.events.put({'type': 'error', 'message': f'녹음 저장 실패: {exc}'})
            if self.streaming:
                self._pcm(self.resampler.resample_chunk(mono) if self.resampler else mono)
            if time.monotonic() - self.last_level >= .1:
                self.last_level = time.monotonic()
                self.events.put({'type': 'level', 'value': min(100, int(float(np.sqrt(np.mean(mono * mono))) * 800))})

    def _close_input(self):
        # Never hold the callback lock while waiting for PortAudio callbacks to finish.
        if self.stream is not None:
            try: self.stream.stop()
            finally:
                self.stream.close()
                self.stream = None

    def _new_file(self):
        self.path = self.folder / ('lecture_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.wav')
        self.file = sf.SoundFile(str(self.path), mode='w', samplerate=self.samplerate, channels=1, subtype='PCM_16')

    def select_device(self, device):
        recording = self.file is not None
        self._close_input()
        with self.lock:
            if self.resampler and self.streaming:
                self._pcm(self.resampler.resample_chunk(np.empty(0, dtype=np.float32), last=True))
            if self.file:
                self.file.flush()
                self.file.close()
                self.file = None
            self.device = device
            try:
                info = sd.query_devices(device, 'input')
                self.samplerate = int(info.get('default_samplerate') or 16000)
                self.resampler = soxr.ResampleStream(self.samplerate, 16000, 1, dtype='float32') if self.samplerate != 16000 else None
                self.stream = sd.InputStream(device=device, channels=1, samplerate=self.samplerate,
                    blocksize=max(1, self.samplerate // 10), dtype='float32', callback=self.callback)
                if recording: self._new_file()
            except Exception:
                if self.file: self.file.close(); self.file = None
                self.events.put({'type': 'recording_stopped'})
                raise
        try: self.stream.start()
        except Exception:
            self._close_input()
            self.stop_recording()
            raise
        if recording: self.events.put({'type': 'recording_started', 'file': str(self.path)})

    def start_recording(self):
        if self.stream is None: self.select_device(self.device)
        with self.lock:
            if self.file is None:
                self._new_file()
                self.events.put({'type': 'recording_started', 'file': str(self.path)})

    def stop_recording(self):
        with self.lock:
            if self.file:
                self.file.flush()
                self.file.close()
                self.file = None
                self.events.put({'type': 'recording_stopped'})

    def enable_asr(self):
        if self.stream is None: self.select_device(self.device)
        with self.lock:
            self.resampler = soxr.ResampleStream(self.samplerate, 16000, 1, dtype='float32') if self.samplerate != 16000 else None
            self.streaming = True

    def stop_asr(self, shutdown=False, drain=True):
        with self.lock:
            if drain and self.streaming and self.resampler:
                self._pcm(self.resampler.resample_chunk(np.empty(0, dtype=np.float32), last=True))
            self.streaming = False
            # This producer's audio and EOF command preserve their ordering in multiprocessing.Queue.
            if drain and self.asr_queue is not None:
                self.asr_queue.put({'type': 'shutdown' if shutdown else 'pause_stream'})

    def close(self):
        self._close_input()
        self.stop_recording()
        self.stop_asr(shutdown=True)

def recorder_process_main(cmd_queue: Queue, event_queue: Queue, asr_cmd_queue=None):
    recorder = Recorder(event_queue, asr_cmd_queue)
    try:
        devices = query_input_devices()
        selected = next((d['index'] for d in devices if d['is_default']), devices[0]['index'] if devices else None)
        event_queue.put({'type': 'devices', 'devices': devices, 'selected': selected})
        if selected is not None: recorder.select_device(selected)
    except Exception as exc: event_queue.put({'type': 'error', 'message': f'마이크 초기화 실패: {exc}'})
    try:
        while True:
            try: message = cmd_queue.get(timeout=.1)
            except Empty: continue
            try:
                kind = message.get('type')
                if kind == 'select_device': recorder.select_device(message.get('device_index'))
                elif kind == 'start_recording': recorder.start_recording()
                elif kind == 'stop_recording': recorder.stop_recording()
                elif kind == 'set_asr_streaming':
                    if message.get('enabled'): recorder.enable_asr()
                    else: recorder.stop_asr()
                elif kind == 'stop_asr': recorder.stop_asr()
                elif kind == 'disable_asr': recorder.stop_asr(drain=False)
                elif kind == 'refresh_devices':
                    event_queue.put({'type': 'devices', 'devices': query_input_devices(), 'selected': recorder.device})
                elif kind == 'shutdown': return
            except Exception as exc:
                event_queue.put({'type': 'error', 'message': str(exc)})
    finally: recorder.close()
