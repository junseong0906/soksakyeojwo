from __future__ import annotations
from dataclasses import dataclass
import json
import logging
from logging.handlers import RotatingFileHandler
from multiprocessing import Queue
from queue import Empty
import socket
import threading
import time
import websocket
from app_paths import data_dir
from local_engine import LOCAL_HOST, LocalEngineEndpoint
from runtime_install import redact
from speaker_labeler import normalize_speaker_id


logger = logging.getLogger(__name__)
SOCKET_ERRORS = (OSError, websocket.WebSocketException)
RECONNECT_DELAY = .6

def _time_to_ms(value):
    if value is None or value == '': return None
    try:
        if isinstance(value, (int, float)): return round(value * 1000)
        return round(sum(float(p) * 60 ** i for i, p in enumerate(reversed(str(value).split(':')))) * 1000)
    except (ValueError, TypeError): return None

def _line_event(item, is_final=True):
    text = str(item.get('text') or '').strip()
    if not text or str(item.get('speaker', '')) == '-2': return None
    return {'type': 'subtitle', 'text': text, 'is_final': is_final,
            'speaker_id': normalize_speaker_id(item.get('speaker', item.get('speaker_id'))),
            'start_ms': _time_to_ms(item.get('start', item.get('start_time'))),
            'end_ms': _time_to_ms(item.get('end', item.get('end_time')))}

def buffer_text(value):
    return str(value.get('text') or '') if isinstance(value, dict) else str(value or '')


def _server_error(data):
    if isinstance(data, dict) and (data.get('status') == 'error' or str(data.get('type', '')).lower() == 'error'):
        return str(data.get('error') or data.get('message') or '서버 처리 오류')
    return None


def _parse_server_message(raw_msg):
    text = raw_msg.strip()
    if not text: return []
    try: data = json.loads(text)
    except ValueError:
        if text.lower() in ('ping', 'pong', 'heartbeat'): return [{'type': 'heartbeat'}]
        return [_line_event({'text': text})]
    if isinstance(data, dict):
        kind = str(data.get('type', '')).lower()
        if kind in ('config', 'ready_to_stop', 'heartbeat', 'ping', 'pong'):
            return [{**data, 'type': 'heartbeat' if kind in ('ping', 'pong') else kind}]
        if error := _server_error(data):
            return [{'type': 'asr_error', 'message': error}]
        if isinstance(data.get('lines'), list):
            return [event for line in data['lines'] if isinstance(line, dict)
                    if (event := _line_event(line))]
        transcript = data.get('text') or data.get('transcript') or buffer_text(data.get('buffer_transcription'))
        event = _line_event({'text': transcript}, bool(data.get('is_final', data.get('final', False))))
        return [event] if event else []
    if isinstance(data, list):
        return [event for line in data if isinstance(line, dict)
                if (event := _line_event(line, bool(line.get('is_final', False))))]
    return []

class WlkMessageProcessor:
    """Forward full-state snapshots as updates, never as append-only final sentences."""
    def __init__(self): self.reset()
    def reset(self): self.signature = None
    def process(self, raw_msg):
        try: data = json.loads(raw_msg)
        except ValueError: return _parse_server_message(raw_msg)
        # WLK includes empty lines and buffers even in an error response.
        if error := _server_error(data):
            return [{'type': 'asr_error', 'message': error}]
        if isinstance(data, dict) and ('lines' in data or 'buffer_transcription' in data or 'buffer_diarization' in data):
            lines = [event for item in (data.get('lines') or []) if isinstance(item, dict)
                     if (event := _line_event(item))]
            event = {'type': 'transcript_snapshot', 'lines': lines,
                     'partial': buffer_text(data.get('buffer_transcription')),
                     'diarization_buffer': buffer_text(data.get('buffer_diarization'))}
            signature = json.dumps(event, ensure_ascii=False, sort_keys=True)
            if signature == self.signature: return []
            self.signature = signature
            return [event]
        return _parse_server_message(raw_msg)

@dataclass(frozen=True)
class StreamFailure:
    message: str
    retryable: bool = False
    event_type: str = 'asr_error'


class PcmSession:
    """Own one WebSocket and record its terminal result exactly once."""

    def __init__(self, events, factory=None):
        self.events = events
        self.factory = factory or websocket.WebSocket
        self.ws = None
        self.ready = threading.Event()
        self.receiver = None
        self.processor = WlkMessageProcessor()
        self.done = threading.Event()
        self.draining = threading.Event()
        self.failure = None
        self._result_lock = threading.Lock()

    def open(self, endpoint: LocalEngineEndpoint, *, reconnecting=False):
        self.ws = self.factory(enable_multithread=True)
        direct = None
        try:
            timeout = 3 if reconnecting else 10
            self.ws.settimeout(timeout)
            # A preconnected socket bypasses proxy discovery in websocket-client.
            direct = socket.create_connection((LOCAL_HOST, endpoint.port), timeout=timeout)
            direct.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.ws.connect(endpoint.asr_url, socket=direct, header=endpoint.headers, redirect_limit=0)
            if self.ws.getstatus() != 101:
                raise RuntimeError('로컬 엔진이 WebSocket 연결을 승인하지 않았습니다.')
            raw_config = self.ws.recv()
            if not raw_config:
                raise websocket.WebSocketConnectionClosedException('로컬 엔진이 연결을 종료했습니다.')
            config = json.loads(raw_config)
            if error := _server_error(config):
                raise RuntimeError(error)
            if not isinstance(config, dict) or config.get('type') != 'config' or not config.get('useAudioWorklet'):
                raise RuntimeError('로컬 엔진이 PCM 입력을 지원하지 않습니다. 설정에서 모델을 다시 적용해 주세요.')
        except Exception:
            self.disconnect()
            if direct is not None:
                direct.close()
            raise
        self.ws.settimeout(3)
        logger.info('Subtitle stream %s', 'reconnected' if reconnecting else 'opened')
        self.events.put({'type': 'stream_reopened' if reconnecting else 'stream_opened'})
        self.receiver = threading.Thread(target=self._receive, args=(self.ws,), daemon=True)
        self.receiver.start()

    def _finish(self, failure=None):
        with self._result_lock:
            if self.done.is_set():
                return
            self.failure = failure
            if failure is None:
                self.events.put({'type': 'transcript_finalized'})
                self.ready.set()
            else:
                logger.warning('%s', redact(failure.message))
            self.done.set()

    def _receive(self, ws):
        try:
            while not self.done.is_set():
                try:
                    opcode, raw = ws.recv_data()
                except websocket.WebSocketTimeoutException:
                    continue
                if opcode == websocket.ABNF.OPCODE_CLOSE:
                    code = int.from_bytes(raw[:2], 'big') if len(raw) >= 2 else None
                    reason = raw[2:].decode('utf-8', errors='replace')
                    details = f'{code} {reason}'.strip() if code else '종료 코드 없음'
                    self._finish(StreamFailure(f'자막 서버 연결이 일찍 종료되었습니다. ({details})',
                                               retryable=code in (None, 1000, 1001, 1006, 1011, 1012, 1013)))
                    return
                if self.done.is_set():
                    return
                if isinstance(raw, bytes):
                    raw = raw.decode('utf-8')
                for event in self.processor.process(raw):
                    if event['type'] == 'ready_to_stop':
                        if self.draining.is_set():
                            self._finish()
                        else:
                            self._finish(StreamFailure('로컬 엔진이 자막 처리를 일찍 끝냈습니다.', retryable=True))
                        return
                    if event['type'] == 'asr_error':
                        self._finish(StreamFailure(event['message']))
                        return
                    self.events.put(event)
        except SOCKET_ERRORS as exc:
            self._finish(StreamFailure(f'자막 수신 연결이 끊겼습니다. ({exc})', retryable=True))
        except Exception as exc:
            self._finish(StreamFailure(f'자막 수신 처리 오류: {exc}'))

    def send(self, pcm):
        if pcm:
            self._send(pcm)

    def _send(self, payload):
        if self.ws is None or self.done.is_set() or not self.ws.connected:
            return
        try:
            self.ws.send_binary(payload)
        except SOCKET_ERRORS as exc:
            self._finish(StreamFailure(f'자막 음성 전송 연결이 끊겼습니다. ({exc})', retryable=True))

    def close(self, timeout=45):
        if self.ws is None:
            return
        if not self.done.is_set():
            self.draining.set()
            self.events.put({'type': 'stream_draining'})
            # The recorder enqueues all earlier PCM before its stop command.
            self._send(b'')
            if not self.done.wait(timeout):
                warning = '마지막 자막 처리 완료를 확인하지 못했습니다. 로컬 녹음 파일을 확인해 주세요.'
                self._finish(StreamFailure(warning, event_type='asr_warning'))
        self.report_closed()

    def disconnect(self):
        with self._result_lock:
            self.done.set()
        ws, self.ws = self.ws, None
        if ws is not None:
            try:
                if self.ready.is_set():
                    ws.close(timeout=.5)
                else:
                    ws.shutdown()
            except SOCKET_ERRORS:
                logger.debug('Socket already closed.', exc_info=True)
        if self.receiver is not None:
            self.receiver.join(timeout=1)

    def report_closed(self, *, allow_recovery=False):
        self.disconnect()
        if self.failure:
            self.events.put({'type': self.failure.event_type, 'message': redact(self.failure.message)})
        self.events.put({'type': 'stream_closed', 'retryable': bool(allow_recovery and self.failure and self.failure.retryable)})


def _open_session(events, endpoint, reconnecting=False):
    if type(endpoint) is not LocalEngineEndpoint:
        _report_open_failure(events, '앱에서 시작한 로컬 엔진에만 연결할 수 있습니다.')
        return None
    session = PcmSession(events)
    try:
        session.open(endpoint, reconnecting=reconnecting)
    except websocket.WebSocketBadStatusException as exc:
        _report_open_failure(events, f'로컬 엔진이 연결을 거부했습니다. (HTTP {exc.status_code})',
                             retryable=exc.status_code in (429, 503))
    except SOCKET_ERRORS as exc:
        _report_open_failure(events, f'로컬 엔진 연결에 실패했습니다. ({exc})', retryable=True)
    except (ValueError, RuntimeError) as exc:
        _report_open_failure(events, f'로컬 엔진 연결에 실패했습니다. ({exc})')
    else:
        return session
    return None


def _report_open_failure(events, message, *, retryable=False):
    message = redact(message)
    logger.warning('%s', message)
    events.put({'type': 'asr_error', 'message': message})
    events.put({'type': 'stream_closed', 'retryable': retryable})


def _asr_log_handler():
    try:
        file = data_dir() / 'logs' / 'asr-client.log'
        file.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(file, maxBytes=500_000, backupCount=1, encoding='utf-8')
    except OSError:
        return None
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    return handler


def asr_client_process_main(cmd_queue: Queue, event_queue: Queue):
    handler = _asr_log_handler()
    previous_level = logger.level
    if handler is not None:
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    try:
        _run_client(cmd_queue, event_queue)
    finally:
        if handler is not None:
            logger.removeHandler(handler)
            logger.setLevel(previous_level)
            handler.close()


def _run_client(cmd_queue, event_queue):
    session = None
    endpoint = None
    can_retry = False
    reconnect_at = None
    try:
        while True:
            try:
                message = cmd_queue.get(timeout=.1)
            except Empty:
                message = {}
            kind = message.get('type')

            # An explicit stop takes priority over a pending reconnect.
            if kind in ('pause_stream', 'shutdown'):
                if session:
                    session.close()
                    session = None
                elif reconnect_at is not None:
                    event_queue.put({'type': 'stream_closed', 'retryable': False})
                reconnect_at = None
                if kind == 'shutdown':
                    return
                continue

            if session and session.done.is_set():
                if session.failure and session.failure.retryable and can_retry:
                    can_retry = False
                    session.disconnect()
                    session = None
                    reconnect_at = time.monotonic() + RECONNECT_DELAY
                    event_queue.put({'type': 'stream_reconnecting'})
                else:
                    session.report_closed(allow_recovery=True)
                    session = None

            if kind == 'start_stream':
                if session:
                    session.close()
                reconnect_at = None
                endpoint = message.get('endpoint')
                session = _open_session(event_queue, endpoint)
                can_retry = session is not None
            else:
                if reconnect_at is not None and time.monotonic() >= reconnect_at:
                    reconnect_at = None
                    session = _open_session(event_queue, endpoint, reconnecting=True)
                if kind == 'audio_chunk' and session:
                    session.send(message.get('pcm_bytes', b''))
    finally:
        if session:
            session.disconnect()
