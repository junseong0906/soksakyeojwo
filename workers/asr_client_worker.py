from __future__ import annotations
import json
from multiprocessing import Queue
from queue import Empty
import threading
import time
import websocket
from speaker_labeler import normalize_speaker_id

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
        if kind == 'error' or data.get('status') == 'error':
            return [{'type': 'asr_error', 'message': str(data.get('error') or data.get('message') or '서버 처리 오류')}]
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
        if isinstance(data, dict) and ('lines' in data or 'buffer_transcription' in data or 'buffer_diarization' in data):
            lines = [event for item in data.get('lines', []) if isinstance(item, dict)
                     if (event := _line_event(item))]
            event = {'type': 'transcript_snapshot', 'lines': lines,
                     'partial': buffer_text(data.get('buffer_transcription')),
                     'diarization_buffer': buffer_text(data.get('buffer_diarization'))}
            signature = json.dumps(event, ensure_ascii=False, sort_keys=True)
            if signature == self.signature: return []
            self.signature = signature
            return [event]
        return _parse_server_message(raw_msg)

class PcmSession:
    """Raw PCM client for --pcm-input; receiver remains alive until ready_to_stop."""
    def __init__(self, events, factory=None):
        self.events = events
        self.factory = factory or websocket.WebSocket
        self.ws = None
        self.ready = threading.Event()
        self.receiver = None
        self.processor = WlkMessageProcessor()
        self.stopped = threading.Event()

    def open(self, url):
        self.ws = self.factory(enable_multithread=True)
        self.ws.settimeout(10)
        self.ws.connect(url)
        config = json.loads(self.ws.recv())
        if config.get('type') != 'config' or not config.get('useAudioWorklet'):
            self.ws.close()
            self.ws = None
            raise RuntimeError('서버가 PCM 입력을 지원하지 않습니다. 설정에서 모델을 적용하거나 외부 WLK를 --pcm-input으로 실행해 주세요.')
        self.ws.settimeout(.2)
        self.events.put({'type': 'stream_opened', 'url': url})
        self.receiver = threading.Thread(target=self._receive, daemon=True)
        self.receiver.start()

    def _receive(self):
        try:
            while not self.stopped.is_set():
                try: raw = self.ws.recv()
                except websocket.WebSocketTimeoutException: continue
                if raw in ('', None): break
                if isinstance(raw, bytes): raw = raw.decode('utf-8')
                for event in self.processor.process(raw):
                    if event['type'] == 'ready_to_stop':
                        self.events.put({'type': 'transcript_finalized'})
                        self.ready.set()
                        return
                    self.events.put(event)
        except Exception as exc:
            if not self.stopped.is_set(): self.events.put({'type': 'asr_error', 'message': f'자막 수신 중단: {exc}'})
        finally:
            if not self.ready.is_set() and not self.stopped.is_set():
                self.events.put({'type': 'asr_error', 'message': '서버 연결이 처리 완료 전에 종료되었습니다.'})

    def send(self, pcm):
        if self.ws and pcm: self.ws.send(pcm, opcode=websocket.ABNF.OPCODE_BINARY)

    def close(self, timeout=45):
        if not self.ws: return
        try:
            # Previous audio from the recorder producer is ahead of this EOF in its queue.
            self.ws.send(b'', opcode=websocket.ABNF.OPCODE_BINARY)
            self.events.put({'type': 'stream_draining'})
            deadline = time.monotonic() + timeout
            while not self.ready.is_set() and time.monotonic() < deadline:
                if self.receiver is not None and not self.receiver.is_alive(): break
                self.ready.wait(.1)
            if not self.ready.is_set():
                self.events.put({'type': 'asr_warning', 'message': '마지막 자막 처리 완료를 확인하지 못했습니다. 로컬 녹음 파일을 확인해 주세요.'})
        except Exception as exc:
            self.events.put({'type': 'asr_warning', 'message': f'스트림 종료 확인 실패: {exc}'})
        finally:
            self.stopped.set()
            self.ws.close()
            if self.receiver: self.receiver.join(timeout=1)
            self.ws = None
            self.events.put({'type': 'stream_closed'})

def asr_client_process_main(cmd_queue: Queue, event_queue: Queue):
    session = None
    while True:
        try: message = cmd_queue.get(timeout=.1)
        except Empty: continue
        kind = message.get('type')
        try:
            if kind == 'start_stream':
                if session: session.close()
                session = PcmSession(event_queue)
                session.open(str(message['url']))
            elif kind == 'audio_chunk' and session:
                session.send(message.get('pcm_bytes', b''))
            elif kind in ('pause_stream', 'shutdown'):
                if session:
                    session.close()
                    session = None
                else: event_queue.put({'type': 'stream_closed'})
                if kind == 'shutdown': return
        except Exception as exc:
            event_queue.put({'type': 'asr_error', 'message': str(exc)})
            if session and session.ws:
                session.stopped.set()
                session.ws.close()
            session = None
            event_queue.put({'type': 'stream_closed'})
