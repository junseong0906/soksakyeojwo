from __future__ import annotations

import json
from multiprocessing import Queue
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import websocket


def _resolve_ffmpeg_executable() -> str:
    """패키징 환경(PyInstaller / 설치 폴더)에 동봉된 ffmpeg.exe를 우선 찾습니다."""
    candidates: list[Path] = []

    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        meipass = Path(getattr(sys, "_MEIPASS", exe_dir)).resolve()
        candidates.extend([
            meipass / "ffmpeg" / "ffmpeg.exe",
            meipass / "ffmpeg.exe",
            exe_dir / "ffmpeg" / "ffmpeg.exe",
            exe_dir / "ffmpeg.exe",
        ])

    project_root = Path(__file__).resolve().parent.parent
    candidates.extend([
        project_root / "assets" / "ffmpeg" / "ffmpeg.exe",
        project_root / "ffmpeg" / "ffmpeg.exe",
    ])

    for path in candidates:
        if path.is_file():
            return str(path)

    found = shutil.which("ffmpeg")
    return found if found else "ffmpeg"


def _time_to_ms(value: object) -> int | None:
    """WLK seconds (or H:MM:SS text) -> milliseconds; unknown stays None."""
    if value is None or value == "":
        return None
    try:
        if isinstance(value, (int, float)):
            return round(float(value) * 1000)
        parts = str(value).split(":")
        seconds = sum(float(part) * (60 ** i) for i, part in enumerate(reversed(parts)))
        return round(seconds * 1000)
    except (TypeError, ValueError):
        return None


def _speaker_id(value: object) -> str | None:
    if value is None or str(value).strip() in {"", "-2"}:
        return None
    raw = str(value).strip()
    return raw if raw.startswith("spk_") else f"spk_{raw}"


def _line_event(item: dict[str, object], is_final: bool = True) -> dict[str, object] | None:
    text = str(item.get("text", "")).strip()
    if not text:
        return None
    speaker = _speaker_id(item.get("speaker", item.get("speaker_id")))
    event: dict[str, object] = {
        "type": "subtitle",
        "text": text,
        "is_final": is_final,
        "speaker_id": speaker,
        "start_ms": _time_to_ms(item.get("start", item.get("start_time"))),
        "end_ms": _time_to_ms(item.get("end", item.get("end_time"))),
    }
    return event


def _parse_server_message(raw_msg: str) -> list[dict[str, object]]:
    """WLK JSON/text -> subtitle events. Confirmed lines retain speaker/time metadata."""
    text = raw_msg.strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except Exception:
        if text.lower() in {"ping", "pong", "heartbeat"}:
            return [{"type": "heartbeat"}]
        return [{
            "type": "subtitle",
            "text": text,
            "is_final": True,
            "speaker_id": None,
            "start_ms": None,
            "end_ms": None,
        }]

    if isinstance(data, dict):
        msg_type = str(data.get("type", data.get("status", ""))).lower()
        if msg_type in {"heartbeat", "ping", "pong"}:
            return [{"type": "heartbeat"}]
        events: list[dict[str, object]] = []
        lines = data.get("lines")
        if isinstance(lines, list):
            for item in lines:
                if isinstance(item, dict):
                    event = _line_event(item, True)
                    if event and event.get("speaker_id") is not None:
                        events.append(event)
                    elif event:  # WLK speaker=-2 is silence; don't present it as a speaker.
                        if str(item.get("speaker", "")) != "-2":
                            events.append(event)
        buf = data.get("buffer_transcription", "")
        if isinstance(buf, dict):
            partial = _line_event(buf, False)
        else:
            partial = _line_event({"text": buf}, False)
        if partial:
            events.append(partial)
        if events:
            return events
        transcript = (
            data.get("text")
            or data.get("transcript")
            or data.get("buffer")
            or data.get("transcription")
            or ""
        )
        segments = data.get("segments")
        if not transcript and isinstance(segments, list):
            transcript = " ".join(
                str(s.get("text", "")).strip() for s in segments if isinstance(s, dict)
            )
        if str(transcript).strip():
            events.append({
                "type": "subtitle",
                "text": str(transcript).strip(),
                "is_final": bool(
                    data.get("is_final", data.get("final", data.get("completed", False)))
                ),
                "speaker_id": None,
                "start_ms": None,
                "end_ms": None,
            })
        return events

    if isinstance(data, list):
        result: list[dict[str, object]] = []
        for item in data:
            if isinstance(item, dict):
                event = _line_event(item, bool(item.get("is_final", item.get("final", False))))
                if event and str(item.get("speaker", "")) != "-2":
                    result.append(event)
        return result

    return []


class WlkMessageProcessor:
    """Handles WLK mode=full snapshots without emitting the same confirmed line repeatedly."""
    def __init__(self) -> None:
        self._seen: set[tuple[object, ...]] = set()

    def reset(self) -> None:
        self._seen.clear()

    def process(self, raw_msg: str) -> list[dict[str, object]]:
        result: list[dict[str, object]] = []
        for event in _parse_server_message(raw_msg):
            if event.get("type") == "subtitle" and event.get("is_final"):
                key = (
                    event.get("speaker_id"),
                    event.get("start_ms"),
                    event.get("end_ms"),
                    event.get("text"),
                )
                if key in self._seen:
                    continue
                self._seen.add(key)
            result.append(event)
        return result


def asr_client_process_main(cmd_queue: Queue, event_queue: Queue) -> None:
    """별도 프로세스에서 실행되는 WLK(/asr) WebSocket 자막 클라이언트"""
    server_url = "ws://127.0.0.1:8000/asr"
    ws: websocket.WebSocket | None = None
    ffmpeg_proc: subprocess.Popen[bytes] | None = None
    stream_active = False
    message_processor = WlkMessageProcessor()

    def _ffmpeg_stdout_reader(proc: subprocess.Popen[bytes], get_ws) -> None:
        """ffmpeg가 만든 WebM 청크를 읽어 WebSocket으로 송출합니다."""
        try:
            assert proc.stdout is not None
            while True:
                chunk = proc.stdout.read(4096)
                if not chunk:
                    break
                cur_ws = get_ws()
                if cur_ws is not None:
                    try:
                        cur_ws.send(chunk, opcode=websocket.ABNF.OPCODE_BINARY)
                    except Exception:
                        break
        except Exception:
            pass

    def _start_ffmpeg() -> subprocess.Popen[bytes] | None:
        try:
            ffmpeg_bin = _resolve_ffmpeg_executable()
            creationflags = (
                subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
            )
            proc = subprocess.Popen(
                [
                    ffmpeg_bin,
                    "-loglevel",
                    "error",
                    "-f",
                    "s16le",
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    "-i",
                    "pipe:0",
                    "-c:a",
                    "libopus",
                    "-b:a",
                    "32k",
                    "-f",
                    "webm",
                    "pipe:1",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=creationflags,
            )
            reader = threading.Thread(
                target=_ffmpeg_stdout_reader,
                args=(proc, lambda: ws if stream_active else None),
                daemon=True,
            )
            reader.start()
            return proc
        except FileNotFoundError:
            event_queue.put({
                "type": "asr_error",
                "message": "ffmpeg를 찾을 수 없습니다. ffmpeg 설치 후 다시 시도해 주세요.",
            })
            return None
        except Exception as exc:
            event_queue.put({"type": "asr_error", "message": f"ffmpeg 시작 실패: {exc}"})
            return None

    def close_stream_gracefully() -> None:
        nonlocal ws, ffmpeg_proc, stream_active
        stream_active = False

        if ffmpeg_proc is not None:
            try:
                if ffmpeg_proc.stdin:
                    ffmpeg_proc.stdin.close()
            except Exception:
                pass
            time.sleep(0.15)
            try:
                ffmpeg_proc.terminate()
                ffmpeg_proc.wait(timeout=1.5)
            except Exception:
                try:
                    ffmpeg_proc.kill()
                except Exception:
                    pass
            ffmpeg_proc = None

        if ws is not None:
            try:
                ws.send(json.dumps({"type": "stop"}))
            except Exception:
                pass
            time.sleep(0.15)
            try:
                ws.recv()
            except Exception:
                pass
            try:
                ws.close()
            except Exception:
                pass
            ws = None

        event_queue.put({"type": "stream_closed"})

    def open_new_stream(target_url: str) -> bool:
        nonlocal ws, ffmpeg_proc, stream_active
        close_stream_gracefully()
        message_processor.reset()
        proc = _start_ffmpeg()
        if proc is None:
            return False
        try:
            client = websocket.WebSocket()
            client.settimeout(0.35)
            client.connect(target_url)
            ws = client
            ffmpeg_proc = proc
            stream_active = True
            event_queue.put({"type": "stream_opened", "url": target_url})
            return True
        except Exception as exc:
            try:
                proc.terminate()
            except Exception:
                pass
            ws = None
            ffmpeg_proc = None
            stream_active = False
            event_queue.put({"type": "asr_error", "message": f"WLK 연결 실패: {exc}"})
            return False

    while True:
        while not cmd_queue.empty():
            msg = cmd_queue.get_nowait()
            mtype = msg.get("type")

            if mtype == "set_url":
                new_url = str(msg.get("url", "")).strip()
                if new_url:
                    server_url = new_url

            elif mtype == "start_stream":
                url = str(msg.get("url") or server_url).strip()
                if url:
                    server_url = url
                open_new_stream(server_url)

            elif mtype == "pause_stream":
                close_stream_gracefully()

            elif mtype == "audio_chunk":
                if stream_active and ffmpeg_proc is not None and ffmpeg_proc.stdin is not None:
                    pcm_bytes = msg.get("pcm_bytes")
                    if pcm_bytes:
                        try:
                            ffmpeg_proc.stdin.write(pcm_bytes)
                            ffmpeg_proc.stdin.flush()
                        except Exception as exc:
                            stream_active = False
                            event_queue.put({
                                "type": "asr_error",
                                "message": f"오디오 인코딩 중 오류: {exc}",
                            })

            elif mtype == "shutdown":
                close_stream_gracefully()
                return

        if stream_active and ws is not None:
            try:
                raw = ws.recv()
                if raw is not None:
                    if isinstance(raw, bytes):
                        raw_str = raw.decode("utf-8", errors="ignore")
                    else:
                        raw_str = str(raw)
                    for ev in message_processor.process(raw_str):
                        event_queue.put(ev)
            except websocket.WebSocketTimeoutException:
                pass
            except Exception as exc:
                stream_active = False
                try:
                    if ws is not None:
                        ws.close()
                except Exception:
                    pass
                ws = None
                event_queue.put({"type": "asr_error", "message": f"자막 수신 중단: {exc}"})

        time.sleep(0.02)