from __future__ import annotations

from datetime import datetime
from multiprocessing import Queue
from pathlib import Path
import time

import numpy as np
import sounddevice as sd
import soundfile as sf


def query_input_devices() -> list[dict[str, object]]:
    """Windows 입력 장치 목록과 기본 입력 장치 인덱스를 반환합니다."""
    devices = sd.query_devices()
    default_input = sd.default.device[0]
    result: list[dict[str, object]] = []

    for idx, dev in enumerate(devices):
        max_in = int(dev.get("max_input_channels", 0))
        if max_in > 0:
            name = str(dev.get("name", f"입력 장치 {idx}"))
            is_default = idx == default_input
            label = f"[기본] {name}" if is_default else name
            result.append({
                "index": idx,
                "name": name,
                "label": label,
                "is_default": is_default,
                "channels": min(max_in, 1),
                "samplerate": int(dev.get("default_samplerate") or 16000),
            })
    return result


def recorder_process_main(
    cmd_queue: Queue,
    event_queue: Queue,
    asr_cmd_queue: Queue | None = None,
) -> None:
    """별도 프로세스에서 실행되는 마이크 레벨 모니터링·로컬 녹음·ASR 오디오 송출 워커"""
    base_dir = Path.cwd() / "recordings"
    base_dir.mkdir(parents=True, exist_ok=True)

    selected_device: int | None = None
    is_recording = False
    asr_stream_enabled = False
    sf_file: sf.SoundFile | None = None
    stream: sd.InputStream | None = None
    current_samplerate = 16000
    current_channels = 1
    last_push_time = 0.0

    def close_stream() -> None:
        nonlocal stream, sf_file
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
            stream = None
        if sf_file is not None:
            try:
                sf_file.close()
            except Exception:
                pass
            sf_file = None

    def audio_callback(indata: np.ndarray, frames: int, time_info, status) -> None:
        nonlocal last_push_time
        if indata.size == 0:
            return

        mono = indata[:, 0].astype(np.float32) if indata.ndim > 1 else indata.reshape(-1).astype(np.float32)
        rms = float(np.sqrt(np.mean(np.square(mono))))
        level = max(0, min(100, int(min(1.0, rms * 8.0) * 100)))

        if is_recording and sf_file is not None:
            try:
                sf_file.write(indata)
            except Exception:
                pass

        # 실시간 자막 스트림이 켜져 있으면 16kHz mono int16 PCM 청크 전송
        if asr_stream_enabled and asr_cmd_queue is not None:
            try:
                if current_samplerate != 16000 and len(mono) > 0:
                    target_len = max(1, int(round(len(mono) * 16000.0 / current_samplerate)))
                    src_x = np.linspace(0.0, 1.0, num=len(mono), endpoint=False)
                    dst_x = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
                    resampled = np.interp(dst_x, src_x, mono)
                else:
                    resampled = mono

                pcm16 = (np.clip(resampled, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
                asr_cmd_queue.put_nowait({"type": "audio_chunk", "pcm_bytes": pcm16})
            except Exception:
                pass

        now = time.monotonic()
        if now - last_push_time >= 0.1:
            last_push_time = now
            try:
                event_queue.put_nowait({"type": "level", "value": level})
            except Exception:
                pass

    def open_input_stream(device_idx: int | None) -> bool:
        nonlocal stream, current_samplerate, current_channels
        close_stream()
        try:
            dev_info = sd.query_devices(device_idx if device_idx is not None else sd.default.device[0], "input")
            sr = int(dev_info.get("default_samplerate") or 16000)
            ch = min(int(dev_info.get("max_input_channels", 1)), 1) or 1
            current_samplerate = sr
            current_channels = ch

            stream = sd.InputStream(
                device=device_idx,
                channels=ch,
                samplerate=sr,
                dtype="float32",
                callback=audio_callback,
            )
            stream.start()
            return True
        except Exception as exc:
            event_queue.put({"type": "error", "message": f"마이크 열기 실패: {exc}"})
            return False

    try:
        devs = query_input_devices()
        default_dev = next((d for d in devs if d["is_default"]), devs[0] if devs else None)
        selected_device = int(default_dev["index"]) if default_dev is not None else None
        event_queue.put({"type": "devices", "devices": devs, "selected": selected_device})
        if selected_device is not None:
            open_input_stream(selected_device)
    except Exception as exc:
        event_queue.put({"type": "error", "message": f"오디오 장치 초기화 실패: {exc}"})

    while True:
        while not cmd_queue.empty():
            msg = cmd_queue.get_nowait()
            mtype = msg.get("type")

            if mtype == "select_device":
                selected_device = msg.get("device_index")
                was_recording = is_recording
                rec_path = None
                if was_recording:
                    close_stream()
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    rec_path = str(base_dir / f"lecture_{ts}.wav")
                    sf_file = sf.SoundFile(
                        rec_path,
                        mode="w",
                        samplerate=current_samplerate,
                        channels=current_channels,
                        subtype="PCM_16",
                    )
                if open_input_stream(selected_device):
                    if was_recording and rec_path:
                        event_queue.put({"type": "recording_started", "file": rec_path})

            elif mtype == "start_recording":
                if not is_recording:
                    if stream is None and not open_input_stream(selected_device):
                        continue
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    rec_path = str(base_dir / f"lecture_{ts}.wav")
                    try:
                        sf_file = sf.SoundFile(
                            rec_path,
                            mode="w",
                            samplerate=current_samplerate,
                            channels=current_channels,
                            subtype="PCM_16",
                        )
                        is_recording = True
                        event_queue.put({"type": "recording_started", "file": rec_path})
                    except Exception as exc:
                        event_queue.put({"type": "error", "message": f"녹음 파일 생성 실패: {exc}"})

            elif mtype == "stop_recording":
                if is_recording:
                    is_recording = False
                    if sf_file is not None:
                        try:
                            sf_file.close()
                        except Exception:
                            pass
                        sf_file = None
                    event_queue.put({"type": "recording_stopped"})

            elif mtype == "set_asr_streaming":
                asr_stream_enabled = bool(msg.get("enabled", False))
                if asr_stream_enabled and stream is None:
                    open_input_stream(selected_device)

            elif mtype == "refresh_devices":
                devs = query_input_devices()
                event_queue.put({"type": "devices", "devices": devs, "selected": selected_device})

            elif mtype == "shutdown":
                is_recording = False
                asr_stream_enabled = False
                close_stream()
                return

        time.sleep(0.05)