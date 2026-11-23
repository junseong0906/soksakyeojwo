"""Supervise only our installer process tree, including quiet wheel extraction."""
from __future__ import annotations

from collections import deque
from datetime import datetime
import os
from pathlib import Path
import queue
import re
import subprocess
import threading
import time

import psutil
from external_process import external_popen


class InstallCancelled(Exception):
    pass


def terminate_tree(proc):
    if proc.poll() is not None: return
    if os.name == 'nt':
        subprocess.run(['taskkill', '/PID', str(proc.pid), '/T', '/F'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       creationflags=subprocess.CREATE_NO_WINDOW, timeout=10)
    else:
        try:
            for child in psutil.Process(proc.pid).children(recursive=True): child.kill()
        except psutil.Error: pass
        proc.kill()
    proc.wait(timeout=10)


def activity(pid):
    """CPU and I/O counters keep large wheel extraction from looking idle."""
    try:
        parent = psutil.Process(pid)
        processes = [parent, *parent.children(recursive=True)]
    except psutil.Error: return ()
    counters = []
    for process in processes:
        try:
            cpu, io = process.cpu_times(), process.io_counters()
            counters.append((process.pid, cpu.user + cpu.system, io.read_bytes, io.write_bytes))
        except psutil.Error: pass
    return tuple(sorted(counters))


def redact(text, env=None):
    for key, value in (env or os.environ).items():
        if value and ('TOKEN' in key.upper() or 'PASSWORD' in key.upper()): text = text.replace(value, '[secret]')
    text = re.sub(r'https?://[^\s/@]+:[^\s/@]+@', 'https://[credentials]@', text)
    return re.sub(r'hf_[A-Za-z0-9]{12,}', '[token]', text)


def describe(line, previous):
    raw = re.search(r'Progress (\d+) of (\d+)', line)
    if raw:
        current, total = map(int, raw.groups())
        if total:
            return f'{previous.split(" · ")[0]} · {current / 1e6:.1f} / {total / 1e6:.1f} MB ({min(100, current * 100 // total)}%)'
    if line.startswith(('Downloading ', 'Using cached ')):
        return '파일 준비: ' + line.split(maxsplit=1)[1][:130]
    if line.startswith('Installing collected packages:'):
        return '구성요소 압축 해제·설치 중 (CUDA Torch는 수 분이 걸릴 수 있습니다)'
    if line.startswith('Collecting '): return '패키지 확인: ' + line[11:][:130]
    if line.startswith('Successfully installed '): return '패키지 설치 완료'
    if line.startswith('No broken requirements'): return '패키지 호환성 검사 통과'
    if line.startswith('CUDA:'): return line[:150]
    if line.startswith(('ERROR:', 'WARNING:', 'pip is looking at', 'INFO: pip is looking at')): return line[:180]
    return previous


def run_installer(command, progress, cancel, log_path, *, env=None, stage='실행환경 준비',
                  idle_timeout=600, total_timeout=2700, heartbeat=1):
    if cancel.is_set(): raise InstallCancelled('작업을 중지했습니다. 다시 시도하면 설치를 이어갑니다.')
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists() and log_path.stat().st_size > 10_000_000:
        log_path.replace(log_path.with_suffix('.previous.log'))
    lines, messages = deque(maxlen=30), queue.Queue()
    proc = external_popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding='utf-8', errors='replace', env=env,
                          creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    def read_output():
        try:
            for line in proc.stdout: messages.put(line.rstrip())
        finally: messages.put(None)
    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    start = last_activity = last_emit = time.monotonic()
    counters, detail, eof = (), '시작 중…', False
    try:
        with log_path.open('a', encoding='utf-8', buffering=1) as log:
            log.write(f'\n[{datetime.now().isoformat(timespec="seconds")}] {stage}\n')
            while not eof or proc.poll() is None:
                if cancel.is_set(): raise InstallCancelled('작업을 중지했습니다. 다시 시도하면 설치를 이어갑니다.')
                try:
                    line = messages.get(timeout=min(.2, heartbeat))
                    if line is None: eof = True
                    else:
                        line = redact(line, env)
                        log.write(line + '\n')
                        lines.append(line)
                        detail = describe(line.strip(), detail)
                        last_activity = time.monotonic()
                except queue.Empty: pass
                now = time.monotonic()
                if now - last_emit >= heartbeat:
                    current = activity(proc.pid)
                    if current and current != counters: last_activity = now
                    counters = current
                    elapsed = int(now - start)
                    progress(-1, f'{stage} · {elapsed // 60:02}:{elapsed % 60:02} 경과\n{detail}')
                    last_emit = now
                if now - last_activity > idle_timeout:
                    raise TimeoutError(f'설치 출력과 CPU·디스크 활동이 {int(idle_timeout)}초 동안 없습니다. 중지 후 다시 적용해 주세요.')
                if now - start > total_timeout:
                    raise TimeoutError('설치 제한 시간(45분)을 초과했습니다. 네트워크를 확인한 뒤 다시 적용해 주세요.')
            code = proc.wait()
            if code:
                raise RuntimeError(f'실행환경 준비 실패 (종료 코드 {code}).\n' + '\n'.join(lines)[-3500:])
            log.write(f'[{stage}] 완료\n')
    except Exception as exc:
        terminate_tree(proc)
        with log_path.open('a', encoding='utf-8') as log: log.write(f'{type(exc).__name__}: {redact(str(exc), env)}\n')
        if isinstance(exc, InstallCancelled): raise
        raise RuntimeError(f'{exc}\n설치 로그: {log_path}') from exc
    finally:
        proc.stdout.close()
        reader.join(timeout=2)
