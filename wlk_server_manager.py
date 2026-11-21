from __future__ import annotations

from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time


def _log_diag(msg: str) -> None:
    print(f"[WLK_MGR] {msg}", file=sys.stderr, flush=True)


def _is_port_open(host: str, port: int, timeout: float = 0.25) -> bool:
    """대상 호스트/포트가 이미 열려 있는지 확인합니다."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _resolve_wlk_command(
    model: str = "tiny",
    backend: str = "faster-whisper",
    language: str = "ko",
    host: str = "127.0.0.1",
    port: int = 8000,
) -> list[str] | None:
    """실행 환경(동봉 런타임 / 가상환경 / PATH)에서 wlk 실행 명령을 찾습니다."""
    base_args = [
        "--model",
        model,
        "--backend",
        backend,
        "--lan",
        language,
        "--host",
        host,
        "--port",
        str(port),
    ]

    project_root = Path(__file__).resolve().parent
    candidates: list[Path] = []

    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        meipass = Path(getattr(sys, "_MEIPASS", exe_dir)).resolve()
        candidates.extend([
            exe_dir / "wlk.exe",
            meipass / "wlk.exe",
            exe_dir / "wlk" / "wlk.exe",
            meipass / "wlk" / "wlk.exe",
        ])
    else:
        venv_dir = Path(sys.executable).resolve().parent
        candidates.extend([
            venv_dir / "wlk.exe",
            project_root / ".venv" / "Scripts" / "wlk.exe",
        ])

    for exe_path in candidates:
        if exe_path.is_file():
            return [str(exe_path), *base_args]

    found_wlk = shutil.which("wlk")
    if found_wlk:
        return [found_wlk, *base_args]

    return None


class WlkServerManager:
    """앱 실행 시 wlk 서버를 백그라운드로 띄우고 종료 시 정리합니다."""

    def __init__(
        self,
        model: str = "tiny",
        backend: str = "faster-whisper",
        language: str = "ko",
        host: str = "127.0.0.1",
        port: int = 8000,
    ) -> None:
        self.model = model
        self.backend = backend
        self.language = language
        self.host = host
        self.port = port
        self._proc: subprocess.Popen[bytes] | None = None
        self._log_file = None
        self._started_by_us = False

    def start_if_needed(self, wait_timeout: float = 20.0) -> bool:
        """포트가 닫혀 있을 때만 wlk 서버를 실행하고 포트가 열릴 때까지 기다립니다."""
        _log_diag(f"sys.executable={sys.executable}, frozen={getattr(sys, 'frozen', False)}")

        if _is_port_open(self.host, self.port):
            _log_diag(f"이미 {self.host}:{self.port} 포트가 열려 있음")
            self._started_by_us = False
            return True

        cmd = _resolve_wlk_command(
            model=self.model,
            backend=self.backend,
            language=self.language,
            host=self.host,
            port=self.port,
        )
        _log_diag(f"resolved cmd={cmd}")
        if not cmd:
            _log_diag("wlk 실행 파일을 찾지 못함")
            return False

        log_path = Path(__file__).resolve().parent / "wlk_server.log"
        try:
            self._log_file = open(log_path, "ab", buffering=0)
            creationflags = (
                subprocess.CREATE_NO_WINDOW
                | subprocess.CREATE_NEW_PROCESS_GROUP
                if hasattr(subprocess, "CREATE_NO_WINDOW")
                else 0
            )
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=self._log_file,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
            )
            self._started_by_us = True
            _log_diag(f"프로세스 시작됨 pid={self._proc.pid}, log={log_path}")
        except Exception as exc:
            _log_diag(f"프로세스 시작 실패: {exc}")
            self._close_log_file()
            self._proc = None
            self._started_by_us = False
            return False

        deadline = time.monotonic() + wait_timeout
        while time.monotonic() < deadline:
            if _is_port_open(self.host, self.port):
                _log_diag(f"{self.host}:{self.port} 포트 열림 확인")
                return True
            exit_code = self._proc.poll()
            if exit_code is not None:
                _log_diag(f"wlk 프로세스가 조기 종료됨 (exit_code={exit_code})")
                return False
            time.sleep(0.25)

        opened = _is_port_open(self.host, self.port)
        _log_diag(f"대기 시간 종료, 포트 열림 여부={opened}")
        return opened

    def _close_log_file(self) -> None:
        if self._log_file is not None:
            try:
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None

    def stop_if_started(self) -> None:
        """이 매니저가 직접 실행한 wlk 서버 프로세스만 종료합니다."""
        if not self._started_by_us or self._proc is None:
            self._close_log_file()
            return

        proc = self._proc
        self._proc = None
        self._started_by_us = False

        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=3.0)
            except Exception:
                try:
                    proc.kill()
                    proc.wait(timeout=2.0)
                except Exception:
                    pass

        self._close_log_file()