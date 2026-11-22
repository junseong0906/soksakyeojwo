import os
from pathlib import Path
import sys
import threading

import pytest

import runtime_install as installer
from external_process import clean_environment


def test_quiet_installer_keeps_reporting_elapsed_time(tmp_path):
    updates = []
    installer.run_installer([sys.executable, '-c', 'import time; time.sleep(.35); print("done")'],
                            lambda p, m: updates.append(m), threading.Event(), tmp_path / 'install.log',
                            heartbeat=.05)
    assert len(updates) >= 2
    assert all('경과' in message for message in updates)
    assert 'done' in (tmp_path / 'install.log').read_text(encoding='utf-8')


def test_hung_installer_times_out_and_terminates_child(tmp_path, monkeypatch):
    monkeypatch.setattr(installer, 'activity', lambda pid: ())
    pid_file = tmp_path / 'pid'
    command = [sys.executable, '-c', f'import os,time; from pathlib import Path; Path({str(pid_file)!r}).write_text(str(os.getpid())); time.sleep(60)']
    with pytest.raises(RuntimeError, match='CPU·디스크 활동'):
        installer.run_installer(command, lambda *_: None, threading.Event(), tmp_path / 'install.log',
                                heartbeat=.05, idle_timeout=.3)
    import psutil
    assert not psutil.pid_exists(int(pid_file.read_text()))


def test_cancel_silent_installer_does_not_wait_for_stdout(tmp_path):
    cancel = threading.Event()
    timer = threading.Timer(.3, cancel.set)
    timer.start()
    try:
        with pytest.raises(installer.InstallCancelled):
            installer.run_installer([sys.executable, '-c', 'import time; time.sleep(60)'],
                                    lambda *_: None, cancel, tmp_path / 'install.log', heartbeat=.05)
    finally: timer.join()


def test_installer_activity_prevents_false_idle_timeout(tmp_path, monkeypatch):
    counter = iter(range(1000))
    monkeypatch.setattr(installer, 'activity', lambda pid: (next(counter),))
    installer.run_installer([sys.executable, '-c', 'import time; time.sleep(.4)'],
                            lambda *_: None, threading.Event(), tmp_path / 'install.log',
                            heartbeat=.04, idle_timeout=.12)


def test_pip_raw_progress_and_token_redaction():
    detail = installer.describe('Downloading torch-2.8.0.whl', '')
    assert '50%' in installer.describe('Progress 500000 of 1000000', detail)
    assert '0.5 / 1.0 MB' in installer.describe('Progress 500000 of 1000000', detail)
    assert 'secret-value' not in installer.redact('token: secret-value', {'HF_TOKEN': 'secret-value'})


def test_external_python_environment_does_not_inherit_gui_paths(monkeypatch, tmp_path):
    bundle = tmp_path / 'bundle'
    monkeypatch.setattr(sys, '_MEIPASS', str(bundle), raising=False)
    env = clean_environment({'PYTHONHOME': 'bad', 'PYTHONPATH': 'bad', '_PYI_PARENT_PROCESS_LEVEL': '1',
                             'QT_PLUGIN_PATH': 'bad', 'HF_TOKEN': 'keep',
                             'PATH': os.pathsep.join([str(bundle / 'Qt'), str(tmp_path / 'python')])})
    assert 'PYTHONHOME' not in env and 'PYTHONPATH' not in env
    assert '_PYI_PARENT_PROCESS_LEVEL' not in env and 'QT_PLUGIN_PATH' not in env
    assert env['HF_TOKEN'] == 'keep'
    assert env['PATH'] == str(tmp_path / 'python')


def test_partial_python_repairs_missing_pip(tmp_path, monkeypatch):
    from model_service import RuntimeManager
    runtime = RuntimeManager(tmp_path)
    calls = []
    def run(command, *args, **kwargs):
        calls.append(command)
        if len(calls) == 1: raise RuntimeError('pip missing')
    monkeypatch.setattr(runtime, 'run', run)
    class Download:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return b'# get-pip'
    monkeypatch.setattr('urllib.request.urlopen', lambda *a, **k: Download())
    runtime._ensure_pip(tmp_path / 'python.exe', tmp_path, lambda *_: None, threading.Event())
    assert len(calls) == 2
    assert (tmp_path / 'get-pip.py').read_bytes() == b'# get-pip'
