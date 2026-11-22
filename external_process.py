"""Launch external Python without inheriting the frozen GUI's DLL search path."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import threading

_launch_lock = threading.Lock()


def clean_environment(env=None):
    result = dict(os.environ if env is None else env)
    for key in list(result):
        if key.upper() in ('PYTHONHOME', 'PYTHONPATH', 'QT_PLUGIN_PATH', 'QT_QPA_PLATFORM_PLUGIN_PATH') or key.startswith(('_PYI_', '_MEIPASS')):
            result.pop(key, None)
    bundle = getattr(sys, '_MEIPASS', None)
    if bundle:
        root = Path(bundle).resolve()
        result['PATH'] = os.pathsep.join(
            part for part in result.get('PATH', '').split(os.pathsep)
            if part and not Path(part).resolve().is_relative_to(root))
    if 'LD_LIBRARY_PATH_ORIG' in result:
        result['LD_LIBRARY_PATH'] = result['LD_LIBRARY_PATH_ORIG']
    elif bundle:
        result.pop('LD_LIBRARY_PATH', None)
    result.update(PYTHONUTF8='1', PYTHONUNBUFFERED='1', PYTHONNOUSERSITE='1')
    return result


def external_popen(command, *, env=None, **kwargs):
    env = clean_environment(env)
    with _launch_lock:
        if os.name != 'nt' or not getattr(sys, 'frozen', False):
            return subprocess.Popen(command, env=env, **kwargs)
        # PyInstaller documents that SetDllDirectoryW affects child processes.
        # Restore the GUI's search path immediately after creating this child.
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetDllDirectoryW.argtypes = [wintypes.DWORD, wintypes.LPWSTR]
        kernel.GetDllDirectoryW.restype = wintypes.DWORD
        kernel.SetDllDirectoryW.argtypes = [wintypes.LPCWSTR]
        kernel.SetDllDirectoryW.restype = wintypes.BOOL
        buffer = ctypes.create_unicode_buffer(32768)
        kernel.GetDllDirectoryW(len(buffer), buffer)
        previous = buffer.value
        if not kernel.SetDllDirectoryW(None): raise ctypes.WinError(ctypes.get_last_error())
        try:
            return subprocess.Popen(command, env=env, **kwargs)
        finally:
            if not kernel.SetDllDirectoryW(previous or None): raise ctypes.WinError(ctypes.get_last_error())
