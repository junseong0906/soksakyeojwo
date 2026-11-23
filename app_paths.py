from __future__ import annotations
import json
import os
from pathlib import Path
import sys
import tempfile

def app_dir() -> Path:
    return Path(sys.executable).resolve().parent if getattr(sys, 'frozen', False) else Path(__file__).resolve().parent

def data_dir() -> Path:
    path = Path(os.environ['SOKSAK_DATA_DIR']) if os.environ.get('SOKSAK_DATA_DIR') else app_dir() / 'data'
    path.mkdir(parents=True, exist_ok=True)
    return path

def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)

class SettingsStore:
    def __init__(self, path: Path | None = None):
        self.path = path or data_dir() / 'settings.json'
        self.value = {'model_id': 'whisper-tiny', 'diarization': False, 'speaker_labels': {}, 'device': 'auto'}
        try:
            loaded = json.loads(self.path.read_text(encoding='utf-8'))
            if isinstance(loaded, dict): self.value.update({k: v for k, v in loaded.items() if k in self.value})
        except (OSError, ValueError): pass
    def save(self, **changes):
        self.value.update(changes)
        atomic_json(self.path, self.value)
