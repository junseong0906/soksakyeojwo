from __future__ import annotations

def normalize_speaker_id(value) -> str | None:
    raw = str(value).strip() if value is not None else ''
    if raw in ('', '-2', '-1', '0', 'spk_0'):
        return None
    return raw if raw.startswith('spk_') else f'spk_{raw}'

class SpeakerLabeler:
    """One shared implementation for the UI and tests. IDs are session-local."""
    def __init__(self, labels=None):
        self._labels = {}
        for sid, name in (labels or {}).items():
            self.set_label(sid, name)

    def set_label(self, speaker_id, label):
        sid = normalize_speaker_id(speaker_id)
        if not sid:
            raise ValueError('유효한 화자 ID를 입력해 주세요. 예: 1 또는 spk_1')
        if str(label).strip():
            self._labels[sid] = str(label).strip()
        else:
            self._labels.pop(sid, None)

    set_custom_name = set_label

    def get_custom_name(self, speaker_id):
        return self._labels.get(normalize_speaker_id(speaker_id), '')

    def get_label(self, speaker_id):
        sid = normalize_speaker_id(speaker_id)
        return self._labels.get(sid, f'화자 {sid[4:]}') if sid else '화자 미확인'

    def apply(self, event):
        return {**event, 'speaker_label': self.get_label(event.get('speaker_id'))}

    def as_dict(self):
        return dict(self._labels)
