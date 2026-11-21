from __future__ import annotations

from typing import Any


class SpeakerLabeler:
    """Maps stream-local anonymous speaker IDs (e.g. spk_1) to user labels."""
    def __init__(self, labels: dict[str, str] | None = None) -> None:
        self._labels = dict(labels or {})

    def set_label(self, speaker_id: str, label: str) -> None:
        sid = str(speaker_id).strip()
        name = str(label).strip()
        if not sid:
            raise ValueError("speaker_id는 비어 있을 수 없습니다.")
        if name:
            self._labels[sid] = name
        else:
            self._labels.pop(sid, None)

    def get_label(self, speaker_id: str | None) -> str:
        if not speaker_id:
            return "화자 미확인"
        return self._labels.get(speaker_id, f"화자 {speaker_id.removeprefix('spk_')}")

    def apply(self, event: dict[str, Any]) -> dict[str, Any]:
        """Return a copy with display label; does not alter original event."""
        result = dict(event)
        result["speaker_label"] = self.get_label(result.get("speaker_id"))
        return result

    def as_dict(self) -> dict[str, str]:
        return dict(self._labels)
