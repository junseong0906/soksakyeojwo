from __future__ import annotations

class TranscriptStore:
    """Retain pruned history and replace the current WLK window, including speaker revisions."""
    def __init__(self):
        self.archived = []
        self.active = []
        self.partial = ''
        self.diarization_buffer = ''

    def update(self, event):
        lines = [dict(line) for line in event.get('lines', [])]
        if lines:
            starts = [line['start_ms'] for line in lines if line.get('start_ms') is not None]
            if starts:
                cutoff = min(starts)
                for line in self.active:
                    if line.get('start_ms') is not None and line['start_ms'] < cutoff and (line.get('end_ms') or 0) <= cutoff:
                        self.archived.append(line)
            self.active = lines
        self.partial = event.get('partial', '')
        self.diarization_buffer = event.get('diarization_buffer', '')

    @property
    def lines(self): return self.archived + self.active

    def finalize(self):
        # Keep pending text visible if a backend finishes with a buffer only.
        pending = ' '.join(filter(None, [self.diarization_buffer, self.partial])).strip()
        if pending:
            self.active.append({'text': pending, 'speaker_id': None, 'start_ms': None, 'end_ms': None, 'is_final': True})
        self.partial = self.diarization_buffer = ''
