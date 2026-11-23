from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

WLK_VERSION = '0.2.28'

@dataclass(frozen=True)
class ModelSpec:
    id: str
    name: str
    repo: str
    family: str
    backend: str
    size_hint: str
    description: str
    def arguments(self, path: Path, diarization: bool = False, device: str = 'auto') -> list[str]:
        args = ['--backend', self.backend, '--language', 'ko', '--model-path', str(path), '--pcm-input']
        if self.family == 'whisper':
            # LocalAgreement repeatedly decodes overlapping audio. A modest
            # batch interval leaves GPU time for transcription and Diart.
            args += ['--backend-policy', 'localagreement', '--asr-coalesce-min-s', '0.5']
        if self.family == 'qwen':
            # Windowed mode supports Korean; the causal checkpoint is English-only.
            args += ['--qwen3-streaming-audio-backend', 'windowed', '--qwen3-streaming-device', device]
        if diarization:
            args += ['--diarization', '--diarization-backend', 'diart']
        return args

MODELS = (
    ModelSpec('whisper-tiny', 'Whisper Tiny', 'Systran/faster-whisper-tiny', 'whisper', 'faster-whisper', '약 78 MB', '가벼운 모델 · 빠른 동작 확인에 적합'),
    ModelSpec('whisper-small', 'Whisper Small', 'Systran/faster-whisper-small', 'whisper', 'faster-whisper', '약 486 MB', '한국어 지원 · 속도와 정확도의 균형'),
    ModelSpec('whisper-medium', 'Whisper Medium', 'Systran/faster-whisper-medium', 'whisper', 'faster-whisper', '약 1.53 GB', '한국어 지원 · GPU 권장'),
    ModelSpec('whisper-large-v3', 'Whisper Large-v3', 'Systran/faster-whisper-large-v3', 'whisper', 'faster-whisper', '약 3.09 GB', '한국어 지원 · 높은 정확도 · GPU 권장'),
    ModelSpec('whisper-turbo', 'Whisper Large-v3 Turbo', 'mobiuslabsgmbh/faster-whisper-large-v3-turbo', 'whisper', 'faster-whisper', '약 1.62 GB', '한국어 지원 · Large-v3 계열의 빠른 모델'),
    ModelSpec('qwen-0.6b', 'Qwen3-ASR-0.6B', 'Qwen/Qwen3-ASR-0.6B', 'qwen', 'qwen3-streaming', '약 1.9 GB', '한국어 지원 · Transformers 스트리밍 · GPU 권장'),
    ModelSpec('voxtral-4b', 'Voxtral Mini 4B Realtime', 'mistralai/Voxtral-Mini-4B-Realtime-2602', 'voxtral', 'voxtral', '약 8.9 GB', '한국어 지원 · 실시간 스트리밍 · 충분한 GPU 메모리 권장'),
)
MODEL_BY_ID = {model.id: model for model in MODELS}
def get_model(model_id: str) -> ModelSpec:
    return MODEL_BY_ID.get(model_id, MODELS[0])
