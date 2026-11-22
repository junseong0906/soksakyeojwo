import sys
import types
from contextlib import contextmanager
from importlib.machinery import ModuleSpec

import pytest

from wlk_server_manager import SERVER_BOOTSTRAP


@pytest.mark.parametrize('custom', [False, True])
def test_wlk_core_arguments_reach_legacy_diart_constructor(monkeypatch, custom):
    """Reproduce the real 0.2.28 core call against its backend's signature."""
    captured = {}
    scope = {'active': False}
    @contextmanager
    def safe_globals(classes):
        assert len(classes) == 17
        scope['active'] = True
        try: yield
        finally: scope['active'] = False
    torch = types.ModuleType('torch')
    torch.__spec__ = ModuleSpec('torch', loader=None, origin='fake-torch.py')
    torch.serialization = types.SimpleNamespace(safe_globals=safe_globals)
    version = types.ModuleType('torch.torch_version')
    version.TorchVersion = type('TorchVersion', (str,), {})
    task = types.ModuleType('pyannote.audio.core.task')
    for name in ('Specifications', 'Problem', 'Resolution'): setattr(task, name, type(name, (), {}))
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setitem(sys.modules, version.__name__, version)
    monkeypatch.setitem(sys.modules, task.__name__, task)
    imports = types.ModuleType('speechbrain.utils.importutils')
    class LazyModule(types.ModuleType):
        def __init__(self):
            super().__init__('unused-speechbrain-alias')
            self.lazy_module = None
        def __getattr__(self, name):
            if name == '__file__': raise RecursionError('inspect re-entered the lazy loader')
            return 'lazy-value'
    imports.LazyModule = LazyModule
    monkeypatch.setitem(sys.modules, imports.__name__, imports)
    for name, attributes in {
        'pyannote.audio.core.model': ('Introspection',),
        'omegaconf': ('DictConfig', 'ListConfig'),
        'omegaconf.base': ('ContainerMetadata', 'Metadata'),
        'omegaconf.nodes': ('AnyNode',),
        'pytorch_lightning.callbacks': ('EarlyStopping', 'ModelCheckpoint'),
    }.items():
        module = types.ModuleType(name)
        for attribute in attributes: setattr(module, attribute, type(attribute, (), {}))
        monkeypatch.setitem(sys.modules, name, module)
    class OriginalDiart:
        def __init__(self, block_duration=1.5, segmentation_model_name='default-seg', embedding_model_name='default-emb'):
            assert scope['active'], 'Checkpoint metadata must be scoped to the model load'
            alias = LazyModule()
            assert not hasattr(alias, '__file__')
            assert alias.some_member == 'lazy-value'
            captured.update(block_duration=block_duration, segmentation=segmentation_model_name, embedding=embedding_model_name)
    backend = types.ModuleType('whisperlivekit.diarization.diart_backend')
    backend.DiartDiarization = OriginalDiart
    server = types.ModuleType('whisperlivekit.basic_server')
    server.main = lambda: backend.DiartDiarization(block_duration=.5,
        segmentation_model='custom-seg' if custom else None, embedding_model='custom-emb' if custom else None)
    faster = types.ModuleType('faster_whisper')
    faster.WhisperModel = type('WhisperModel', (), {})
    monkeypatch.setitem(sys.modules, 'faster_whisper', faster)
    monkeypatch.setitem(sys.modules, backend.__name__, backend)
    monkeypatch.setitem(sys.modules, server.__name__, server)
    monkeypatch.setattr(sys, 'argv', ['server', '--diarization'])
    import huggingface_hub
    download = huggingface_hub.hf_hub_download
    try: exec(SERVER_BOOTSTRAP, {})
    finally: monkeypatch.setattr(huggingface_hub, 'hf_hub_download', download)
    assert captured == {'block_duration': .5, 'segmentation': 'custom-seg' if custom else 'default-seg',
                        'embedding': 'custom-emb' if custom else 'default-emb'}
    assert not scope['active']


def test_old_runtime_is_repaired_for_windows_tls_dependency(tmp_path):
    from app_paths import atomic_json
    from model_catalog import MODELS, WLK_VERSION
    from model_service import RuntimeManager
    runtime, model = RuntimeManager(tmp_path), MODELS[0]
    target = runtime.profile(model, True, 'cuda')
    target.mkdir()
    (target / 'python.exe').touch()
    atomic_json(target / 'ready.json', {'wlk': WLK_VERSION, 'device': 'cuda', 'diarization': True, 'family': model.family})
    assert not runtime.ready(model, True, 'cuda')
