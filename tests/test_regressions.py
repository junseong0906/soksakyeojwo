import json
import queue
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from model_catalog import MODELS
from model_service import ModelStore, RuntimeManager, Cancelled, wanted_file
from app_paths import SettingsStore, atomic_json
from speaker_labeler import SpeakerLabeler
from transcript_store import TranscriptStore
from workers.asr_client_worker import WlkMessageProcessor, PcmSession
from workers.recorder_worker import Recorder


def line(text, start=0, end=1000, speaker='spk_1'):
    return dict(text=text, start_ms=start, end_ms=end, speaker_id=speaker, is_final=True)


def test_growing_snapshot_replaces_line():
    parser, transcript = WlkMessageProcessor(), TranscriptStore()
    for text, end in [('오늘은', 1), ('오늘은 강의를 시작합니다', 3)]:
        event = parser.process(json.dumps({'lines': [{'text': text, 'start': 0, 'end': end, 'speaker': 1}]}))[0]
        transcript.update(event)
    assert len(transcript.lines) == 1
    assert transcript.lines[0]['text'] == '오늘은 강의를 시작합니다'


def test_speaker_revision_updates_instead_of_duplicate():
    transcript = TranscriptStore()
    transcript.update({'lines': [line('질문', speaker='spk_1')]})
    transcript.update({'lines': [line('질문', speaker='spk_2')]})
    assert [item['speaker_id'] for item in transcript.lines] == ['spk_2']


def test_pruned_window_preserves_old_history():
    transcript = TranscriptStore()
    transcript.update({'lines': [line('A'), line('B', 2000, 3000)]})
    transcript.update({'lines': [line('B expanded', 2000, 4000), line('C', 5000, 6000)]})
    assert [item['text'] for item in transcript.lines] == ['A', 'B expanded', 'C']


def test_diarization_buffer_retained_and_cleared():
    parser = WlkMessageProcessor()
    transcript = TranscriptStore()
    transcript.update(parser.process(json.dumps({'lines': [], 'buffer_diarization': '화자 확인 중'}))[0])
    assert transcript.diarization_buffer == '화자 확인 중'
    transcript.update(parser.process(json.dumps({'lines': [{'text': '화자 확인 중', 'speaker': 2}]}))[0])
    assert transcript.diarization_buffer == ''
    assert len(transcript.lines) == 1


def test_speaker_id_normalized_for_names():
    labeler = SpeakerLabeler()
    labeler.set_custom_name('1', '교수님')
    assert labeler.apply({'speaker_id': 'spk_1'})['speaker_label'] == '교수님'


@pytest.mark.parametrize('model', MODELS)
def test_model_arguments_include_real_backend_pcm_and_diarization(model, tmp_path):
    args = model.arguments(tmp_path, True, 'cpu')
    assert args[args.index('--backend') + 1] == model.backend
    assert '--pcm-input' in args and '--diarization' in args
    assert args[args.index('--diarization-backend') + 1] == 'diart'
    if model.family == 'qwen': assert 'windowed' in args
    if model.family == 'whisper': assert 'localagreement' in args


def test_incomplete_download_never_installed(tmp_path):
    store, model = ModelStore(tmp_path), MODELS[0]
    path = store.path(model)
    path.mkdir()
    (path / 'model.bin.part').write_bytes(b'abc')
    assert not store.installed(model)
    atomic_json(path / 'complete.json', {'repo': model.repo, 'files': {'model.bin': 5}})
    (path / 'model.bin').write_bytes(b'abc')
    assert not store.installed(model)
    (path / 'model.bin').write_bytes(b'abcde')
    assert store.installed(model)


def test_weight_format_selection_avoids_duplicate_voxtral_weights():
    assert wanted_file('model-00001-of-00002.safetensors', 'voxtral')
    assert not wanted_file('consolidated.safetensors', 'voxtral')
    assert not wanted_file('pytorch_model.bin', 'qwen')
    assert wanted_file('model.bin', 'whisper')


def test_runtime_profiles_are_separate(tmp_path):
    runtime = RuntimeManager(tmp_path)
    assert len({runtime.profile(model, False) for model in MODELS}) == 3
    assert runtime.profile(MODELS[0], False) != runtime.profile(MODELS[0], True)
    assert runtime.profile(MODELS[0], False, 'cpu') != runtime.profile(MODELS[0], False, 'cuda')
    assert runtime.profile(MODELS[0], False, 'cpu') == runtime.profile(MODELS[0], False, 'auto')


def test_settings_persist_but_never_contain_token(tmp_path):
    store = SettingsStore(tmp_path / 'settings.json')
    store.save(model_id='qwen-0.6b', diarization=True, speaker_labels={'spk_1': '교수님'})
    restored = SettingsStore(store.path)
    assert restored.value['model_id'] == 'qwen-0.6b'
    assert restored.value['diarization'] is True
    assert 'token' not in restored.value


def test_real_websocket_drains_final_results_before_close():
    from websockets.sync.server import serve
    received = []
    def handle(socket):
        socket.send(json.dumps({'type': 'config', 'useAudioWorklet': True}))
        while True:
            audio = socket.recv()
            received.append(audio)
            if audio == b'': break
        socket.send(json.dumps({'lines': [{'text': '마지막 자막', 'speaker': 1, 'start': 0, 'end': 1}]}))
        socket.send(json.dumps({'type': 'ready_to_stop'}))
    with serve(handle, '127.0.0.1', 0) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = server.socket.getsockname()[1]
        events = queue.Queue()
        session = PcmSession(events)
        session.open(f'ws://127.0.0.1:{port}/asr')
        session.send(b'first pcm')
        session.send(b'tail pcm')
        session.close(timeout=3)
        server.shutdown()
    output = list(events.queue)
    kinds = [event['type'] for event in output]
    assert received == [b'first pcm', b'tail pcm', b'']
    assert kinds.index('transcript_snapshot') < kinds.index('transcript_finalized') < kinds.index('stream_closed')
    assert 'asr_error' not in kinds and 'asr_warning' not in kinds


def test_switching_mic_writes_second_file_at_new_sample_rate(monkeypatch, tmp_path):
    import workers.recorder_worker as module
    class Input:
        def __init__(self, **kwargs): pass
        def start(self): pass
        def stop(self): pass
        def close(self): pass
    monkeypatch.setattr(module.sd, 'InputStream', Input)
    monkeypatch.setattr(module.sd, 'query_devices', lambda device, _: {'default_samplerate': 16000 if device == 0 else 48000})
    events, asr = queue.Queue(), queue.Queue()
    recorder = Recorder(events, asr, tmp_path)
    recorder.select_device(0)
    recorder.start_recording()
    first = recorder.path
    recorder.callback(np.ones((1600, 1), dtype=np.float32) * .1, 1600, None, None)
    recorder.select_device(1)
    second = recorder.path
    assert recorder.file is not None and not recorder.file.closed
    recorder.callback(np.ones((4800, 1), dtype=np.float32) * .2, 4800, None, None)
    recorder.close()
    a, sr_a = sf.read(first)
    b, sr_b = sf.read(second)
    assert sr_a == 16000 and sr_b == 48000
    assert len(a) == 1600 and len(b) == 4800
    assert first != second


def test_asr_stop_fence_keeps_audio_before_stop_and_recording_alive(monkeypatch, tmp_path):
    import workers.recorder_worker as module
    monkeypatch.setattr(module.sd, 'query_devices', lambda *_: {'default_samplerate': 16000})
    class Input:
        def __init__(self, **kwargs): pass
        def start(self): pass
        def stop(self): pass
        def close(self): pass
    monkeypatch.setattr(module.sd, 'InputStream', Input)
    asr = queue.Queue()
    recorder = Recorder(queue.Queue(), asr, tmp_path)
    recorder.start_recording()
    recorder.enable_asr()
    recorder.callback(np.ones((1600, 1), dtype=np.float32), 1600, None, None)
    recorder.stop_asr()
    recorder.callback(np.ones((1600, 1), dtype=np.float32), 1600, None, None)
    assert recorder.file is not None
    assert [msg['type'] for msg in list(asr.queue)] == ['audio_chunk', 'pause_stream']
    path = recorder.path
    recorder.close()
    assert sf.info(path).frames == 3200


@pytest.fixture
def window(monkeypatch, tmp_path):
    monkeypatch.setenv('SOKSAK_DATA_DIR', str(tmp_path))
    from PySide6.QtWidgets import QApplication
    from ui.main_window import MainWindow
    app = QApplication.instance() or QApplication([])
    ui = MainWindow(start_workers=False)
    yield ui
    if ui.model_dialog: ui.model_dialog.close()
    ui.close()
    app.processEvents()


def test_final_partial_final_keeps_both_sentences(window):
    window._apply_incoming_subtitle('문장 A', True)
    window._apply_incoming_subtitle('문장 B 인식 중', False)
    window._apply_incoming_subtitle('문장 B', True)
    assert len(window.history_entries) == 2
    assert '문장 A' in window.history_entries[0]
    assert '문장 B' in window.history_entries[1]


def test_ui_snapshot_growth_does_not_duplicate_history(window):
    window.transcript_store.update({'lines': [line('오늘은')]})
    window._render_snapshot()
    window.transcript_store.update({'lines': [line('오늘은 강의를 시작합니다', end=3000)]})
    window._render_snapshot()
    assert len(window.history_entries) == 1
    assert '오늘은 강의를 시작합니다' in window.history_browser.toPlainText()


def test_speaker_name_button_saves_and_updates_existing_subtitles(window, monkeypatch):
    from PySide6.QtWidgets import QPushButton
    from ui.main_window import SpeakerLabelDialog

    window.transcript_store.update({'lines': [line('강의를 시작합니다')]})
    window._render_snapshot()
    assert '[화자 1]' in window.history_entries[0]

    def enter_name(dialog):
        dialog.speaker_id_input.setText('1')
        dialog.display_name_input.setText('교수님')
        next(button for button in dialog.findChildren(QPushButton) if button.text() == '적용').click()
        return dialog.reject()

    monkeypatch.setattr(SpeakerLabelDialog, 'exec', enter_name)
    window.speaker_btn.click()

    saved = SettingsStore(window.controller.settings.path)
    assert saved.value['speaker_labels'] == {'spk_1': '교수님'}
    assert len(window.history_entries) == 1
    assert '[교수님] 강의를 시작합니다' in window.history_entries[0]
    assert '[교수님] 강의를 시작합니다' in window.current_subtitle_box.toPlainText()


def test_dialog_lists_all_seven_models_and_disables_apply_until_download(window):
    window.open_model_settings()
    dialog = window.model_dialog
    assert dialog.table.rowCount() == 7
    assert not dialog.apply.isEnabled()
    dialog.table.selectRow(6)
    assert dialog.selected().id == 'voxtral-4b'


def test_subtitle_text_is_html_escaped(window):
    window.current_subtitle_box.set_subtitle_text('<script> & 수업', 28)
    assert '<script> & 수업' in window.current_subtitle_box.toPlainText()
    window.append_history_line('<x> 부등식 & 조건')
    assert '<x> 부등식 & 조건' in window.history_browser.toPlainText()


def test_download_cancel_and_resume_preserve_partial(monkeypatch, tmp_path):
    import huggingface_hub
    import requests
    info = SimpleNamespace(sha='fixed-revision', siblings=[SimpleNamespace(rfilename='model.bin', size=8, lfs=None)])
    monkeypatch.setattr(huggingface_hub, 'HfApi', lambda **kwargs: SimpleNamespace(model_info=lambda *a, **kw: info))
    calls = []
    class Response:
        def __init__(self, headers):
            self.status_code = 206 if headers.get('Range') else 200
            self.headers = {'Content-Range': 'bytes 4-7/8'} if self.status_code == 206 else {}
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def raise_for_status(self): pass
        def iter_content(self, _):
            if self.status_code == 206: yield b'efgh'
            else: yield b'abcd'; yield b'efgh'
    def get(url, **kwargs):
        calls.append(kwargs['headers'])
        return Response(kwargs['headers'])
    monkeypatch.setattr(requests, 'get', get)
    cancel = threading.Event()
    def progress(percent, text):
        if percent > 0: cancel.set()
    store = ModelStore(tmp_path)
    with pytest.raises(Cancelled): store.download(MODELS[0], progress, cancel)
    assert not store.installed(MODELS[0])
    assert (store.path(MODELS[0]) / 'model.bin.part').read_bytes() == b'abcd'
    store.download(MODELS[0], lambda *_: None, threading.Event())
    assert calls[-1]['Range'] == 'bytes=4-'
    assert store.installed(MODELS[0])
    assert (store.path(MODELS[0]) / 'model.bin').read_bytes() == b'abcdefgh'


def test_complete_partial_file_promoted_without_invalid_range(monkeypatch, tmp_path):
    import huggingface_hub
    import requests
    info = SimpleNamespace(sha='fixed', siblings=[SimpleNamespace(rfilename='model.bin', size=4, lfs=None)])
    monkeypatch.setattr(huggingface_hub, 'HfApi', lambda **kw: SimpleNamespace(model_info=lambda *a, **kw: info))
    monkeypatch.setattr(requests, 'get', lambda *a, **kw: pytest.fail('Completed partial should not be fetched'))
    store = ModelStore(tmp_path)
    store.path(MODELS[0]).mkdir()
    (store.path(MODELS[0]) / 'model.bin.part').write_bytes(b'done')
    store.download(MODELS[0], lambda *_: None, threading.Event())
    assert store.installed(MODELS[0])


def test_device_override_applies_before_wlk_import(monkeypatch):
    import sys
    import types
    from wlk_server_manager import SERVER_BOOTSTRAP
    captured = {}
    class Original:
        def __init__(self, *args, **kwargs): captured.update(kwargs)
    faster = types.ModuleType('faster_whisper')
    faster.WhisperModel = Original
    server = types.ModuleType('whisperlivekit.basic_server')
    server.main = lambda: faster.WhisperModel('tiny', device='auto', compute_type='auto')
    monkeypatch.setitem(sys.modules, 'faster_whisper', faster)
    monkeypatch.setitem(sys.modules, 'whisperlivekit.basic_server', server)
    monkeypatch.setenv('SOKSAK_DEVICE', 'cpu')
    import huggingface_hub
    original_download = huggingface_hub.hf_hub_download
    try: exec(SERVER_BOOTSTRAP, {})
    finally: monkeypatch.setattr(huggingface_hub, 'hf_hub_download', original_download)
    assert captured['device'] == 'cpu' and captured['compute_type'] == 'int8'


def pump_until(predicate, timeout=5):
    import time
    from PySide6.QtWidgets import QApplication
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(.01)
    assert predicate()


def test_model_selection_persisted_only_after_engine_ready(window, monkeypatch):
    controller = window.controller
    monkeypatch.setattr(controller.models, 'installed', lambda model: True)
    monkeypatch.setattr(controller.runtimes, 'prepare', lambda *args: 'python.exe')
    states = []
    def start(*args):
        states.append(controller.settings.value['model_id'])
        controller.server.port = 30000
    monkeypatch.setattr(controller.server, 'start', start)
    assert controller.apply('qwen-0.6b', False, 'cpu')
    pump_until(lambda: not controller.busy)
    assert states == ['whisper-tiny']
    assert controller.active_id == 'qwen-0.6b'
    assert controller.settings.value['model_id'] == 'qwen-0.6b'


def test_failed_engine_apply_does_not_persist_new_selection(window, monkeypatch):
    controller = window.controller
    monkeypatch.setattr(controller.models, 'installed', lambda model: True)
    monkeypatch.setattr(controller.runtimes, 'prepare', lambda *args: 'python.exe')
    def fail(*args): raise RuntimeError('load failed')
    monkeypatch.setattr(controller.server, 'start', fail)
    assert controller.apply('voxtral-4b', False, 'cpu')
    pump_until(lambda: not controller.busy)
    assert controller.active_id is None
    assert controller.settings.value['model_id'] == 'whisper-tiny'
    assert 'load failed' in controller.status


def test_background_operation_cancellation_returns_to_idle(window):
    from model_service import check_cancel
    controller = window.controller
    def action(progress, cancel):
        cancel.wait(3)
        check_cancel(cancel)
    controller.run_task(action)
    controller.stop_task()
    pump_until(lambda: not controller.busy)
    assert '중지' in controller.status


def test_user_can_clear_previously_invalid_token(window, monkeypatch):
    controller = window.controller
    controller.token = 'invalid-token'
    monkeypatch.setattr(controller, 'run_task', lambda *args, **kwargs: True)
    controller.download('whisper-tiny', '')
    assert controller.token == ''


def test_each_model_has_its_own_download_button(window, monkeypatch):
    requested = []
    monkeypatch.setattr(window.controller, 'download', lambda model_id, token='': requested.append(model_id))
    window.open_model_settings()
    dialog = window.model_dialog
    assert len(dialog.download_buttons) == 7
    assert len(dialog.row_progress) == 7
    for i, button in enumerate(dialog.download_buttons):
        button.click()
        assert requested[-1] == MODELS[i].id
    assert len(requested) == 7


def test_model_checkbox_selection_is_exclusive(window):
    window.open_model_settings()
    dialog = window.model_dialog
    for i, check in enumerate(dialog.model_checks):
        check.click()
        assert sum(item.isChecked() for item in dialog.model_checks) == 1
        assert dialog.selected().id == MODELS[i].id


def test_apply_button_uses_checked_model(window, monkeypatch):
    monkeypatch.setattr(window.controller.models, 'installed', lambda model: True)
    window.open_model_settings()
    dialog = window.model_dialog
    requested = []
    dialog.applyRequested.disconnect()
    dialog.applyRequested.connect(lambda model_id, *args: requested.append(model_id))
    dialog.model_checks[4].click()
    dialog.apply.click()
    assert requested == ['whisper-turbo']


def test_download_does_not_change_apply_selection(window, monkeypatch):
    monkeypatch.setattr(window.controller, 'download', lambda *args: True)
    window.open_model_settings()
    dialog = window.model_dialog
    dialog.model_checks[1].click()
    dialog.download_buttons[5].click()
    assert dialog.selected().id == 'whisper-small'


def test_model_progress_updates_only_its_row(window):
    window.open_model_settings()
    controller = window.controller
    controller.task_kind = 'download'
    controller.task_model_id = MODELS[5].id
    controller._progress(42, '모델 파일을 받는 중')
    dialog = window.model_dialog
    assert dialog.row_progress[5].value() == 42
    assert all(dialog.row_progress[i].value() == 0 for i in range(7) if i != 5)


def test_tls_client_context_keeps_certificate_and_hostname_verification():
    import ssl
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname is True
