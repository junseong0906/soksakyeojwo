import json
import queue
import threading

import pytest
from websockets.exceptions import ConnectionClosed
from websockets.sync.server import serve

from workers.asr_client_worker import PcmSession, _run_client


def send_config(socket):
    socket.send(json.dumps({'type': 'config', 'useAudioWorklet': True}))


def test_server_error_with_snapshot_fields_closes_once_without_sending_eof():
    received = []

    def handle(socket):
        send_config(socket)
        received.append(socket.recv(timeout=4))
        socket.send(json.dumps({'status': 'error', 'error': 'Unable to format transcription results.',
                                'lines': [], 'buffer_transcription': '', 'buffer_diarization': ''}))
        try:
            received.append(socket.recv(timeout=4))
        except ConnectionClosed:
            pass

    with serve(handle, '127.0.0.1', 0) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        url = f'ws://127.0.0.1:{server.socket.getsockname()[1]}/asr'
        events = queue.Queue()
        session = PcmSession(events)
        session.open(url)
        session.send(b'pcm')
        assert session.done.wait(4)
        session.close(timeout=1)
        session.close(timeout=1)
        server.shutdown()

    assert received == [b'pcm']
    assert list(events.queue)[1:] == [
        {'type': 'asr_error', 'message': 'Unable to format transcription results.'},
        {'type': 'stream_closed'},
    ]


@pytest.mark.parametrize('second_connection_works', [True, False])
def test_worker_retries_transient_close_at_most_once(second_connection_works):
    streams = []

    def handle(socket):
        frames = []
        streams.append(frames)
        send_config(socket)
        frames.append(socket.recv(timeout=4))
        if len(streams) == 1 or not second_connection_works:
            socket.close(code=1012, reason='temporary restart')
        else:
            frames.append(socket.recv(timeout=4))
            socket.send(json.dumps({'type': 'ready_to_stop'}))

    with serve(handle, '127.0.0.1', 0) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        events, commands = queue.Queue(), queue.Queue()
        client = threading.Thread(target=_run_client, args=(commands, events), daemon=True)
        client.start()
        try:
            url = f'ws://127.0.0.1:{server.socket.getsockname()[1]}/asr'
            commands.put({'type': 'start_stream', 'url': url})
            assert events.get(timeout=5)['type'] == 'stream_opened'
            commands.put({'type': 'audio_chunk', 'pcm_bytes': b'before'})
            assert events.get(timeout=5)['type'] == 'stream_reconnecting'
            assert events.get(timeout=5)['type'] == 'stream_reopened'
            commands.put({'type': 'audio_chunk', 'pcm_bytes': b'after'})
            if second_connection_works:
                commands.put({'type': 'pause_stream'})
                expected = ['stream_draining', 'transcript_finalized', 'stream_closed']
            else:
                expected = ['asr_error', 'stream_closed']
            assert [events.get(timeout=5)['type'] for _ in expected] == expected
        finally:
            commands.put({'type': 'shutdown'})
            client.join(timeout=5)
            server.shutdown()

    assert streams == [[b'before'], [b'after', b''] if second_connection_works else [b'after']]
    assert not client.is_alive()
    assert events.empty()
