import json
import queue
import socket
import threading

import pytest
from websockets.exceptions import ConnectionClosed
from websockets.sync.server import serve

from local_engine import LocalEngineEndpoint
from workers.asr_client_worker import AudioSocket, PcmSession, _run_client


def send_config(socket):
    socket.send(json.dumps({'type': 'config', 'useAudioWorklet': True}))


def test_server_error_with_snapshot_fields_closes_once_without_sending_eof(monkeypatch):
    received = []
    monkeypatch.setenv('HTTP_PROXY', 'http://127.0.0.1:1')
    monkeypatch.setenv('NO_PROXY', '')

    def handle(socket):
        assert socket.request.headers['Authorization'] == 'Bearer test-engine-key'
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
        endpoint = LocalEngineEndpoint(server.socket.getsockname()[1], 'test-engine-key')
        events = queue.Queue()
        session = PcmSession(events)
        session.open(endpoint)
        session.send(b'pcm')
        assert session.done.wait(4)
        session.close(timeout=1)
        session.close(timeout=1)
        server.shutdown()

    assert received == [b'pcm']
    assert list(events.queue)[1:] == [
        {'type': 'asr_error', 'message': 'Unable to format transcription results.'},
        {'type': 'stream_closed', 'retryable': False},
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
            endpoint = LocalEngineEndpoint(server.socket.getsockname()[1], 'test-engine-key')
            commands.put({'type': 'start_stream', 'endpoint': endpoint})
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


def test_worker_rejects_external_address_before_opening_a_socket(monkeypatch):
    def unexpected_connect(*args, **kwargs):
        pytest.fail('worker created a WebSocket for an external address')

    monkeypatch.setattr('workers.asr_client_worker.websocket.WebSocket', unexpected_connect)
    commands, events = queue.Queue(), queue.Queue()
    commands.put({'type': 'start_stream', 'url': 'ws://example.invalid/asr'})
    commands.put({'type': 'start_stream', 'endpoint': 'ws://example.invalid/asr'})
    commands.put({'type': 'shutdown'})
    _run_client(commands, events)

    for _ in range(2):
        assert events.get_nowait()['type'] == 'asr_error'
        assert events.get_nowait() == {'type': 'stream_closed', 'retryable': False}
    assert events.empty()


def test_pcm_write_timeout_after_partial_send_keeps_the_connection(monkeypatch):
    received = []
    attempts = []
    winsock_send = socket.socket.send

    def write_with_temporary_backpressure(sock, data, flags=0):
        if isinstance(sock, AudioSocket) and sock.streaming and len(attempts) < 2:
            attempts.append(bytes(data))
            if len(attempts) == 1:
                return winsock_send(sock, data[:4], flags)
            raise TimeoutError('timed out while waiting for the local engine to read')
        return winsock_send(sock, data, flags)

    def handle(peer):
        send_config(peer)
        received.extend((peer.recv(timeout=4), peer.recv(timeout=4)))
        peer.send(json.dumps({'type': 'ready_to_stop'}))

    monkeypatch.setattr(socket.socket, 'send', write_with_temporary_backpressure)
    pcm = bytes(range(64))
    with serve(handle, '127.0.0.1', 0) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        endpoint = LocalEngineEndpoint(server.socket.getsockname()[1], 'test-engine-key')
        events = queue.Queue()
        session = PcmSession(events)
        session.open(endpoint)
        session.send(pcm)
        session.close(timeout=4)
        server.shutdown()

    assert attempts[1] == attempts[0][4:]
    assert received == [pcm, b'']
    assert [event['type'] for event in events.queue] == [
        'stream_opened', 'stream_delayed', 'stream_delivery_resumed',
        'stream_draining', 'transcript_finalized', 'stream_closed',
    ]


def test_persistent_backpressure_requests_engine_restart_without_reopening(monkeypatch):
    connections = []
    winsock_send = socket.socket.send

    def write_while_engine_stalled(sock, data, flags=0):
        if isinstance(sock, AudioSocket) and sock.streaming:
            raise TimeoutError('timed out')
        return winsock_send(sock, data, flags)

    def handle(peer):
        connections.append(peer)
        send_config(peer)
        with pytest.raises(ConnectionClosed):
            peer.recv(timeout=4)

    monkeypatch.setattr(socket.socket, 'send', write_while_engine_stalled)
    monkeypatch.setattr('workers.asr_client_worker.AUDIO_WRITE_TIMEOUT', 0)
    with serve(handle, '127.0.0.1', 0) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        endpoint = LocalEngineEndpoint(server.socket.getsockname()[1], 'test-engine-key')
        commands, events = queue.Queue(), queue.Queue()
        client = threading.Thread(target=_run_client, args=(commands, events), daemon=True)
        client.start()
        try:
            commands.put({'type': 'start_stream', 'endpoint': endpoint})
            assert events.get(timeout=5)['type'] == 'stream_opened'
            commands.put({'type': 'audio_chunk', 'pcm_bytes': b'pcm'})
            assert events.get(timeout=5)['type'] == 'asr_error'
            assert events.get(timeout=5) == {'type': 'stream_closed', 'retryable': True}
        finally:
            commands.put({'type': 'shutdown'})
            client.join(timeout=5)
            server.shutdown()

    assert len(connections) == 1
    assert not client.is_alive()
