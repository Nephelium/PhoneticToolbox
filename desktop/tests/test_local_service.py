from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import threading
import time
import urllib.error
import urllib.request

import pytest
from ptb_desktop.local_service import LocalService


def test_cleanup_waits_for_a_response_beyond_the_normal_five_second_timeout():
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            assert self.path == '/api/v1/jobs/local-storage/cleanup'
            time.sleep(5.2)
            raw = json.dumps({'count': 12, 'bytes': 1024, 'complete': True}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    service = LocalService()
    service.url = f'http://127.0.0.1:{server.server_port}'
    try:
        assert service.request('/api/v1/jobs/local-storage/cleanup', 'POST') == {
            'count': 12, 'bytes': 1024, 'complete': True,
        }
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_storage_status_keeps_the_existing_five_second_timeout(monkeypatch):
    class Opener:
        def open(self, request, *, timeout):
            assert request.full_url.endswith('/api/v1/jobs/local-storage')
            assert timeout == 5
            return io.BytesIO(b'{"result_count": 72}')

    monkeypatch.setattr(urllib.request, 'build_opener', lambda *args: Opener())
    service = LocalService()
    service.url = 'http://127.0.0.1:12345'
    assert service.get('/api/v1/jobs/local-storage') == {'result_count': 72}


def test_two_owned_instances_have_separate_sessions_and_clean_exit():
    first, second = LocalService(), LocalService()
    try:
        assert first.start()['status'] == 'ok'
        assert second.start()['status'] == 'ok'
        assert first.url != second.url
        assert first.token != second.token
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        for headers in [{}, {'Authorization': 'Bearer ' + second.token},
                        {'Authorization': 'Bearer ' + first.token, 'Origin': 'https://external.invalid'},
                        {'Authorization': 'Bearer ' + first.token, 'Host': 'external.invalid'}]:
            request = urllib.request.Request(first.url + '/api/v1/health', headers=headers)
            with pytest.raises(urllib.error.HTTPError) as error:
                opener.open(request, timeout=3)
            assert error.value.code == 403
        assert first.get('/api/v1/capabilities')['algorithms'] == []
        first.close()
        assert first.exit_code == 0
        assert second.get('/api/v1/health')['status'] == 'ok'
    finally:
        first.close()
        second.close()
    assert second.exit_code == 0
