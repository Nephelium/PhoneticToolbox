"""Process-local HTTP(S) rejection for offline QA; never forwards a connection."""
from __future__ import annotations
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from urllib.parse import urlsplit


class RejectingProxy:
    def __init__(self, report: Path):
        self.report = report
        self.requests = []
        self.lock = threading.Lock()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def reject(self):
                target = self.path if self.command == 'CONNECT' else urlsplit(self.path).netloc
                with owner.lock:
                    owner.requests.append({'method': self.command, 'target': target, 'status': 502,
                                           'forwarded': False})
                self.send_response(502, 'Owned offline verification')
                self.send_header('Content-Length', '0')
                self.send_header('Connection', 'close')
                self.end_headers()
                self.close_connection = True
            do_CONNECT = reject
            do_GET = reject
            do_POST = reject
            do_HEAD = reject
            def log_message(self, *_args):
                pass

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       kwargs={'poll_interval': .1}, daemon=True)
        self.url = 'http://127.0.0.1:' + str(self.server.server_port)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(5)
        assert not self.thread.is_alive()
        self.report.write_text(json.dumps({'scope': 'Owned rejecting proxy only; no external socket forwarding',
                                           'requests': self.requests, 'forwardedConnections': 0}, indent=2) + '\n', encoding='utf-8')

    def environment(self):
        return {**{name: self.url for name in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY',
                                               'http_proxy', 'https_proxy', 'all_proxy')},
                'NO_PROXY': '127.0.0.1,localhost,::1', 'no_proxy': '127.0.0.1,localhost,::1'}
