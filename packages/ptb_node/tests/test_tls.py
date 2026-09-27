"""Real loopback TLS. Test-only routes do not describe B's API."""
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
from pathlib import Path
import socket
import ssl
import subprocess
import tempfile
import threading
import time
import unittest

from ptb_node.config import NodeError
from ptb_node.transport import HttpsTransport, download, upload


DATA = bytes(range(256)) * 4096


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.headers.get('Authorization') != 'Bearer synthetic-token':
            self.send_response(403)
            self.send_header('Content-Length', '0')
            self.end_headers()
            return
        if self.path == '/redirect':
            self.send_response(302)
            self.send_header('Location', 'https://example.invalid/steal')
            self.send_header('Content-Length', '0')
            self.end_headers()
            return
        if self.path == '/slow':
            self.send_response(200)
            self.send_header('Content-Length', '2')
            self.end_headers()
            self.wfile.write(b'a')
            self.wfile.flush()
            time.sleep(0.5)
            try:
                self.wfile.write(b'b')
            except OSError:
                pass
            return
        if self.path == '/oversize':
            self.send_response(200)
            self.send_header('Content-Length', '2000000')
            self.end_headers()
            return
        offset, count = map(int, self.path.removeprefix('/data/').split('/'))
        body = DATA[offset:offset+count]
        self.send_response(200)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_PUT(self):
        data = self.rfile.read(int(self.headers['Content-Length']))
        offset = int(self.path.removeprefix('/put/'))
        if self.headers['X-Block-SHA256'] != hashlib.sha256(data).hexdigest():
            self.send_response(400)
            self.send_header('Content-Length', '0')
            self.end_headers()
            return
        self.server.received[offset] = data
        # Lose first acknowledgement after storing: same offset is retried safely.
        if self.server.fail_once:
            self.server.fail_once = False
            self.connection.shutdown(socket.SHUT_RDWR)
            self.connection.close()
            return
        body = str(offset + len(data)).encode()
        self.send_response(200)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class TlsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        path = Path(cls.temp.name)
        cls.cert, cls.key = path / 'cert.pem', path / 'key.pem'
        subprocess.run(['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes',
                        '-keyout', str(cls.key), '-out', str(cls.cert), '-days', '1',
                        '-subj', '/CN=localhost', '-addext', 'subjectAltName=DNS:localhost'],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(cls.cert, cls.key)
        cls.server.socket = context.wrap_socket(cls.server.socket, server_side=True)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.origin = 'https://localhost:' + str(cls.server.server_port)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(2)
        cls.temp.cleanup()

    def transport(self, token='synthetic-token'):
        return HttpsTransport(self.origin, token, context=ssl.create_default_context(cafile=str(self.cert)))

    def test_real_https_download(self):
        transport = self.transport()
        target = io.BytesIO()
        download(target, size=len(DATA), sha256=hashlib.sha256(DATA).hexdigest(),
                 read_chunk=lambda offset, count: transport.request('GET', f'/data/{offset}/{count}')[2],
                 check=lambda: None, wait=lambda _: None)
        self.assertEqual(target.getvalue(), DATA)

    def test_real_https_upload_lost_ack(self):
        self.server.received, self.server.fail_once = {}, True
        transport = self.transport()
        def write(offset, data, sha):
            return int(transport.request('PUT', f'/put/{offset}', body=data, headers={'X-Block-SHA256': sha})[2])
        upload(io.BytesIO(DATA), size=len(DATA), sha256=hashlib.sha256(DATA).hexdigest(),
               write_chunk=write, check=lambda: None, wait=lambda _: None)
        self.assertEqual(b''.join(self.server.received[k] for k in sorted(self.server.received)), DATA)

    def test_redirect_forbidden(self):
        with self.assertRaisesRegex(NodeError, 'redirect_forbidden'):
            self.transport().request('GET', '/redirect')

    def test_revoked_identity(self):
        with self.assertRaisesRegex(NodeError, 'credential_rejected'):
            self.transport('revoked-synthetic-token').request('GET', '/data/0/2')

    def test_untrusted_certificate(self):
        with self.assertRaisesRegex(NodeError, 'network_unavailable'):
            HttpsTransport(self.origin, 'synthetic-token').request('GET', '/data/0/2')

    def test_disabling_tls_validation_rejected(self):
        with self.assertRaisesRegex(NodeError, 'tls_verification_required'):
            HttpsTransport(self.origin, 'synthetic-token', context=ssl._create_unverified_context())

    def test_response_cap(self):
        with self.assertRaisesRegex(NodeError, 'response_size_invalid'):
            self.transport().request('GET', '/oversize')

    def test_lease_interrupts_live_download(self):
        deadline = time.monotonic() + 0.15
        def check():
            if time.monotonic() >= deadline:
                raise NodeError('lease_lost')
        start = time.monotonic()
        with self.assertRaisesRegex(NodeError, 'lease_lost'):
            self.transport().request('GET', '/slow', check=check)
        self.assertLess(time.monotonic() - start, 0.45)

    def test_remote_url_and_header_injection_rejected(self):
        transport = self.transport()
        for route in ('https://example.invalid/data', '//example.invalid/data', '/data?token=bad'):
            with self.subTest(route=route), self.assertRaises(NodeError):
                transport.request('GET', route)
        with self.assertRaises(NodeError):
            transport.request('GET', '/data/0/2', headers={'Host': 'evil.invalid'})


if __name__ == '__main__':
    unittest.main()
