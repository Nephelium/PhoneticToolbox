"""HTTPS primitives. Only a reviewed binding may select relative routes/headers."""
import hashlib
import http.client
import random
import socket
import ssl
import threading
from urllib.parse import urlsplit
from .config import NodeError, origin


def backoff(failures, rng=random.uniform):
    return rng(0.5, min(30.0, 0.5 * 2 ** min(max(failures, 1), 6)))


def retry(call, check, wait, attempts=5):
    for number in range(attempts):
        check()
        try:
            return call()
        except NodeError as exc:
            if str(exc) != 'network_unavailable' or number == attempts - 1:
                raise
            wait(backoff(number + 1))
    raise NodeError('network_unavailable')


class HttpsTransport:
    def __init__(self, server_origin, token, *, context=None, timeout=2.0):
        self.origin = urlsplit(origin(server_origin))
        self.token = token
        self.context = context or ssl.create_default_context()
        if not self.context.check_hostname or self.context.verify_mode != ssl.CERT_REQUIRED:
            raise NodeError('tls_verification_required')
        self.timeout = min(2.0, max(0.1, timeout))
        self.active = set()
        self.lock = threading.Lock()

    def abort(self):
        with self.lock:
            connections = tuple(self.active)
        for conn in connections:
            if conn.sock:
                try:
                    conn.sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            conn.close()

    def request(self, method, route, *, body=b'', headers=None, check=lambda: None, cap=1048576):
        if (method not in ('GET', 'POST', 'PUT') or not route.startswith('/')
                or route.startswith('//') or any(c in route for c in ('?', '#', '\\', '\r', '\n'))
                or len(body) > 1048576 or not 0 <= cap <= 1048576):
            raise NodeError('transport_request_invalid')
        supplied = headers or {}
        if any(k.lower() in ('authorization', 'host', 'content-length', 'transfer-encoding') for k in supplied):
            raise NodeError('transport_header_invalid')
        conn = http.client.HTTPSConnection(self.origin.hostname, self.origin.port or 443,
                                           context=self.context, timeout=self.timeout)
        closed = threading.Event()
        def watch():
            while not closed.wait(0.05):
                try:
                    check()
                except NodeError:
                    if conn.sock:
                        try:
                            conn.sock.shutdown(socket.SHUT_RDWR)
                        except OSError:
                            pass
                    conn.close()
                    return
        watcher = threading.Thread(target=watch, daemon=True)
        with self.lock:
            self.active.add(conn)
        try:
            check()
            watcher.start()
            conn.connect()
            check()  # A delayed DNS/TLS connection cannot send after lease expiry.
            conn.request(method, route, body=body,
                         headers={**supplied, 'Authorization': 'Bearer ' + self.token})
            response = conn.getresponse()
            if 300 <= response.status < 400:
                raise NodeError('redirect_forbidden')
            if response.status in (401, 403):
                raise NodeError('credential_rejected')
            if response.status in (409, 410):
                raise NodeError('attempt_rejected')
            if response.status == 429 or response.status >= 500:
                raise NodeError('network_unavailable')
            if not 200 <= response.status < 300:
                raise NodeError('remote_rejected')
            length = response.getheader('Content-Length')
            if length is None or not length.isdigit() or int(length) > cap:
                raise NodeError('response_size_invalid')
            output = bytearray()
            while len(output) < int(length):
                check()
                chunk = response.read(min(65536, int(length) - len(output)))
                if not chunk:
                    raise NodeError('network_unavailable')
                output.extend(chunk)
            check()
            return response.status, dict(response.getheaders()), bytes(output)
        except (OSError, http.client.HTTPException):
            check()
            raise NodeError('network_unavailable') from None
        finally:
            closed.set()
            conn.close()
            with self.lock:
                self.active.discard(conn)


def download(stream, *, size, sha256, read_chunk, check, wait, chunk_bytes=262144):
    """Binding maps asset ID/offset; no remote filename or URL enters this function."""
    if type(size) is not int or size < 0 or not 1 <= chunk_bytes <= 1048576:
        raise NodeError('asset_invalid')
    digest = hashlib.sha256()
    offset = 0
    while offset < size:
        check()
        count = min(chunk_bytes, size - offset)
        data = retry(lambda: read_chunk(offset, count), check, wait)
        if not isinstance(data, bytes) or len(data) != count:
            raise NodeError('asset_length_mismatch')
        check()
        stream.write(data)
        digest.update(data)
        offset += len(data)
    if digest.hexdigest() != sha256:
        raise NodeError('asset_hash_mismatch')
    check()


def upload(stream, *, size, sha256, write_chunk, check, wait, chunk_bytes=262144):
    """Fixed bounded blocks; retries use identical offset/content/hash.

    B binding must supply the frozen idempotency mapping and committed next offset.
    """
    if type(size) is not int or size < 0 or not 1 <= chunk_bytes <= 1048576:
        raise NodeError('output_invalid')
    digest = hashlib.sha256()
    offset = 0
    while offset < size:
        check()
        data = stream.read(min(chunk_bytes, size - offset))
        if not data:
            raise NodeError('output_length_mismatch')
        block_hash = hashlib.sha256(data).hexdigest()
        next_offset = retry(lambda: write_chunk(offset, data, block_hash), check, wait)
        if type(next_offset) is not int or next_offset != offset + len(data):
            raise NodeError('upload_offset_mismatch')
        digest.update(data)
        offset = next_offset
    if stream.read(1) or digest.hexdigest() != sha256:
        raise NodeError('output_hash_mismatch')
    check()
