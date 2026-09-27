"""Local control is a private Unix socket, never a laboratory inbound TCP port."""
from datetime import datetime
import json
import os
import signal
import socket
import stat
import struct
import threading
import time
from .config import NodeError
from .protocol import UnavailableBinding
from .runtime import P11Runtime
from .storage import Attempts, Instance, write_json
from .transport import backoff


class Service:
    def __init__(self, config, binding=None, runtime=None):
        self.config = config
        self.binding = binding or UnavailableBinding()
        self.runtime = runtime or P11Runtime()
        self.paused = False
        self.stopping = threading.Event()
        self.abort = threading.Event()
        self.active = None
        self.reason = 'starting'
        self.lock = threading.Lock()
        self.started = time.time()

    def status(self):
        with self.lock:
            return {'schema': 'ptb-node-status/1', 'pid': os.getpid(), 'started': self.started,
                    'paused': self.paused, 'stopping': self.stopping.is_set(),
                    'active': self.active is not None, 'reason': self.reason,
                    'capabilities': self.runtime.capabilities(),
                    'parallelism': 1, 'cpu_percent': self.config.cpu_percent,
                    'memory_bytes': self.config.memory_bytes}

    def control(self, command):
        with self.lock:
            if command == 'pause':
                self.paused = True
            elif command == 'resume':
                self.paused = False
            elif command == 'abort':
                self.abort.set()
            elif command == 'stop':
                self.paused = True
                self.abort.set()
                self.stopping.set()
            elif command != 'status':
                raise NodeError('control_invalid')
        return self.status()

    def run(self):
        with Instance(self.config.state_dir):
            attempts = Attempts(self.config.state_dir, self.config.temporary_bytes, self.config.reserve_disk_bytes)
            # Lock first. Unknown/unowned residue or cleanup failure blocks startup.
            attempts.recover()
            path = self.config.state_dir / 'control.sock'
            if path.exists() or path.is_symlink():
                if not stat.S_ISSOCK(path.lstat().st_mode):
                    raise NodeError('control_path_invalid')
                path.unlink()  # Proven stale by exclusive instance lock.
            listener = socket.socket(socket.AF_UNIX)
            listener.bind(str(path))
            os.chmod(path, 0o600)
            listener.listen(4)
            listener.settimeout(0.2)
            worker = threading.Thread(target=self._work, daemon=True)
            worker.start()
            previous = {}
            if threading.current_thread() is threading.main_thread():
                for sig in (signal.SIGTERM, signal.SIGINT):
                    previous[sig] = signal.signal(sig, lambda *_: self.control('stop'))
            try:
                while not self.stopping.is_set():
                    try:
                        connection, _ = listener.accept()
                    except socket.timeout:
                        continue
                    with connection:
                        connection.settimeout(0.5)
                        pid, uid, gid = struct.unpack('3i', connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
                        if uid != os.getuid():
                            continue
                        try:
                            data = connection.recv(64).decode('ascii').strip()
                            reply = self.control(data)
                            connection.sendall(json.dumps(reply).encode('utf-8'))
                        except (OSError, UnicodeError, NodeError):
                            pass
            finally:
                self.control('stop')
                worker.join(timeout=3)
                listener.close()
                path.unlink(missing_ok=True)
                for sig, handler in previous.items():
                    signal.signal(sig, handler)
                self.reason = 'stopped'
                write_json(self.config.state_dir / 'last-status.json', self.status())

    def _work(self):
        failures = 0
        try:
            while not self.stopping.is_set():
                try:
                    self.binding.health(self.runtime.capabilities())
                    failures = 0
                    with self.lock:
                        accept = not self.paused and self.config.in_hours(datetime.now())
                    caps = self.runtime.capabilities()
                    if not caps['modules']:
                        self.reason = caps['reason']
                    elif not accept:
                        self.reason = 'paused'
                    else:
                        # This gate stays closed until B/P11 integration is reviewed.
                        # It is deliberately impossible to launch an arbitrary manifest.
                        self.reason = 'attempt_binding_unavailable'
                    self.stopping.wait(1)
                except NodeError as exc:
                    allowed = {'protocol_unavailable', 'credential_rejected', 'version_mismatch', 'network_unavailable'}
                    self.reason = str(exc) if str(exc) in allowed else 'binding_failed'
                    failures += 1
                    self.stopping.wait(backoff(failures))
        finally:
            try:
                self.binding.goodbye()
            except Exception:
                pass  # Lease expiry remains authoritative after lost shutdown notification.


def control(config, command):
    # Require owner-only state before connecting to a possibly forged socket.
    info = config.state_dir.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_mode & 0o077 or info.st_uid != os.getuid():
        raise NodeError('state_permissions')
    path = config.state_dir / 'control.sock'
    if path.is_symlink() or not stat.S_ISSOCK(path.lstat().st_mode):
        raise NodeError('node_not_running')
    try:
        with socket.socket(socket.AF_UNIX) as sock:
            sock.settimeout(2)
            sock.connect(str(path))
            sock.sendall(command.encode('ascii') + b'\n')
            return json.loads(sock.recv(16384))
    except (OSError, ValueError):
        raise NodeError('node_not_running') from None
