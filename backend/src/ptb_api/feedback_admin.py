"""Revocable manager devices, separate from feedback SQLite and research data."""
from contextlib import contextmanager
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import tempfile
import threading

_locks = {}
_locks_guard = threading.Lock()


class AdminDevices:
    def __init__(self, path, *, secret, clock, error):
        self.path = Path(path)
        self.secret, self.clock, self.error = secret, clock, error
        with _locks_guard:
            self.lock = _locks.setdefault(str(self.path.resolve()), threading.RLock())

    def digest(self, value):
        return hmac.new(self.secret, value.encode(), hashlib.sha256).hexdigest()

    @contextmanager
    def transaction(self, *, write=False):
        with self.lock:
            # Linux production may also be accessed by a maintenance process.
            handle = None
            if os.name != 'nt' and (write or self.path.exists()):
                import fcntl
                handle = self.path.with_suffix('.lock').open('a+b')
                os.fchmod(handle.fileno(), 0o600)
                fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                state = json.loads(self.path.read_text(encoding='utf-8')) if self.path.exists() else {'version': 1, 'devices': [], 'pairing': None}
                if state.get('version') != 1 or not isinstance(state.get('devices'), list):
                    raise ValueError('Unknown manager-device store')
                yield state
                if write:
                    self.save(state)
            finally:
                if handle:
                    handle.close()

    def save(self, state):
        descriptor, name = tempfile.mkstemp(prefix='.feedback-admin-', dir=self.path.parent)
        try:
            if os.name != 'nt':
                os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, 'w', encoding='utf-8', newline='\n') as stream:
                json.dump(state, stream, ensure_ascii=False, separators=(',', ':'))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, self.path)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def authenticate(self, token):
        if not isinstance(token, str) or not 32 <= len(token) <= 128:
            return False
        digest = self.digest(token)
        with self.transaction() as state:
            return any(hmac.compare_digest(digest, device['hash']) for device in state['devices'])

    def start_pairing(self):
        code = f'{secrets.randbelow(100_000_000):08d}'
        now = self.clock()
        with self.transaction(write=True) as state:
            recent = [stamp for stamp in state.get('pairingStarts', []) if stamp > now-3600]
            if len(recent) >= 10:
                raise self.error('pairing_rate_limited', 429)
            if len(state['devices']) >= 8:
                raise self.error('device_limit', 409)
            state['pairingStarts'] = recent + [now]
            state['pairing'] = {'hash': self.digest('pair:'+code), 'expires': now+300, 'attempts': 0}
        return {'code': code, 'expiresIn': 300}

    def complete_pairing(self, code, name):
        result = None
        with self.transaction(write=True) as state:
            pairing = state.get('pairing')
            if pairing and pairing['expires'] > self.clock() and pairing['attempts'] < 5:
                pairing['attempts'] += 1
                if hmac.compare_digest(pairing['hash'], self.digest('pair:'+code)) and len(state['devices']) < 8:
                    token = secrets.token_urlsafe(32)
                    device_id = secrets.token_urlsafe(18)
                    state['devices'].append({'id': device_id, 'name': name.strip() or '安卓管理端', 'hash': self.digest(token), 'created': self.clock()})
                    state['pairing'] = None
                    result = {'token': token, 'deviceId': device_id}
        if result is None:
            raise self.error('pairing_invalid', 401)
        return result

    def list_devices(self):
        with self.transaction() as state:
            return {'devices': [{key: device[key] for key in ('id', 'name', 'created')} for device in state['devices']]}

    def revoke(self, device_id):
        with self.transaction(write=True) as state:
            devices = [device for device in state['devices'] if device['id'] != device_id]
            if len(devices) == len(state['devices']):
                raise self.error('device_unavailable', 404)
            state['devices'] = devices
        return {'ok': True}
