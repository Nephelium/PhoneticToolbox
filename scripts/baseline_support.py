"""P03 fixture recipes and result comparison, independent of either scientific implementation."""
import gzip
import hashlib
import json
import math
import struct
import wave
from pathlib import Path

RECIPES = [
    {'id': 'SYN-VOWEL-44100', 'rate': 44100, 'frames': 35280, 'channels': 1, 'kind': 'harmonic', 'mode': 'acoustic'},
    {'id': 'SYN-MIXED-16000', 'rate': 16000, 'frames': 19200, 'channels': 1, 'kind': 'mixed', 'mode': 'acoustic'},
    {'id': 'SYN-SILENCE-16000', 'rate': 16000, 'frames': 8000, 'channels': 1, 'kind': 'silence', 'mode': 'acoustic'},
    {'id': 'SYN-SINE440-16000', 'rate': 16000, 'frames': 8000, 'channels': 1, 'kind': 'sine', 'mode': 'acoustic'},
    {'id': 'SYN-EGG-44100', 'rate': 44100, 'frames': 35280, 'channels': 2, 'kind': 'egg_shape', 'mode': 'egg'},
    {'id': 'SYN-ONE-FRAME', 'rate': 16000, 'frames': 1, 'channels': 1, 'kind': 'silence', 'mode': 'acoustic'},
    {'id': 'SYN-EMPTY', 'rate': 16000, 'frames': 0, 'channels': 1, 'kind': 'silence', 'mode': 'acoustic'},
    {'id': 'SYN-BROKEN', 'rate': None, 'frames': None, 'channels': None, 'kind': 'broken', 'mode': 'acoustic'},
]


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def create_fixture(folder: Path, recipe: dict) -> Path:
    """Original deterministic analytic test signals; no research recording synthesis claims."""
    path = folder / (recipe['id'] + '.wav')
    if recipe['kind'] == 'broken':
        path.write_bytes(b'P03 deliberately invalid WAV\n')
        return path
    payload = bytearray()
    rate, frames = recipe['rate'], recipe['frames']
    for i in range(frames):
        t = i / rate
        phase = 2 * math.pi * 120 * t
        harmonic = sum(math.sin(k * phase) / (k * k) for k in range(1, 13)) * 0.4
        value = {'harmonic': harmonic, 'egg_shape': harmonic, 'silence': 0,
                 'sine': 0.5 * math.sin(2 * math.pi * 440 * t),
                 'mixed': harmonic if 0.15 <= t < 0.55 or 0.8 <= t < 1.1 else 0}[recipe['kind']]
        channels = [value]
        if recipe['channels'] == 2:
            # Left is a declared synthetic contact-like shape; not a physiological EGG model.
            channels = [0.5 * math.sin(phase) + 0.1 * math.sin(3 * phase), harmonic]
        payload.extend(struct.pack('<' + 'h' * len(channels), *(round(v * 32767) for v in channels)))
    with wave.open(str(path), 'wb') as handle:
        handle.setparams((recipe['channels'], 2, rate, frames, 'NONE', 'not compressed'))
        handle.writeframes(payload)
    return path


def load_json(path):
    raw = Path(path).read_bytes()
    if str(path).endswith('.gz'):
        raw = gzip.decompress(raw)
    return json.loads(raw)


def write_json(path, value):
    path = Path(path)
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')
    if str(path).endswith('.gz'):
        raw = gzip.compress(raw, mtime=0)
    path.write_bytes(raw)


def compare(expected, actual, path='', rtol=1e-7, atol=1e-10):
    """Exact keys, shape, time and nonfinite masks; tight numeric tolerance elsewhere."""
    differences = []
    if type(expected) != type(actual) and not (type(expected) in (int, float) and type(actual) in (int, float)):
        return [path + ': type changed']
    if isinstance(expected, dict):
        if expected.keys() != actual.keys():
            return [path + ': keys changed']
        for key in expected:
            differences += compare(expected[key], actual[key], path + '/' + key, rtol, atol)
    elif isinstance(expected, list):
        if len(expected) != len(actual):
            return [path + ': length changed']
        for i, (a, b) in enumerate(zip(expected, actual)):
            differences += compare(a, b, path + '/' + str(i), rtol, atol)
            if len(differences) >= 20:
                break
    elif type(expected) in (int, float):
        parts = path.lower().split('/')
        exact = type(expected) is int or any(
            key in ('time_axis', 'time_vector', 'times', 'rtimes', 'shape', 'nonfinite')
            or key.endswith('_times') or key.startswith('sample_') for key in parts)
        exact = exact or any(key in path for key in [
            '/lowlevel_returns/irapt/2/', '/lowlevel_returns/_extract_f0_for_wm/1/', '/cq_sq_roi/0/', '/config/'])
        if not math.isfinite(expected) or not math.isfinite(actual):
            differences.append(path + ': illegal JSON nonfinite number')
        elif (expected != actual if exact else not math.isclose(expected, actual, rel_tol=rtol, abs_tol=atol)):
            differences.append(path + ': numeric difference')
    elif expected != actual:
        differences.append(path + ': value changed')
    return differences[:20]
