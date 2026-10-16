"""Make compact, browser-readable manual media in a generated package only.

The editable project and its source WAV/PNG files are never changed.
"""
from __future__ import annotations

from io import BytesIO
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import uuid


@lru_cache(maxsize=4)
def encoder_identity(kind: str) -> dict:
    if kind == 'image':
        import PIL
        from PIL import features
        return {'pillow': PIL.__version__, 'webp': features.version('webp')}
    if kind == 'audio':
        ffmpeg = shutil.which('ffmpeg')
        if not ffmpeg:
            raise RuntimeError('Compact manual media requires the existing FFmpeg executable')
        with Path(ffmpeg).open('rb') as stream:
            return {'ffmpegSha256': hashlib.file_digest(stream, 'sha256').hexdigest()}
    return {}


def cache_key(asset: dict, source: Path, original_hash: str) -> str:
    # Changing this encoder implementation or its native codec invalidates reuse.
    recipe = {'source': original_hash, 'kind': asset['kind'], 'extension': source.suffix.lower(),
              'mime': asset.get('mime'), 'sampleRate': asset.get('sampleRate'),
              'channels': asset.get('channels'), 'encoder': encoder_identity(asset['kind']),
              'implementation': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    return hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()


def checked_cache_root(root: Path) -> Path:
    owned = Path(__file__).resolve().parents[2] / 'output'
    root = Path(root).absolute()
    if not root.resolve().is_relative_to(owned.resolve()):
        raise ValueError('Media cache must be inside project output')
    for path in (root, *root.parents):
        if is_link(path):
            raise ValueError('Media cache path is linked')
    root.mkdir(parents=True, exist_ok=True)
    return root


def is_link(path: Path) -> bool:
    return path.is_symlink() or (path.exists() and bool(getattr(path.lstat(), 'st_file_attributes', 0) & 0x400))


def package_asset(asset: dict, source: Path, output_root: Path, cache_root: Path | None = None) -> tuple[dict, dict]:
    original = source.read_bytes()
    original_hash = hashlib.sha256(original).hexdigest()
    if asset.get('sha256') != original_hash:
        raise ValueError('Manual source media hash changed: ' + str(asset.get('id')))
    kind = asset['kind']
    packed = original
    extension = source.suffix.lower()
    mime = asset.get('mime')
    cached = None
    if cache_root is not None:
        cache_root = checked_cache_root(cache_root)
        key = cache_key(asset, source, original_hash)
        record, blob = cache_root / (key + '.json'), cache_root / (key + '.bin')
        if any(is_link(p) for p in (record, blob)):
            raise ValueError('Media cache entry is linked')
        if record.exists():
            cached = json.loads(record.read_text('utf8'))
            packed = blob.read_bytes()
            if (cached['key'] != key or cached['sourceSha256'] != original_hash or
                    len(packed) != cached['size'] or hashlib.sha256(packed).hexdigest() != cached['sha256']):
                raise ValueError('Manual media cache identity mismatch')
            extension, mime = cached['extension'], cached['mime']
            if cached.get('audioMetadata'):
                asset.update(cached['audioMetadata'])
    if cached is None and kind == 'image' and extension in ('.png', '.jpg', '.jpeg'):
        from PIL import Image
        with Image.open(BytesIO(original)) as image:
            image.load()
            if image.mode not in ('RGB', 'RGBA'):
                has_alpha = 'A' in image.getbands() or 'transparency' in image.info
                image = image.convert('RGBA' if has_alpha else 'RGB')
            stream = BytesIO()
            image.save(stream, 'WEBP', lossless=True, method=6)
            candidate = stream.getvalue()
        if len(candidate) < len(original):
            packed, extension, mime = candidate, '.webp', 'image/webp'
    elif cached is None and kind == 'audio' and extension == '.wav':
        ffmpeg = shutil.which('ffmpeg')
        if not ffmpeg:
            raise RuntimeError('Compact manual media requires the existing FFmpeg executable')
        sample_rate = int(asset.get('sampleRate') or 44100)
        sample_rate = sample_rate if 8000 <= sample_rate <= 48000 else 44100
        channels = min(2, max(1, int(asset.get('channels') or 1)))
        command = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-nostdin', '-i', str(source),
                   '-map', '0:a:0', '-vn', '-map_metadata', '-1', '-c:a', 'libmp3lame',
                   '-q:a', '2', '-ar', str(sample_rate), '-ac', str(channels), '-f', 'mp3', 'pipe:1']
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if process.returncode or not process.stdout:
            raise RuntimeError('Cannot encode manual example ' + str(asset.get('id')) + ': '
                               + process.stderr.decode('utf-8', 'replace')[-500:])
        if len(process.stdout) < len(original):
            packed, extension, mime = process.stdout, '.mp3', 'audio/mpeg'
            asset['sampleRate'], asset['channels'] = sample_rate, channels
    packed_hash = hashlib.sha256(packed).hexdigest()
    if cache_root is not None and cached is None:
        metadata = {'key': key, 'sourceSha256': original_hash, 'size': len(packed),
                    'sha256': packed_hash, 'extension': extension, 'mime': mime,
                    'audioMetadata': {k: asset[k] for k in ('sampleRate', 'channels') if k in asset} if kind == 'audio' else {}}
        # Blob first, manifest last; an interrupted first write is not a cache hit.
        for target, content in ((blob, packed), (record, (json.dumps(metadata) + '\n').encode())):
            temporary = target.with_name(target.name + '.' + uuid.uuid4().hex + '.tmp')
            temporary.write_bytes(content)
            os.replace(temporary, target)
    relative = 'assets/packed/' + hashlib.sha256((str(asset['id']) + original_hash).encode()).hexdigest()[:24] + extension
    destination = output_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() != packed_hash:
        raise ValueError('Compact media identity collision')
    if not destination.exists():
        destination.write_bytes(packed)
    asset['originalSha256'] = original_hash
    asset['path'], asset['sha256'] = relative, packed_hash
    if mime:
        asset['mime'] = mime
    return asset, {'id': asset['id'], 'sourceBytes': len(original), 'packageBytes': len(packed),
                   'sourceSha256': original_hash, 'packageSha256': packed_hash, 'mime': mime,
                   'cacheHit': cached is not None}
