"""Make compact, browser-readable manual media in a generated package only.

The editable project and its source WAV/PNG files are never changed.
"""
from __future__ import annotations

from io import BytesIO
import hashlib
from pathlib import Path
import shutil
import subprocess


def package_asset(asset: dict, source: Path, output_root: Path) -> tuple[dict, dict]:
    original = source.read_bytes()
    original_hash = hashlib.sha256(original).hexdigest()
    if asset.get('sha256') != original_hash:
        raise ValueError('Manual source media hash changed: ' + str(asset.get('id')))
    kind = asset['kind']
    packed = original
    extension = source.suffix.lower()
    mime = asset.get('mime')
    if kind == 'image' and extension in ('.png', '.jpg', '.jpeg'):
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
    elif kind == 'audio' and extension == '.wav':
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
                   'sourceSha256': original_hash, 'packageSha256': packed_hash, 'mime': mime}
