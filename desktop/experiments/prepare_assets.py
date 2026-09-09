"""Prepare deterministic P01 fixtures and unchanged, attributed assets."""
import hashlib
import json
import math
import struct
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / 'frontend/experiments/audio-viewport/public'


def font_names(data: bytes) -> dict:
    tables = {data[12 + i * 16:16 + i * 16].decode(): struct.unpack_from('>II', data, 20 + i * 16)
              for i in range(struct.unpack_from('>H', data, 4)[0])}
    offset, _ = tables['name']
    _, count, base = struct.unpack_from('>HHH', data, offset)
    names = {}
    for i in range(count):
        platform, _, _, name_id, length, start = struct.unpack_from('>HHHHHH', data, offset + 6 + i * 12)
        if name_id not in names:
            names[name_id] = data[offset + base + start:offset + base + start + length].decode(
                'utf-16-be' if platform in (0, 3) else 'mac-roman')
    return names


if __name__ == '__main__':
    PUBLIC.mkdir(parents=True, exist_ok=True)
    font = (ROOT / 'phonetic_toolbox/gui/resources/ipa_trans/DoulosSIL-Regular.ttf').read_bytes()
    names = font_names(font)
    assert 'PREAMBLE' in names[13] and 'PERMISSION & CONDITIONS' in names[13]
    (PUBLIC / 'DoulosSIL-Regular.ttf').write_bytes(font)
    (PUBLIC / 'Doulos-OFL.txt').write_text(names[0] + '\n\n' + names[13] + '\n', encoding='utf-8')
    (PUBLIC / 'icon.png').write_bytes((ROOT / 'docs/design/assets/K2-wave-dumpling.png').read_bytes())
    with wave.open(str(PUBLIC / 'fixture.wav'), 'wb') as output:
        output.setparams((2, 2, 44100, 88200, 'NONE', 'not compressed'))
        pcm = bytearray()
        for i in range(88200):
            time = i / 44100
            envelope = min(1, i / 441, (88199 - i) / 441)
            for frequency in (440, 660):
                value = 0.55 * envelope * (0.7 + 0.3 * math.cos(2 * math.pi * 3 * time)) * math.sin(2 * math.pi * frequency * time)
                pcm.extend(struct.pack('<h', round(value * 32767)))
        output.writeframes(pcm)
    result = {'font_family': names[1], 'font_version': names[5], 'font_copyright': names[0],
              'font_sha256': hashlib.sha256(font).hexdigest(),
              'fixture': {'kind': 'deterministic synthetic stereo PCM16', 'sample_rate_hz': 44100,
                          'sample_count': 88200, 'frequencies_hz': [440, 660], 'scientific_result': False},
              'files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in PUBLIC.iterdir()
                        if p.is_file() and p.name != 'asset-manifest.json'}}
    (PUBLIC / 'asset-manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'font_version': names[5], 'files': list(result['files'])}))
