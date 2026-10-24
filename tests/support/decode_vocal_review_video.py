"""Decode the actual exported MP4, retain one PNG and reject empty pictures."""
import json
import struct
import sys
import zlib
from pathlib import Path
import av


def chunk(kind, data):
    return struct.pack('!I', len(data)) + kind + data + struct.pack('!I', zlib.crc32(kind + data) & 0xffffffff)


with av.open(sys.argv[1]) as container:
    frames = list(container.decode(video=0))
assert len(frames) == 6
rgb = frames[0].to_ndarray(format='rgb24')
assert rgb.shape == (720, 1280, 3)
assert float(rgb.mean()) > 10, 'exported video contains an empty/black frame'
png = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!2I5B', 1280, 720, 8, 2, 0, 0, 0))
png += chunk(b'IDAT', zlib.compress(b''.join(b'\0' + row.tobytes() for row in rgb))) + chunk(b'IEND', b'')
Path(sys.argv[2]).write_bytes(png)
print(json.dumps(dict(frames=len(frames),width=1280,height=720,mean_rgb=float(rgb.mean()),decoder='PyAV/FFmpeg')))
