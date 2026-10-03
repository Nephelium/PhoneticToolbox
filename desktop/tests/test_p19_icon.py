import hashlib
import struct
from pathlib import Path

from PyQt6.QtGui import QImage
from ptb_desktop.app_icon import ICON_SIZES, icon_images, write_windows_icon

ROOT = Path(__file__).resolve().parents[2]


def test_p19_real_icon_has_consistent_visible_footprint_and_keeps_source(tmp_path):
    source = ROOT / 'frontend/src/assets/k2.png'
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    images = icon_images(source)
    assert [im.width() for im in images] == list(ICON_SIZES)
    for image in images:
        size = image.width()
        positions = [(x, y) for y in range(size) for x in range(size)
                     if image.pixelColor(x, y).alpha() >= 32]
        xs, ys = zip(*positions)
        assert 0 <= min(xs) <= max(xs) < size
        assert 0 <= min(ys) <= max(ys) < size
        if size < 32:
            assert min(xs) > 0 and max(xs) < size - 1
            assert min(ys) > 0 and max(ys) < size - 1
        footprint = (max(xs) - min(xs) + 1) / size
        assert (.98 <= footprint <= 1) if size >= 32 else (.85 <= footprint <= .97)
        # The enlarged artwork keeps its rounded transparent corners.
        assert all(image.pixelColor(x, y).alpha() < 32
                   for x, y in [(0, 0), (size - 1, 0), (0, size - 1), (size - 1, size - 1)])
        # Independent audit of the original alpha>=32 bounds: 990 x 973.
        # The visible artwork is slightly rectangular, even though its canvas is square.
        # Preserve that ratio to within one output pixel of raster quantization.
        assert abs((max(ys) - min(ys) + 1) - (max(xs) - min(xs) + 1) * 973 / 990) <= 1
    target = tmp_path / 'app.ico'
    write_windows_icon(source, target)
    blob = target.read_bytes()
    assert struct.unpack_from('<HHH', blob) == (0, 1, len(ICON_SIZES))
    for index, size in enumerate(ICON_SIZES):
        w, h, _, _, _, _, length, offset = struct.unpack_from('<BBBBHHII', blob, 6 + 16 * index)
        image = QImage.fromData(blob[offset:offset + length], 'PNG')
        assert not image.isNull() and image.width() == image.height() == size
        assert (w or 256) == (h or 256) == size
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before


def test_p19_invalid_icon_returns_empty_without_setting_an_unrelated_icon(tmp_path):
    assert icon_images(tmp_path / 'missing.png') == []
