"""Use the same bundled K2 asset as the workbench home and sidebar."""
import sys
import struct
from pathlib import Path

from PyQt6.QtCore import Qt, QBuffer, QIODevice
from PyQt6.QtGui import QIcon, QImage, QPainter, QPixmap
from PyQt6.QtWidgets import QApplication


ICON_SIZES = (16, 24, 32, 48, 64, 128, 256)


def icon_images(source: Path) -> list[QImage]:
    """Normalize K2's visible footprint for Windows without changing the artwork.

    The source includes wide transparent padding and very faint edge specks.
    Bounds use alpha >= 32, with a small safety margin around the visible body.
    Scaling at each target size retains the square aspect ratio and rounded corners.
    """
    image = QImage(str(source)).convertToFormat(QImage.Format.Format_RGBA8888)
    if image.isNull():
        return []
    raw = image.constBits().asstring(image.sizeInBytes())
    left, top, right, bottom = image.width(), image.height(), -1, -1
    for y in range(image.height()):
        row = raw[y * image.bytesPerLine() + 3:y * image.bytesPerLine() + image.width() * 4:4]
        occupied = [x for x, alpha in enumerate(row) if alpha >= 32]
        if occupied:
            left, right = min(left, occupied[0]), max(right, occupied[-1])
            top, bottom = min(top, y), y
    if right < left:
        return []
    margin = max(1, round(max(right - left, bottom - top) * .003))
    left, top = max(0, left - margin), max(0, top - margin)
    right, bottom = min(image.width() - 1, right + margin), min(image.height() - 1, bottom + margin)
    cropped = image.copy(left, top, right - left + 1, bottom - top + 1)
    result = []
    for size in ICON_SIZES:
        # Keep a pixel of separation for tiny title-bar icons. At taskbar sizes,
        # the source crop's safety margin is enough; use the full canvas.
        inset = 1 if size < 32 else 0
        scaled = cropped.scaled(size - 2 * inset, size - 2 * inset,
                                Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        target = QImage(size, size, QImage.Format.Format_ARGB32)
        target.fill(Qt.GlobalColor.transparent)
        painter = QPainter(target)
        painter.drawImage((size - scaled.width()) // 2, (size - scaled.height()) // 2, scaled)
        painter.end()
        result.append(target)
    return result


def configure_app_icon(dist: Path) -> QIcon:
    # Vite emits this imported asset with a content hash in both source and EXE builds.
    candidates = sorted((dist / 'assets').glob('k2-*.png'))
    icon = QIcon()
    if candidates:
        for image in icon_images(candidates[0]):
            icon.addPixmap(QPixmap.fromImage(image))
    app = QApplication.instance()
    if app is not None and not icon.isNull():
        app.setWindowIcon(icon)
    if sys.platform == 'win32':
        import ctypes
        # Set before showing the first window; this is process-local, not a registry change.
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('PhoneticToolbox.Desktop.3')
    return icon


def write_windows_icon(source: Path, destination: Path) -> None:
    """Write the same PNG-backed resolutions into a Windows ICO for future builds."""
    images = icon_images(source)
    if not images:
        raise ValueError('K2 application icon is unavailable')
    payloads = []
    for image in images:
        buffer = QBuffer()
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        if not image.save(buffer, 'PNG'):
            raise RuntimeError('Unable to encode application icon')
        payloads.append(bytes(buffer.data()))
    header = struct.pack('<HHH', 0, 1, len(images))
    offset = 6 + 16 * len(images)
    entries = []
    for image, data in zip(images, payloads):
        entries.append(struct.pack('<BBBBHHII', image.width() % 256, image.height() % 256,
                                   0, 0, 1, 32, len(data), offset))
        offset += len(data)
    destination.write_bytes(header + b''.join(entries) + b''.join(payloads))
