"""Render selected artwork identically in the workbench and Windows host."""
import json
import sys
import struct
from pathlib import Path

from PyQt6.QtCore import Qt, QBuffer, QIODevice, QRectF
from PyQt6.QtGui import QIcon, QImage, QPainter, QPainterPath, QPixmap
from PyQt6.QtWidgets import QApplication


ICON_SIZES = (16, 24, 32, 48, 64, 128, 256)
MACOS_BUILTIN_PERCENT = 82


def builtin_catalog(dist: Path) -> dict:
    path = dist / 'app-icons/catalog.json'
    if not path.is_file():
        return {}
    catalog = json.loads(path.read_text(encoding='utf-8'))
    if catalog.get('schema') != 'ptb-app-icons/1':
        raise ValueError('Unsupported application icon catalog')
    return {entry['id']: entry for entry in catalog['icons']}


def builtin_image(dist: Path, identity='default') -> QImage:
    entry = builtin_catalog(dist).get(identity)
    if not entry:
        raise ValueError('内置图标不存在。')
    if Path(entry['sheet']).name != entry['sheet']:
        raise ValueError('Invalid icon sheet path')
    image = QImage(str(dist / 'app-icons/sheets' / entry['sheet']))
    x, y, width, height = entry['rect']
    if image.isNull() or x < 0 or y < 0 or x + width > image.width() or y + height > image.height():
        raise ValueError('内置图标素材无法读取。')
    scale = 256 / max(width, height)
    w, h = width * scale, height * scale
    target = QImage(256, 256, QImage.Format.Format_ARGB32_Premultiplied)
    target.fill(Qt.GlobalColor.transparent)
    painter = QPainter(target)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    rect = QRectF((256 - w) / 2, (256 - h) / 2, w, h)
    clip = QPainterPath()
    radius = min(w, h) * entry['radius']
    clip.addRoundedRect(rect, radius, radius)
    painter.setClipPath(clip)
    painter.drawImage(rect, image, QRectF(x, y, width, height))
    painter.end()
    return target


def icon_images(source: Path | QImage) -> list[QImage]:
    """Normalize K2's visible footprint for Windows without changing the artwork.

    The source includes wide transparent padding and very faint edge specks.
    Bounds use alpha >= 32, with a small safety margin around the visible body.
    Scaling at each target size retains the square aspect ratio and rounded corners.
    """
    image = (source if isinstance(source, QImage) else QImage(str(source))).convertToFormat(QImage.Format.Format_RGBA8888)
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


def macos_icon_images(source: Path | QImage, percent=MACOS_BUILTIN_PERCENT, *, builtin=True) -> list[QImage]:
    """Add the Mac canvas gutter; keep custom PNG transparency and aspect ratio."""
    if type(percent) is not int or not 60 <= percent <= 100:
        raise ValueError('Dock 图标大小应为 60%–100%。')
    if builtin:
        normalized = icon_images(source)
        if not normalized:
            return []
        image = normalized[-1]
    else:
        image = source if isinstance(source, QImage) else QImage(str(source))
    if image.isNull():
        return []
    result = []
    for size in ICON_SIZES:
        span = max(1, round(size * percent / 100))
        scaled = image.scaled(span, span, Qt.AspectRatioMode.KeepAspectRatio,
                              Qt.TransformationMode.SmoothTransformation)
        target = QImage(size, size, QImage.Format.Format_ARGB32_Premultiplied)
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
    source = builtin_image(dist) if builtin_catalog(dist) else candidates[0] if candidates else None
    if source is not None:
        renderer = macos_icon_images if sys.platform == 'darwin' else icon_images
        for image in renderer(source):
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
    images = icon_images(builtin_image(source) if source.is_dir() else source)
    if not images:
        raise ValueError('Application icon is unavailable')
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
