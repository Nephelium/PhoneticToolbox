"""Application icon choices live with user preferences, outside release caches."""
import base64
import hashlib
import json
import sys
from pathlib import Path

from PyQt6.QtCore import QBuffer, QIODevice, QSaveFile
from PyQt6.QtGui import QIcon, QImage, QImageReader, QPixmap
from PyQt6.QtWidgets import QApplication
from .app_icon import builtin_catalog, builtin_image, icon_images, macos_icon_images, MACOS_BUILTIN_PERCENT

MAX_ICONS = 8
MAX_PNG_BYTES = 400_000


def decode_icon(value):
    if not isinstance(value, str) or not value.startswith('data:image/png;base64,') or len(value) > 540_000:
        raise ValueError('图标图片无效，请重新导入。')
    try:
        data = base64.b64decode(value.split(',', 1)[1], validate=True)
    except (ValueError, TypeError):
        raise ValueError('图标图片编码无效。') from None
    if len(data) > MAX_PNG_BYTES:
        raise ValueError('图标图片过大，请重新导入。')
    buffer = QBuffer()
    buffer.setData(data)
    buffer.open(QIODevice.OpenModeFlag.ReadOnly)
    reader = QImageReader(buffer, b'png')
    size = reader.size()
    if size.width() != 256 or size.height() != 256:
        raise ValueError('图标需要转换为 256 × 256 像素。')
    image = reader.read()
    if image.isNull():
        raise ValueError('无法读取图标图片。')
    # Re-encode pixels to discard metadata and never retain a source file path.
    output = QBuffer()
    output.open(QIODevice.OpenModeFlag.WriteOnly)
    if not image.save(output, 'PNG'):
        raise ValueError('无法保存图标图片。')
    return bytes(output.data()), image


class IconPreferences:
    def __init__(self, root: Path, default_icon: QIcon, window=None, dist: Path | None = None):
        self.path = Path(root) / 'preferences.json'
        self.default_icon = default_icon
        self.window = window
        self.dist = dist
        self.builtin = builtin_catalog(dist) if dist else {}
        self.builtin_ids = {'default', *self.builtin}

    def read(self):
        if not self.path.exists():
            return self.expose({'version': 1, 'selected': 'default', 'custom': []})
        if self.path.stat().st_size > 4_400_000:
            raise ValueError('图标设置文件过大，当前使用默认图标。')
        try:
            state = json.loads(self.path.read_text(encoding='utf-8'))
            if not isinstance(state, dict) or state.get('version') != 1:
                raise ValueError()
            # Earlier W1 and the two retired builtins fall back to the current default.
            if state.get('selected') in {'builtin-001', 'builtin-002', 'builtin-008'}:
                state['selected'] = 'default'
            entries = state['custom']
            if not isinstance(entries, list) or len(entries) > MAX_ICONS:
                raise ValueError()
            ids = set(self.builtin_ids)
            for entry in entries:
                data, _ = decode_icon(entry['dataUrl'])
                identity = 'custom-' + hashlib.sha256(data).hexdigest()
                if entry['id'] != identity or identity in ids or not isinstance(entry['name'], str) or len(entry['name']) > 80:
                    raise ValueError()
                ids.add(identity)
            if state['selected'] not in ids:
                raise ValueError()
            sizes = state.get('macosSizes', {})
            if not isinstance(sizes, dict) or len(sizes) > 128 or any(
                not isinstance(k, str) or len(k) > 80 or type(v) is not int or not 60 <= v <= 100
                for k, v in sizes.items()
            ):
                raise ValueError()
            return self.expose(state)
        except (KeyError, TypeError, ValueError):
            raise ValueError('图标设置无法读取，原文件已保留。') from None

    def write(self, state):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        file = QSaveFile(str(self.path))
        if not file.open(QIODevice.OpenModeFlag.WriteOnly):
            raise OSError('图标设置无法保存，原选择保留。')
        saved = {k: v for k, v in state.items() if k != 'dockSize'}
        payload = (json.dumps(saved, ensure_ascii=False, separators=(',', ':')) + '\n').encode('utf-8')
        if file.write(payload) != len(payload) or not file.commit():
            raise OSError('图标设置无法保存，原选择保留。')

    def expose(self, state):
        value = dict(state)
        value.pop('dockSize', None)
        if sys.platform == 'darwin':
            default = MACOS_BUILTIN_PERCENT if state['selected'] in self.builtin_ids else 100
            value['dockSize'] = dict(value=state.get('macosSizes', {}).get(state['selected'], default),
                                     default=default, min=60, max=100)
        return value

    def icon_for(self, state):
        icon = self.default_icon
        if state['selected'] in self.builtin:
            icon = QIcon()
            source = builtin_image(self.dist, state['selected'])
            images = macos_icon_images(source, self.expose(state)['dockSize']['value']) if sys.platform == 'darwin' else icon_images(source)
            for image in images:
                icon.addPixmap(QPixmap.fromImage(image))
        elif state['selected'] != 'default':
            item = next(entry for entry in state['custom'] if entry['id'] == state['selected'])
            _, image = decode_icon(item['dataUrl'])
            if sys.platform == 'darwin':
                icon = QIcon()
                for rendered in macos_icon_images(image, self.expose(state)['dockSize']['value'], builtin=False):
                    icon.addPixmap(QPixmap.fromImage(rendered))
            else:
                icon = QIcon(QPixmap.fromImage(image))
        return icon

    def apply(self, state, icon=None):
        icon = self.icon_for(state) if icon is None else icon
        app = QApplication.instance()
        if app:
            app.setWindowIcon(icon)
            for window in app.topLevelWidgets():
                window.setWindowIcon(icon)
        if self.window:
            self.window.setWindowIcon(icon)

    def request(self, body):
        if not isinstance(body, dict) or set(body) - {'action', 'id', 'name', 'dataUrl', 'size'}:
            raise ValueError('不支持的图标请求。')
        action = body.get('action')
        if action not in ('status', 'select', 'import', 'replace', 'remove', 'resize'):
            raise ValueError('不支持的图标操作。')
        state = self.read()
        if action == 'status':
            return state
        if action == 'resize':
            if sys.platform != 'darwin':
                raise ValueError('Dock 图标大小仅适用于 macOS 桌面版。')
            size = body.get('size')
            if type(size) is not int or not 60 <= size <= 100:
                raise ValueError('Dock 图标大小应为 60%–100%。')
            if body.get('id') != state['selected']:
                raise ValueError('图标已切换，请调整当前图标的大小。')
            state.setdefault('macosSizes', {})[state['selected']] = size
        elif action in ('import', 'replace'):
            previous = next((entry for entry in state['custom'] if entry['id'] == body.get('id')), None) if action == 'replace' else None
            if action == 'replace' and previous is None:
                raise ValueError('待裁剪的自定义图标不存在。')
            data, _ = decode_icon(body.get('dataUrl'))
            identity = 'custom-' + hashlib.sha256(data).hexdigest()
            name = body.get('name', previous['name'] if previous else '自定义图标')
            if not isinstance(name, str):
                raise ValueError('图标名称无效。')
            name = name.replace('\\', '/').rsplit('/', 1)[-1].strip()[:80] or '自定义图标'
            replacement = {'id': identity, 'name': name, 'dataUrl': 'data:image/png;base64,' + base64.b64encode(data).decode('ascii')}
            if previous:
                if identity == previous['id']:
                    state['custom'] = [replacement if entry['id'] == previous['id'] else entry for entry in state['custom']]
                elif any(entry['id'] == identity for entry in state['custom']):
                    state['custom'] = [entry for entry in state['custom'] if entry['id'] != previous['id']]
                else:
                    state['custom'] = [replacement if entry['id'] == previous['id'] else entry for entry in state['custom']]
            elif not any(entry['id'] == identity for entry in state['custom']):
                if len(state['custom']) >= MAX_ICONS:
                    raise ValueError('最多保留 8 张自定义图标，请先移除不需要的图标。')
                state['custom'].append(replacement)
            state['selected'] = identity
            if previous and previous['id'] != identity and 'macosSizes' in state:
                old_size = state['macosSizes'].pop(previous['id'], None)
                if old_size is not None:
                    state['macosSizes'].setdefault(identity, old_size)
        else:
            identity = body.get('id')
            if identity not in [*self.builtin_ids, *(entry['id'] for entry in state['custom'])]:
                raise ValueError('所选图标不存在。')
            if action == 'select':
                state['selected'] = identity
            else:
                if identity in self.builtin_ids:
                    raise ValueError('内置图标不能移除。')
                state['custom'] = [entry for entry in state['custom'] if entry['id'] != identity]
                state.get('macosSizes', {}).pop(identity, None)
                if state['selected'] == identity:
                    state['selected'] = 'default'
        icon = self.icon_for(state)
        self.write(state)
        self.apply(state, icon)
        return self.expose(state)
