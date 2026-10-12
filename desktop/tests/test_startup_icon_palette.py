import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from ptb_desktop.startup_progress import DEFAULT_PALETTE, light_palette, startup_choice


def test_default_and_unreadable_preferences_keep_warm_w1_and_original_file(tmp_path):
    path = tmp_path / 'preferences.json'
    assert startup_choice(path)['palette'] == DEFAULT_PALETTE
    path.write_bytes(b'{broken settings')
    original = path.read_bytes()
    assert startup_choice(path)['id'] == 'default' and path.read_bytes() == original
    path.write_text(json.dumps({'version': 1, 'selected': 'builtin-002', 'custom': []}), 'utf8')
    assert startup_choice(path)['id'] == 'default'


def test_palette_preserves_hue_but_does_not_follow_dark_icon_brightness():
    bright = light_palette([(240, 30, 30, 255)] * 10)
    dark = light_palette([(80, 10, 10, 255)] * 10)
    assert bright == dark and min(int(bright['surface'][i:i + 2], 16) for i in (0, 2, 4)) >= 220
    assert bright != DEFAULT_PALETTE
    assert light_palette([(0, 0, 0, 0), (240, 30, 30, 255)]) == bright


def test_existing_builtin_choice_uses_bounded_bootstrap_artwork(tmp_path):
    path = tmp_path / 'preferences.json'
    path.write_text(json.dumps({'version': 1, 'selected': 'builtin-003', 'custom': []}), 'utf8')
    ico = tmp_path / 'builtin-003.ico'; ico.write_bytes(b'preview')
    palette = light_palette([(20, 180, 100, 255)])
    (tmp_path / 'catalog.json').write_text(json.dumps({'schema': 'ptb-startup-icons/1', 'icons': {'builtin-003': {'file': ico.name, 'palette': palette}}}), 'utf8')
    choice = startup_choice(path, tmp_path)
    assert choice['id'] == 'builtin-003' and choice['path'] == ico and choice['palette'] == palette
    ico.write_bytes(b'x' * 50_001)
    assert startup_choice(path, tmp_path)['id'] == 'default'


def test_custom_png_is_bound_to_existing_preference_identity(tmp_path):
    # Reading is bounded here. The actual native decoder validates the bitmap.
    data = b'\x89PNG\r\n\x1a\n' + b'\0\0\0\rIHDR' + (256).to_bytes(4, 'big') * 2
    identity = 'custom-' + hashlib.sha256(data).hexdigest()
    path = tmp_path / 'preferences.json'
    state = {'version': 1, 'selected': identity, 'custom': [{'id': identity, 'dataUrl': 'data:image/png;base64,' + base64.b64encode(data).decode('ascii')}]}
    path.write_text(json.dumps(state), 'utf8')
    assert startup_choice(path)['png'] == data
    state['selected'] = 'custom-' + '0' * 64
    path.write_text(json.dumps(state), 'utf8')
    assert startup_choice(path)['id'] == 'default'


def test_preparation_reader_has_no_qt_or_image_dependency():
    root = Path(__file__).resolve().parents[2]
    code = 'import sys; sys.path.insert(0,' + repr(str(root / 'desktop/src')) + '); from ptb_desktop.startup_progress import startup_choice; startup_choice(); assert not any(n.startswith(("PyQt6", "PIL", "numpy")) for n in sys.modules)'
    subprocess.run([sys.executable, '-B', '-c', code], check=True, capture_output=True)
