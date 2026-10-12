import base64
import hashlib
import json
from pathlib import Path

import pytest
from PyQt6.QtCore import QBuffer, QIODevice
from PyQt6.QtGui import QIcon, QImage
from PyQt6.QtWidgets import QApplication

from ptb_desktop.app_icon import builtin_catalog, builtin_image, write_windows_icon
from ptb_desktop.icon_preferences import IconPreferences

ROOT = Path(__file__).resolve().parents[2]
DIST = ROOT / 'frontend/public'


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication(['owned-icon-unit-test'])


def png_url(color='#ee6655', size=256):
    image = QImage(size, size, QImage.Format.Format_ARGB32)
    image.fill(0x00000000)
    from PyQt6.QtGui import QColor
    image.setPixelColor(size // 2, size // 2, QColor(color))
    buffer = QBuffer()
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    assert image.save(buffer, 'PNG')
    return 'data:image/png;base64,' + base64.b64encode(bytes(buffer.data())).decode('ascii')


def test_selected_catalog_order_default_and_distinct_real_artwork(app, tmp_path):
    catalog = builtin_catalog(DIST)
    assert len(catalog) == 31
    entries = list(catalog.values())
    assert [e['group'] for e in entries] == ['waveform'] * 7 + ['spectrogram', 'mouth'] + ['annotation'] * 2 + ['ipa'] * 20
    assert entries[0]['id'] == 'default' and entries[0]['original'].startswith('W1')
    assert entries[1]['original'] == 'W2'
    assert not {'builtin-001', 'builtin-008'} & catalog.keys()
    assert catalog['builtin-033']['original'] == 'IPA-T'
    assert [e['original'] for e in entries[11:]] == ['IPA-' + chr(65 + i) for i in range(20)]
    hashes = set()
    for entry in entries:
        image = builtin_image(DIST, entry['id'])
        assert image.width() == image.height() == 256
        assert image.pixelColor(0, 0).alpha() == 0
        assert image.pixelColor(128, 128).alpha() == 255
        hashes.add(hashlib.sha256(image.constBits().asstring(image.sizeInBytes())).hexdigest())
    assert len(hashes) == 31
    ico = tmp_path / 'default.ico'
    write_windows_icon(DIST, ico)
    assert QImage(str(ico)).width() > 0


def test_every_builtin_selects_and_survives_reopening(app, tmp_path):
    store = IconPreferences(tmp_path, QIcon(), dist=DIST)
    for identity in builtin_catalog(DIST):
        assert store.request({'action': 'select', 'id': identity})['selected'] == identity
        reopened = IconPreferences(tmp_path, QIcon(), dist=DIST)
        assert reopened.read()['selected'] == identity
        assert not reopened.icon_for(reopened.read()).isNull()
    with pytest.raises(ValueError):
        store.request({'action': 'remove', 'id': 'builtin-014'})


def test_custom_import_deduplicates_persists_and_removal_restores_default(app, tmp_path):
    store = IconPreferences(tmp_path, QIcon(), dist=DIST)
    state = store.request({'action': 'import', 'name': '../可爱的图.png', 'dataUrl': png_url()})
    identity = state['selected']
    assert state['custom'][0]['name'] == '可爱的图.png'
    assert len(store.request({'action': 'import', 'dataUrl': png_url()})['custom']) == 1
    assert IconPreferences(tmp_path, QIcon(), dist=DIST).read()['selected'] == identity
    state = store.request({'action': 'remove', 'id': identity})
    assert state['selected'] == 'default' and not state['custom']


def test_recrop_replaces_in_full_library_and_deduplicates_without_losing_other_icons(app, tmp_path):
    store = IconPreferences(tmp_path, QIcon(), dist=DIST)
    for i in range(8):
        store.request({'action': 'import', 'name': f'{i}.png', 'dataUrl': png_url(f'#{i:06x}')})
    original = store.read()
    identity = original['custom'][0]['id']
    replaced = store.request({'action': 'replace', 'id': identity, 'dataUrl': png_url('#abcdef')})
    assert len(replaced['custom']) == 8
    assert replaced['selected'] == replaced['custom'][0]['id'] != identity
    assert replaced['custom'][1:] == original['custom'][1:]
    assert store.read() == replaced
    duplicate = store.request({'action': 'replace', 'id': replaced['custom'][0]['id'], 'dataUrl': png_url('#000001')})
    assert len(duplicate['custom']) == 7
    assert duplicate['selected'] == original['custom'][1]['id']


def test_invalid_or_failed_recrop_keeps_exact_preference_bytes(app, tmp_path, monkeypatch):
    store = IconPreferences(tmp_path, QIcon(), dist=DIST)
    selected = store.request({'action': 'import', 'dataUrl': png_url()})['selected']
    before = store.path.read_bytes()
    for body in [{'action': 'replace', 'id': 'default', 'dataUrl': png_url()}, {'action': 'replace', 'id': selected, 'dataUrl': 'bad'}]:
        with pytest.raises(ValueError):
            store.request(body)
        assert store.path.read_bytes() == before
    monkeypatch.setattr(store, 'write', lambda _: (_ for _ in ()).throw(OSError('simulated disk failure')))
    with pytest.raises(OSError):
        store.request({'action': 'replace', 'id': selected, 'dataUrl': png_url('#123456')})
    assert store.path.read_bytes() == before


@pytest.mark.parametrize('url', ['data:image/png;base64,???', 'data:image/png;base64,AA==', png_url(size=512), 'data:image/png;base64,' + 'A' * 540001], ids=['encoding', 'not-image', 'dimensions', 'payload-limit'])
def test_invalid_import_keeps_existing_choice_and_file(app, tmp_path, url):
    store = IconPreferences(tmp_path, QIcon(), dist=DIST)
    store.request({'action': 'select', 'id': 'builtin-033'})
    original = store.path.read_bytes()
    with pytest.raises(ValueError):
        store.request({'action': 'import', 'dataUrl': url})
    assert store.path.read_bytes() == original


def test_full_library_and_save_failure_keep_choice(app, tmp_path, monkeypatch):
    store = IconPreferences(tmp_path, QIcon(), dist=DIST)
    for i in range(8):
        store.request({'action': 'import', 'dataUrl': png_url(f'#{i:06x}')})
    original = store.path.read_bytes()
    with pytest.raises(ValueError, match='8'):
        store.request({'action': 'import', 'dataUrl': png_url('#ffffff')})
    assert store.path.read_bytes() == original
    def fail(_):
        raise OSError('simulated disk failure')
    monkeypatch.setattr(store, 'write', fail)
    with pytest.raises(OSError):
        store.request({'action': 'select', 'id': 'default'})
    assert store.path.read_bytes() == original


def test_corrupt_preference_is_reported_and_preserved(app, tmp_path):
    store = IconPreferences(tmp_path, QIcon(), dist=DIST)
    store.path.write_text(json.dumps({'version': 1, 'custom': [], 'selected': '../../outside'}), encoding='utf-8')
    original = store.path.read_bytes()
    with pytest.raises(ValueError, match='原文件已保留'):
        store.request({'action': 'status'})
    assert store.path.read_bytes() == original


@pytest.mark.parametrize('retired', ['builtin-001', 'builtin-002', 'builtin-008'])
def test_retired_preference_follows_default_without_rewriting_file(app, tmp_path, retired):
    store = IconPreferences(tmp_path, QIcon(), dist=DIST)
    store.path.write_text(json.dumps({'version': 1, 'custom': [], 'selected': retired}), encoding='utf-8')
    original = store.path.read_bytes()
    assert store.read()['selected'] == 'default'
    assert not store.icon_for(store.read()).isNull()
    assert store.path.read_bytes() == original
