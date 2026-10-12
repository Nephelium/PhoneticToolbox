"""Deletion/reacquisition races and the independent editorial notice."""
import copy
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from io import BytesIO

import pytest
from pypdf import PdfReader, PdfWriter
from ptb_desktop.papers import PaperService, PaperError, BASE_URL, validate_catalog
from ptb_desktop.paper_library import PersonalPaperLibrary
from test_papers import BODY, Opener, paper, setup


def test_removed_recommendation_stays_removed_until_explicit_acquisition(tmp_path):
    s, o = setup(tmp_path)
    s.download(s.first_launch, threading.Event(), lambda _: None)
    note = s.root / ('annotations-' + paper()['original']['sha256'] + '.json')
    note.write_text('retained record', 'utf8')
    s.remove_download('sample')
    assert not s.status()['papers'][0]['downloaded'] and s.status()['papers'][0]['dismissed']
    assert note.read_text('utf8') == 'retained record'
    before = len(o.calls)
    restarted = PaperService(tmp_path, opener=o, today=lambda: date(2026, 10, 7))
    restarted.download(restarted.first_launch, threading.Event(), lambda _: None)
    assert len(o.calls) == before
    restarted.download('0001-01-01', threading.Event(), lambda _: None, ['sample'])
    assert restarted.status()['papers'][0]['downloaded'] and not restarted.status()['papers'][0]['dismissed']
    assert note.read_text('utf8') == 'retained record'


def test_shared_assets_remain_readable_and_removed_paper_remains_selectable(tmp_path):
    s, o = setup(tmp_path, [paper(), paper(id='old')])
    s.download(s.first_launch, threading.Event(), lambda _: None)
    s.remove_download('sample')
    state = {p['id']: p for p in s.status()['papers']}
    assert state['old']['downloaded'] and s._verified(paper()['original'])
    assert state['sample']['dismissed'] and not state['sample']['downloaded']
    before = len(o.calls)
    s.download('0001-01-01', threading.Event(), lambda _: None, ['sample'])
    assert len(o.calls) == before and s.status()['papers'][0]['downloaded']


def test_removal_during_download_does_not_restore_bytes(tmp_path):
    s, _ = setup(tmp_path)
    entered = threading.Event(); release = threading.Event(); original = s._read
    def delayed(*args):
        entered.set(); assert release.wait(5)
        return original(*args)
    s._read = delayed
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(s.download, s.first_launch, threading.Event(), lambda _: None)
        assert entered.wait(5)
        s.remove_download('sample'); release.set(); future.result(timeout=5)
    assert not list(s.files.iterdir()) and s.status()['papers'][0]['dismissed']


def test_preferences_write_failure_rolls_back_removal(tmp_path, monkeypatch):
    s, _ = setup(tmp_path); s.download(s.first_launch, threading.Event(), lambda _: None)
    def denied(*_): raise PermissionError('owned test denial')
    monkeypatch.setattr('ptb_desktop.papers.atomic_json', denied)
    with pytest.raises(PaperError, match='恢复原位置'): s.remove_download('sample')
    assert s.status()['papers'][0]['downloaded'] and not s.status()['papers'][0]['dismissed']
    assert not list(s.files.glob('.removed-*'))


def notice_catalog():
    body = BODY + b'notice'
    n = {'id': 'reading-introduction', 'title': '请看这里~', 'authors': '井韶子',
         'publishedAt': '2026-10-06', 'updatedAt': '2026-10-07',
         'pdf': {'path': 'reading-introduction/note.pdf', 'size': len(body), 'sha256': hashlib.sha256(body).hexdigest()}}
    return {'schema': 'ptb-papers/1', 'papers': [paper()], 'notice': n}, body


def test_notice_is_independent_of_first_date_and_paper_subscription(tmp_path):
    catalog, body = notice_catalog()
    o = Opener({'catalog.json': json.dumps(catalog).encode(), 'reading-introduction/note.pdf': body})
    s = PaperService(tmp_path, opener=o, today=lambda: date(2026, 10, 12))
    s.set_auto_receive(False); s.refresh(threading.Event())
    assert s.first_launch == '2026-10-12'
    assert s.status()['notice']['downloaded'] and s.status()['notice']['notice']
    assert s.status()['papers'][0]['downloaded'] is False and s.status()['autoReceive'] is False
    assert s.asset('reading-introduction', 'original')['sha256'] == catalog['notice']['pdf']['sha256']
    before = len(o.calls); s.refresh(threading.Event())
    assert o.calls[before:] == [BASE_URL + 'catalog.json']
    with pytest.raises(PaperError): s.remove_download('reading-introduction')


@pytest.mark.parametrize('change', [lambda n: n['pdf'].update(path='../private.pdf'),
    lambda n: n['pdf'].update(sha256='bad'), lambda n: n.update(id='sample'), lambda n: n.update(extra='no')])
def test_notice_rejects_unexpected_fields_and_paths(change):
    catalog, _ = notice_catalog(); change(catalog['notice'])
    with pytest.raises(PaperError): validate_catalog(catalog)


def test_disabled_startup_fetches_notice_and_never_downloads_papers(tmp_path):
    from PyQt6.QtCore import QCoreApplication
    from ptb_desktop.papers_bridge import PapersBridge
    app = QCoreApplication.instance() or QCoreApplication([])
    bridge = PapersBridge(tmp_path); catalog, body = notice_catalog()
    bridge.service.opener = Opener({'catalog.json': json.dumps(catalog).encode(), 'reading-introduction/note.pdf': body})
    bridge.service.today = lambda: date(2026, 10, 12)
    bridge.service.set_auto_receive(False)
    try:
        bridge.automatic_check(startup=True)
        bridge.network_pool.submit(lambda: None).result(timeout=5)
        assert bridge.service.status()['notice']['downloaded']
        assert not bridge.service.status()['papers'][0]['downloaded']
        before = len(bridge.service.opener.calls); bridge.automatic_check()
        assert len(bridge.service.opener.calls) == before
    finally:
        bridge.close(); bridge.pool.shutdown(wait=True); bridge.network_pool.shutdown(wait=True)


def test_page_note_persists_and_exports_as_standard_pdf_annotation(tmp_path):
    buf = BytesIO(); writer = PdfWriter(); writer.add_blank_page(595, 842); writer.write(buf); writer.close()
    body = buf.getvalue(); p = paper()
    for k in ('original', 'translation'):
        p[k].update(size=len(body), sha256=hashlib.sha256(body).hexdigest())
    s, o = setup(tmp_path, [p]); o.items['sample/original.pdf'] = body
    s.download(s.first_launch, threading.Event(), lambda _: None)
    note = dict(id='page_note', page=1, kind='note', color='yellow', text='本页笔记', quote='', rects=[[.94, .025, .025, .025]])
    s.save_annotation('sample', 'original', 0, note, False)
    s.export('sample', 'original', True, tmp_path/'export.pdf')
    annotation = PdfReader(tmp_path/'export.pdf').pages[0]['/Annots'][0].get_object()
    assert annotation['/Subtype'] == '/Text' and '本页笔记' in annotation['/Contents']
    s.remove_download('sample'); s.download('0001-01-01', threading.Event(), lambda _: None, ['sample'])
    assert s.annotations('sample', 'original')['items'] == [note]


def test_personal_removal_calls_trash_only_for_selected_library_file(tmp_path):
    root = tmp_path/'library'; root.mkdir()
    writer = PdfWriter(); writer.add_blank_page(595, 842); writer.write(root/'owned.pdf'); writer.close()
    library = PersonalPaperLibrary(root); ident = library.entries()['documents'][0]['id']; calls = []
    def denied(path): calls.append(path); return False, ''
    with pytest.raises(PaperError, match='PDF 保留'): library.remove(ident, trash=denied)
    assert (root/'owned.pdf').exists() and calls == [str(root/'owned.pdf')]
    def owned_trash(path):
        from pathlib import Path
        Path(path).rename(tmp_path/'recycled-owned.pdf'); return True, str(tmp_path/'recycled-owned.pdf')
    assert library.remove(ident, trash=owned_trash)['documents'] == []
    assert (tmp_path/'recycled-owned.pdf').exists()
