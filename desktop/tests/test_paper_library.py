"""M18 filesystem operations, acquisition boundaries and native scheduling."""
import hashlib
import json
from pathlib import Path
import threading

import pytest
from pypdf import PdfWriter
from ptb_desktop.paper_library import PersonalPaperLibrary
from ptb_desktop.papers import PaperError, PaperService
from test_papers import setup, paper, BODY


def test_cancelled_queued_requests_release_slots_without_raising_limit(tmp_path):
    from PyQt6.QtCore import QCoreApplication, Qt
    from ptb_desktop.papers_bridge import PapersBridge
    app = QCoreApplication.instance() or QCoreApplication([])
    bridge = PapersBridge(tmp_path)
    gate = threading.Event()
    blocker = bridge.pool.submit(gate.wait, 10)
    responses = []
    bridge.ready.connect(lambda _, value: responses.append(json.loads(value)), Qt.ConnectionType.DirectConnection)
    try:
        for n in range(40):
            key = 'cancel_' + str(n)
            bridge.request(key, json.dumps({'operation': 'status'}))
            bridge.cancel(key)
            assert not bridge.jobs and not bridge.futures
        assert not responses
        gate.set(); blocker.result(timeout=5)
        with bridge.lock:
            bridge.request('valid', json.dumps({'operation': 'status'}))
            future = bridge.futures['valid']
        future.result(timeout=5)
        assert len(responses) == 1 and responses[0]['ok']
        assert not bridge.jobs and not bridge.futures
    finally:
        gate.set(); bridge.close()
        bridge.pool.shutdown(wait=True); bridge.network_pool.shutdown(wait=True)


def test_cancelled_running_native_request_stays_counted_until_finished(tmp_path, monkeypatch):
    from PyQt6.QtCore import QCoreApplication, Qt
    from ptb_desktop.papers_bridge import PapersBridge
    app = QCoreApplication.instance() or QCoreApplication([])
    bridge = PapersBridge(tmp_path)
    started = threading.Event(); release = threading.Event(); responses = []
    original = bridge.service.status
    def delayed_status():
        started.set(); assert release.wait(10)
        return original()
    monkeypatch.setattr(bridge.service, 'status', delayed_status)
    bridge.ready.connect(lambda _, value: responses.append(value), Qt.ConnectionType.DirectConnection)
    try:
        with bridge.lock:
            bridge.request('running', json.dumps({'operation': 'status'}))
            future = bridge.futures['running']
        assert started.wait(5)
        bridge.cancel('running')
        assert future.running() and list(bridge.jobs) == ['running']
        for n in range(20):
            key = 'queued_' + str(n)
            bridge.request(key, json.dumps({'operation': 'status'})); bridge.cancel(key)
            assert list(bridge.jobs) == ['running']
        release.set(); future.result(timeout=5)
        assert not bridge.jobs and not bridge.futures and not responses
    finally:
        release.set(); bridge.close()
        bridge.pool.shutdown(wait=True); bridge.network_pool.shutdown(wait=True)


def pdf(path):
    writer = PdfWriter(); writer.add_blank_page(width=300, height=400)
    with path.open('wb') as stream: writer.write(stream)
    return path


def test_copy_move_restart_and_original_preserved(tmp_path):
    import os
    original = pdf(tmp_path / '原文件.pdf'); initial = original.read_bytes()
    os.utime(original, (1234567890, 1234567890))
    selected = tmp_path / '论文库'; selected.mkdir()
    service = PaperService(tmp_path / 'preferences')
    service.set_library_root(selected)
    library = service.library(); library.create_folder('方言')
    result = library.import_files([original], '')
    assert not result['failures'] and len(result['imported']) == 1
    document_id = result['imported'][0]
    assert result['documents'][0]['publishedAt'].startswith('2009-02')
    annotation = {'id': 'note', 'page': 1, 'kind': 'note', 'color': 'yellow', 'text': '保留', 'quote': 'test', 'rects': [[.1, .1, .2, .1]]}
    service.save_annotation(document_id, 'original', 0, annotation, False)
    result = library.move(document_id, '方言')
    assert not (selected / original.name).exists()
    assert (selected / '方言' / original.name).read_bytes() == initial == original.read_bytes()
    reopened = PaperService(tmp_path / 'preferences')
    assert reopened.library_status()['documents'][0]['folder'] == '方言'
    assert reopened.annotations(result['selected'], 'original')['items'][0]['text'] == '保留'
    result = reopened.library().import_files([original], '方言')
    assert not result['imported'] and '同名' in result['failures'][0]
    assert (selected / '方言' / original.name).read_bytes() == initial


def test_move_refuses_collision_and_traversal(tmp_path):
    library = PersonalPaperLibrary(tmp_path); library.create_folder('收藏')
    pdf(tmp_path / 'one.pdf'); pdf(tmp_path / '收藏' / 'one.pdf')
    item = next(p for p in library.entries()['documents'] if p['folder'] == '')
    with pytest.raises(PaperError): library.move(item['id'], '收藏')
    assert (tmp_path / 'one.pdf').exists()
    for value in ('../outside', 'C:/outside', '..\\outside'):
        with pytest.raises(PaperError): library.safe(value)
    for value in ('..', 'a/b', 'CON', '文件夹.', 'LPT1', 'x:secret'):
        with pytest.raises(PaperError): library.create_folder(value)


def test_partial_import_keeps_valid_pdf_without_publishing_invalid_file(tmp_path):
    root = tmp_path / 'library'; root.mkdir()
    good = pdf(tmp_path / 'good.pdf'); bad = tmp_path / 'bad.pdf'; bad.write_bytes(b'not a PDF')
    library = PersonalPaperLibrary(root)
    value = library.import_files([good, bad], '')
    assert len(value['imported']) == 1 and len(value['failures']) == 1
    assert (root / 'good.pdf').exists() and not (root / 'bad.pdf').exists()
    assert not list(root.glob('*.tmp'))


def test_selected_download_does_not_fetch_unchecked_or_future(tmp_path):
    old = paper('2026-10-06', 'old'); raw = BODY + b'old'
    for key in ('original', 'translation'):
        old[key].update(size=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    service, opener = setup(tmp_path, [old, paper(), paper('2026-10-08', 'future')])
    service.download('0001-01-01', threading.Event(), lambda _: None, ['sample'])
    assert 'old/original.pdf' not in [url.removeprefix('https://www.phonetictoolbox.com/papers/') for url in opener.calls]
    before = len(opener.calls)
    service.download('0001-01-01', threading.Event(), lambda _: None, ['sample'])
    assert len(opener.calls) == before
    with pytest.raises(PaperError): service.download('0001-01-01', threading.Event(), lambda _: None, ['future'])
    assert service.first_launch == '2026-10-07'


def test_preferences_and_recommendation_folders_survive_restart(tmp_path):
    service, _ = setup(tmp_path)
    service.set_auto_receive(False); service.set_interval(12)
    service.recommendation_folder('构音'); service.collect_recommendation('sample', '构音')
    restarted = PaperService(tmp_path)
    assert restarted.status()['autoReceive'] is False
    assert restarted.status()['checkIntervalHours'] == 12
    assert restarted.status()['collections']['sample'] == '构音'
    for value in (0, -1, float('nan'), True, 721):
        with pytest.raises(PaperError): restarted.set_interval(value)


def test_native_schedule_and_disabled_startup(tmp_path):
    from PyQt6.QtCore import QCoreApplication
    from ptb_desktop.papers_bridge import PapersBridge
    app = QCoreApplication.instance() or QCoreApplication([])
    bridge = PapersBridge(tmp_path)
    try:
        bridge.service.set_auto_receive(False)
        bridge.start_maintenance()
        assert not bridge.timer.isActive()
        bridge.service.set_auto_receive(True); bridge.update_schedule()
        assert bridge.timer.interval() == 6 * 3600000
        bridge.service.set_interval(12); bridge.update_schedule()
        assert bridge.timer.interval() == 12 * 3600000
        bridge.service.set_auto_receive(False); bridge.update_schedule()
        assert not bridge.timer.isActive()
    finally: bridge.close()


def test_supplement_for_acquired_old_paper_downloads_only_analysis(tmp_path):
    from ptb_desktop.papers import validate_catalog
    service, opener = setup(tmp_path, [paper('2026-10-06', 'sample')])
    service.download('0001-01-01', threading.Event(), lambda _: None, ['sample'])
    updated = paper('2026-10-06', 'sample')
    body = BODY + b'analysis'
    updated.update(updatedAt='2026-10-07', analysisPdf={'path': 'sample/analysis.pdf',
        'size': len(body), 'sha256': hashlib.sha256(body).hexdigest()})
    service.catalog = validate_catalog({'schema': 'ptb-papers/1', 'papers': [updated]})
    opener.items['sample/analysis.pdf'] = body
    assert service.status()['papers'][0]['acquired']
    assert not service.status()['papers'][0]['downloaded']
    before = len(opener.calls)
    service.download(service.first_launch, threading.Event(), lambda _: None)
    assert opener.calls[before:] == ['https://www.phonetictoolbox.com/papers/sample/analysis.pdf']
    assert service.status()['papers'][0]['downloaded']
    updated['analysisPdf']['localPath'] = 'C:/private/secret.pdf'
    with pytest.raises(ValueError): validate_catalog({'schema': 'ptb-papers/1', 'papers': [updated]})


def test_application_startup_checks_once_and_honors_disabled_preference(tmp_path):
    from PyQt6.QtCore import QCoreApplication, QEventLoop, QTimer
    from ptb_desktop.papers_bridge import PapersBridge
    app = QCoreApplication.instance() or QCoreApplication([])
    calls = []
    bridge = PapersBridge(tmp_path)
    bridge.service.refresh = lambda cancel: calls.append('refresh')
    bridge.service.download = lambda *args: (calls.append('download') or bridge.service.status())
    try:
        bridge.start_maintenance(); bridge.start_maintenance()
        loop = QEventLoop(); QTimer.singleShot(1600, loop.quit); loop.exec()
        assert calls == ['refresh', 'download']
        bridge.service.set_auto_receive(False); bridge.update_schedule(); bridge.automatic_check()
        assert calls == ['refresh', 'download'] and not bridge.timer.isActive()
    finally:
        bridge.close(); bridge.pool.shutdown(wait=True); bridge.network_pool.shutdown(wait=True)


def test_delete_recommendation_folder_preserves_acquired_paper(tmp_path):
    service, _ = setup(tmp_path)
    service.download(service.first_launch, threading.Event(), lambda _: None)
    service.recommendation_folder('构音'); service.recommendation_folder('声学')
    service.collect_recommendation('sample', '构音')
    before = {p.name: p.read_bytes() for p in service.files.iterdir()}
    result = service.delete_recommendation_folder('构音')
    assert result['folders'] == ['声学'] and 'sample' not in result['collections']
    assert result['papers'][0]['downloaded']
    assert before == {p.name: p.read_bytes() for p in service.files.iterdir()}
    assert PaperService(tmp_path).status()['folders'] == ['声学']
    with pytest.raises(PaperError): service.delete_recommendation_folder('')


def test_delete_local_folder_preserves_all_content_and_annotations(tmp_path):
    root = tmp_path / 'library'; root.mkdir()
    parent = root / '方言'; parent.mkdir()
    selected = parent / '舌叶'; selected.mkdir(); (selected / '附件').mkdir()
    original = pdf(selected / '论文.pdf'); nested = pdf(selected / '附件' / '附录.pdf')
    (selected / '说明.txt').write_text('文件也必须保留', 'utf8')
    before = [original.read_bytes(), nested.read_bytes()]
    service = PaperService(tmp_path / 'state'); service.set_library_root(root)
    item = next(p for p in service.library_status()['documents'] if p['title'] == '论文')
    service.save_annotation(item['id'], 'original', 0, {'id': 'note', 'page': 1, 'kind': 'note', 'color': 'yellow', 'text': '保留', 'quote': '', 'rects': [[.1, .1, .2, .1]]}, False)
    service.close_documents()
    result = service.library().delete_folder('方言/舌叶')
    assert not selected.exists()
    assert [p['folder'] for p in result['documents']] == ['方言', '方言/附件']
    assert [p.read_bytes() for p in (parent / '论文.pdf', parent / '附件' / '附录.pdf')] == before
    assert (parent / '说明.txt').read_text('utf8') == '文件也必须保留'
    assert service.annotations(result['relocated'][item['id']], 'original')['items'][0]['text'] == '保留'
    with pytest.raises(PaperError): service.library().delete_folder('')
    with pytest.raises(PaperError): service.library().delete_folder('../outside')


def test_delete_folder_collision_stops_before_any_move(tmp_path):
    selected = tmp_path / '收藏'; selected.mkdir()
    pdf(selected / 'first.pdf'); pdf(selected / 'same.pdf'); pdf(tmp_path / 'same.pdf')
    library = PersonalPaperLibrary(tmp_path)
    before = {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob('*.pdf')}
    with pytest.raises(PaperError, match='未移动'): library.delete_folder('收藏')
    assert selected.is_dir()
    assert before == {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob('*.pdf')}


def test_delete_folder_failed_move_rolls_back(tmp_path, monkeypatch):
    import os
    selected = tmp_path / '收藏'; selected.mkdir()
    pdf(selected / 'a.pdf'); pdf(selected / 'b.pdf')
    original_rename = os.rename
    def fail_second(source, target):
        if Path(source).name == 'b.pdf': raise PermissionError('held open')
        return original_rename(source, target)
    monkeypatch.setattr(os, 'rename', fail_second)
    with pytest.raises(PaperError, match='原位置'): PersonalPaperLibrary(tmp_path).delete_folder('收藏')
    assert selected.is_dir() and len(list(selected.iterdir())) == 2
    assert not list(tmp_path.glob('*.pdf'))
