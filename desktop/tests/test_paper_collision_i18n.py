"""Real folder collision refuses every move and sends an existing display envelope."""
import json
import threading
from types import SimpleNamespace
from PyQt6.QtCore import QCoreApplication, Qt
import pytest
from ptb_desktop.papers import PaperError
from ptb_desktop.paper_library import PersonalPaperLibrary
from ptb_desktop.papers_bridge import PapersBridge
from ptb_desktop.native_i18n import NativeLocale


def test_folder_collision_descriptor_keeps_original_error_and_all_files(tmp_path):
    root = tmp_path / 'library'; root.mkdir()
    folder = root / 'collection'; folder.mkdir()
    filename = '说明-{page}-é.txt'
    (root / filename).write_bytes(b'parent contents')
    (folder / filename).write_bytes(b'child contents')
    (folder / 'another.txt').write_bytes(b'must not move')
    library = PersonalPaperLibrary(root)
    before = {p.relative_to(root).as_posix():p.read_bytes() for p in root.rglob('*') if p.is_file()}
    raw = '上一级已有同名文件或文件夹：' + filename + '。未移动文件，也未删除收藏夹。'
    with pytest.raises(PaperError) as caught:library.delete_folder('collection')
    assert str(caught.value) == raw
    assert caught.value.code == 'paper_folder_collision' and caught.value.filename == filename
    app = QCoreApplication.instance() or QCoreApplication([])
    bridge = PapersBridge(tmp_path / 'preferences')
    bridge.service = SimpleNamespace(close_documents=lambda:None,library=lambda:library)
    responses=[]
    bridge.ready.connect(lambda _, payload: responses.append(json.loads(payload)),Qt.ConnectionType.DirectConnection)
    try:
        bridge.execute('collision',json.dumps({'operation':'deleteFolder','args':{'folder':'collection'}}),threading.Event())
        assert responses == [{'ok':False,'error':raw,'display':{'key':'native.errors.papers.folderCollision','values':{'filename':filename}}}]
        language=NativeLocale()
        for code in ('zh-CN','en','ja'):
            language.select(code)
            assert filename in language.text('errors.papers.folderCollision',filename=filename)
        assert before == {p.relative_to(root).as_posix():p.read_bytes() for p in root.rglob('*') if p.is_file()}
        assert folder.is_dir() and not (root / 'another.txt').exists()
    finally:
        bridge.close();bridge.pool.shutdown(wait=True);bridge.network_pool.shutdown(wait=True)


@pytest.mark.parametrize('reason',['操作已取消。','未登记诊断-{reason} / private.pdf'])
def test_real_background_failure_keeps_raw_reason_in_existing_envelope(tmp_path,reason):
    app = QCoreApplication.instance() or QCoreApplication([])
    bridge = PapersBridge(tmp_path)
    def fail(_):raise PaperError(reason)
    bridge.service = SimpleNamespace(refresh=fail,status=lambda:{'warning':'','papers':[]},close_documents=lambda:None)
    responses=[]
    bridge.statusChanged.connect(lambda value:responses.append(json.loads(value)),Qt.ConnectionType.DirectConnection)
    try:
        bridge.background_check(threading.Event())
        assert responses == [{'warning':'自动检查未完成：'+reason,'papers':[],'warningError':{'ok':False,'error':reason}}]
    finally:
        bridge.close();bridge.pool.shutdown(wait=True);bridge.network_pool.shutdown(wait=True)
