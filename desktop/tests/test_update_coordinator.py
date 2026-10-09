import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from PyQt6.QtCore import QCoreApplication, QObject, QThread
import pytest
from ptb_desktop.update_coordinator import UpdateCoordinator
from ptb_desktop.updates import UpdateError, Package, _atomic_json
from ptb_desktop.updates_bridge import UpdatesBridge


@pytest.fixture(scope='module')
def app():return QCoreApplication.instance() or QCoreApplication([])


def pump(app,predicate):
    end=time.monotonic()+3
    while not predicate() and time.monotonic()<end:
        app.processEvents();threading.Event().wait(.003)
    assert predicate()


def fixture(app,tmp_path):
    window=QObject();window.closing=False
    window.bridge=SimpleNamespace(task_lock=threading.Lock(),preview_lock=threading.Lock(),recording_lock=threading.Lock(),_recording_bridge=None)
    window.vocal=SimpleNamespace(pending={});window.service=SimpleNamespace(local_files_root=None)
    package=Package('portable','fixture.zip','https://example.invalid',20,'a'*64,'server');path=tmp_path/'fixture.zip'
    service=SimpleNamespace(root=tmp_path,_downloads={'download':(path,package)},_download_versions={'download':'3.0.0-preview.2'},current=SimpleNamespace(value='3.0.0-preview.1'))
    window.updates_bridge=UpdatesBridge(window,service=service)
    calls=[]
    def prepare(*args):
        assert QThread.currentThread()!=app.thread()
        request=tmp_path/'apply'/'request.json';_atomic_json(request,{'fixture':True})
        return request,{'token':'owned-token'}
    def launch(request,plan):
        assert QThread.currentThread()==app.thread();calls.append('helper')
    def close_request():
        assert QThread.currentThread()==app.thread();calls.append('request-close')
    window.request_update_close=close_request
    coordinator=UpdateCoordinator(window,prepare=prepare,launch=launch)
    replies=[]
    window.updates_bridge.prepareClose.connect(lambda token:replies.append((token,QThread.currentThread()==app.thread())))
    return window,coordinator,calls,replies,path


def test_clear_cache_waits_for_actual_close_and_cancellation_preserves_everything(app,tmp_path,monkeypatch):
    window,coordinator,calls,replies,_=fixture(app,tmp_path)
    window.service.request=lambda *args:calls.append('results') or {'complete':True}
    from ptb_desktop import startup_cache
    monkeypatch.setattr(startup_cache,'request_clear',lambda:calls.append('runtime'))
    assert coordinator.clear_caches()['started'] and not calls
    token=replies[-1][0];coordinator.reply(token,True,'')
    assert 'results' not in calls and 'runtime' not in calls
    coordinator.cancel('取消关闭')
    assert not coordinator.closed() and 'results' not in calls
    coordinator.clear_caches();coordinator.reply(replies[-1][0],True,'')
    assert coordinator.closed() and calls[-2:]==['results','runtime']


@pytest.mark.parametrize('allow',[True,False])
def test_worker_queues_gui_guard_and_helper_waits_for_real_close(app,tmp_path,allow):
    window,coordinator,calls,replies,path=fixture(app,tmp_path)
    results=[]
    def invoke():
        try:results.append(coordinator.apply(path,'portable'))
        except UpdateError as error:results.append(error.code)
    thread=threading.Thread(target=invoke);thread.start()
    pump(app,lambda:bool(replies))
    assert replies==[('owned-token',True)] and not calls
    coordinator.reply('untrusted-token',True,'')
    assert not results and not calls
    coordinator.reply('owned-token',allow,'fixture cancellation')
    pump(app,lambda:bool(results));thread.join(3)
    if allow:
        pump(app,lambda:'request-close' in calls)
        assert results[0]['started'] and calls==['request-close']
        assert coordinator.closed() and calls==['request-close','helper']
    else:
        assert results==['APPLY_CANCELLED'] and not coordinator.closed() and not calls


@pytest.mark.parametrize('busy',['task','capture','processing','vocal'])
def test_busy_native_operation_never_prepares_or_closes(app,tmp_path,busy):
    window,coordinator,calls,replies,path=fixture(app,tmp_path)
    if busy=='task':window.bridge.task_lock.acquire()
    if busy=='capture':window.bridge._recording_bridge=SimpleNamespace(capturing=True,service=SimpleNamespace(job=None))
    if busy=='processing':window.bridge._recording_bridge=SimpleNamespace(capturing=False,service=SimpleNamespace(job={'active':True}))
    if busy=='vocal':window.vocal.pending={'active':True}
    with pytest.raises(UpdateError):coordinator.apply(path,'portable')
    assert not calls and not replies
