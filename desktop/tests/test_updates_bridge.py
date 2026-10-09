"""Real Qt signal dispatch, bounded owned requests and cancellation, without network/UI."""
import json
import threading
import time
from PyQt6.QtCore import QCoreApplication
import pytest
from ptb_desktop.updates_bridge import UpdatesBridge
from ptb_desktop.updates import UpdateError


@pytest.fixture(scope='module')
def app():
    return QCoreApplication.instance() or QCoreApplication([])


def wait_for(app, predicate):
    deadline=time.monotonic()+3
    while not predicate() and time.monotonic()<deadline:
        app.processEvents()
        threading.Event().wait(.005)
    assert predicate()


class Service:
    def __init__(self):
        self.calls=[]
        self.block=False
        self.started=threading.Event()
    def preferences(self):return {'source':'auto'}
    def stop_maintenance(self):pass
    def check(self, **args):
        self.calls.append(args)
        self.started.set()
        if self.block:
            args['cancel'].wait(2)
            if args['cancel'].is_set():raise UpdateError('CANCELLED','更新操作已取消。')
        return {'status':'incomplete','sources':{'server':{'status':'error'},'github':{'status':'error'}}}
    def download(self, release_id, **args):
        self.calls.append((release_id,args))
        args['progress']({'phase':'verified','received':20,'total':20,'source':'server'})
        return {'downloadId':'owned','verified':True}


def test_real_qt_signals_preferences_and_progress(app):
    service=Service();bridge=UpdatesBridge(service=service);ready=[];progress=[]
    bridge.ready.connect(lambda rid,payload:ready.append((rid,json.loads(payload))))
    bridge.progress.connect(lambda rid,payload:progress.append((rid,json.loads(payload))))
    try:
        bridge.request('prefs',json.dumps({'operation':'preferences'}))
        wait_for(app,lambda:len(ready)==1)
        assert ready[0]==('prefs',{'ok':True,'value':{'source':'auto'}})
        bridge.request('download',json.dumps({'operation':'download','args':{'releaseId':'selected','confirmed':True}}))
        wait_for(app,lambda:len(ready)==2 and len(progress)==1)
        assert progress[0][1]['phase']=='verified' and ready[1][1]['value']['verified']
        assert service.calls[0][0]=='selected'
    finally:bridge.close()


@pytest.mark.parametrize('payload',[json.dumps({'operation':'run','args':{'command':'malicious'}}),json.dumps({'operation':'download','args':{'url':'https://other.test'}}),json.dumps({'operation':'check','extra':'no'}),'not json','x'*9000])
def test_invalid_requests_never_reach_service(app,payload):
    service=Service();bridge=UpdatesBridge(service=service);ready=[]
    bridge.ready.connect(lambda rid,value:ready.append(json.loads(value)))
    try:
        bridge.request('bad',payload)
        wait_for(app,lambda:len(ready)==1)
        assert not ready[0]['ok'] and ready[0]['error']['code']=='REQUEST_INVALID'
        assert not service.calls
    finally:bridge.close()


def test_cancel_real_worker_request(app):
    service=Service();service.block=True;bridge=UpdatesBridge(service=service);ready=[]
    bridge.ready.connect(lambda rid,value:ready.append(json.loads(value)))
    try:
        bridge.request('check',json.dumps({'operation':'check'}))
        assert service.started.wait(1)
        bridge.cancel('check')
        wait_for(app,lambda:len(ready)==1)
        assert ready[0]['error']['code']=='CANCELLED'
        assert not bridge._jobs
    finally:bridge.close()


def test_duplicate_and_busy_requests_do_not_replace_cancellation(app):
    service=Service();service.block=True;bridge=UpdatesBridge(service=service);ready=[]
    bridge.ready.connect(lambda rid,value:ready.append((rid,json.loads(value))))
    try:
        for name in ('one','two','three'):bridge.request(name,json.dumps({'operation':'check'}))
        event=bridge._jobs['one']
        bridge.request('one',json.dumps({'operation':'check'}))
        assert bridge._jobs['one'] is event
        bridge.request('four',json.dumps({'operation':'check'}))
        wait_for(app,lambda:bool(ready))
        assert ready[0][0]=='four' and ready[0][1]['error']['code']=='BUSY'
    finally:
        bridge.close()
        assert event.is_set()


def test_close_cancels_owned_workers_and_ignores_late_ui_signal(app):
    service=Service();service.block=True;bridge=UpdatesBridge(service=service);ready=[]
    bridge.ready.connect(lambda rid,value:ready.append(value))
    bridge.request('check',json.dumps({'operation':'check'}))
    assert service.started.wait(1)
    event=bridge._jobs['check']
    bridge.close();bridge.close()
    assert event.is_set()
    wait_for(app,lambda:not bridge._jobs)
    bridge.request('after',json.dumps({'operation':'preferences'}))
    app.processEvents()
    assert not ready
