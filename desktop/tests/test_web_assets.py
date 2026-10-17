"""Qt-thread, cancellation and path boundaries for local static resources."""
import os
import mimetypes
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest
from PyQt6 import sip
from PyQt6.QtCore import QByteArray,QObject,QThread,QTimer,QUrl
from PyQt6.QtWidgets import QApplication
from PyQt6.QtWebEngineCore import QWebEngineUrlRequestJob
from PyQt6.QtWebEngineWidgets import QWebEngineView

from ptb_desktop import web_assets

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from verification_artifacts import verification_run,write_json


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication(['owned-static-asset-tests'])


@pytest.fixture(scope='module')
def owned_files(request):
    with verification_run('desktop-assets','unit','Small synthetic static files, Qt QObject jobs, controlled worker waits and owned Windows junctions; no user profiles') as (out,scratch):
        yield scratch
        write_json(out/'report.json',dict(scope='Static resource Qt and filesystem boundaries',
            success=request.session.testsfailed==0))


@pytest.fixture
def root(owned_files,request):
    folder=owned_files/str(len(list(owned_files.iterdir())))
    folder.mkdir()
    return folder


def spin(app,condition):
    deadline=time.monotonic()+4
    while not condition():
        assert time.monotonic()<deadline,'Qt result did not arrive'
        app.processEvents()
        time.sleep(.002)


class Job(QObject):
    def __init__(self,path,*,host='app',method=b'GET'):
        super().__init__()
        self.url=QUrl('ptbapp://'+host+path)
        self.method=method
        self.replies=[];self.errors=[];self.reply_threads=[]

    def requestUrl(self):return self.url
    def requestMethod(self):return QByteArray(self.method)
    def reply(self,mime,device):
        self.replies.append((mime,bytes(device.data())))
        self.reply_threads.append(QThread.currentThread())
        assert device.parent() is self
    def fail(self,error):self.errors.append(error)


@pytest.mark.parametrize('name,mime',[
    ('module.js',b'text/javascript'),('module.mjs',b'text/javascript'),
    ('style.css',b'text/css'),('icon.svg',b'image/svg+xml'),
    ('chapter.json',b'application/json'),('sound.wav',None),
    ('unknown.ptbasset',b'application/octet-stream')])
def test_bytes_mime_and_reply_device_stay_on_gui(app,root,name,mime):
    # WAV uses the existing platform MIME registry, which differs on Windows.
    if mime is None:mime=mimetypes.guess_type(name)[0].encode('ascii')
    raw=b'\x00owned resource\xff';(root/name).write_bytes(raw)
    handler=web_assets.Assets(root,None);job=Job('/'+name)
    handler.requestStarted(job);spin(app,lambda:job.replies or job.errors)
    assert job.replies==[(mime,raw)] and not job.errors
    assert job.reply_threads==[app.thread()]
    assert not handler._pending


@pytest.mark.parametrize('name',['index.html','feedback-manager.html'])
def test_both_html_entries_keep_native_channel_bootstrap(app,root,name):
    (root/name).write_bytes(b'<html><head><title>Owned</title></head></html>')
    handler=web_assets.Assets(root,None);job=Job('/'+name)
    handler.requestStarted(job);spin(app,lambda:job.replies or job.errors)
    assert job.replies==[(b'text/html',b'<html><head><script src="qrc:///qtwebchannel/qwebchannel.js"></script><title>Owned</title></head></html>')]


def test_slow_read_leaves_gui_and_another_resource_responsive(app,root,monkeypatch):
    (root/'slow.txt').write_bytes(b'slow');(root/'fast.txt').write_bytes(b'fast')
    original=web_assets._read_asset;entered=threading.Event();release=threading.Event()
    workers=[];ticks=[]
    def read(folder,name):
        workers.append(QThread.currentThread()==app.thread())
        if name=='slow.txt':
            entered.set();assert release.wait(3)
        return original(folder,name)
    monkeypatch.setattr(web_assets,'_read_asset',read)
    handler=web_assets.Assets(root,None);slow=Job('/slow.txt');fast=Job('/fast.txt')
    try:
        handler.requestStarted(slow);spin(app,entered.is_set)
        QTimer.singleShot(0,lambda:ticks.append('GUI event processed'))
        handler.requestStarted(fast);spin(app,lambda:fast.replies or fast.errors)
        assert ticks and fast.replies==[(b'text/plain',b'fast')]
        assert not slow.replies and not slow.errors
    finally:release.set()
    spin(app,lambda:slow.replies or slow.errors)
    assert slow.replies==[(b'text/plain',b'slow')]
    assert workers and not any(workers)


@pytest.mark.parametrize('destroy',['job','handler'])
def test_destroy_during_read_discards_late_completion(app,root,monkeypatch,destroy):
    entered=threading.Event();release=threading.Event()
    def read(folder,name):
        entered.set();assert release.wait(3)
        return b'text/plain',b'late data'
    monkeypatch.setattr(web_assets,'_read_asset',read)
    handler=web_assets.Assets(root,None);job=Job('/slow.txt')
    replies,errors=job.replies,job.errors
    try:
        handler.requestStarted(job);spin(app,entered.is_set)
        sip.delete(job if destroy=='job' else handler)
        assert not handler._pending
    finally:release.set()
    assert web_assets._read_pool().waitForDone(3000)
    app.processEvents()
    assert not replies and not errors


def test_cancelled_queued_request_does_not_read_file(app,root,monkeypatch):
    release=threading.Event();lock=threading.Lock();reads=[]
    def read(folder,name):
        with lock:reads.append(name)
        assert release.wait(3)
        return b'text/plain',b'owned'
    monkeypatch.setattr(web_assets,'_read_asset',read)
    handler=web_assets.Assets(root,None);jobs=[Job('/busy'+str(i)) for i in range(4)]
    try:
        for job in jobs:handler.requestStarted(job)
        spin(app,lambda:len(reads)==4)
        cancelled=Job('/never-read');handler.requestStarted(cancelled);sip.delete(cancelled)
    finally:release.set()
    spin(app,lambda:all(job.replies for job in jobs))
    assert web_assets._read_pool().waitForDone(3000)
    assert 'never-read' not in reads and not handler._pending


@pytest.mark.parametrize('path,host,method,error',[
    ('/index.html','foreign',b'GET','RequestDenied'),
    ('/index.html','app',b'POST','RequestDenied'),
    ('/%2e%2e/outside','app',b'GET','UrlNotFound'),
    ('/a%5cb','app',b'GET','UrlNotFound'),
    ('/C%3a/test','app',b'GET','UrlNotFound'),
    ('/missing.png','app',b'GET','UrlNotFound'),
    ('/','app',b'GET','UrlNotFound')])
def test_denied_missing_and_unsafe_paths_keep_errors(app,root,path,host,method,error):
    handler=web_assets.Assets(root,None);job=Job(path,host=host,method=method)
    handler.requestStarted(job);spin(app,lambda:job.replies or job.errors)
    assert not job.replies and job.errors==[getattr(QWebEngineUrlRequestJob.Error,error)]


def test_unexpected_reader_error_finishes_request_without_escaping_worker(app,root,monkeypatch):
    def broken(*args):raise RuntimeError('controlled reader failure')
    monkeypatch.setattr(web_assets,'_read_asset',broken)
    handler=web_assets.Assets(root,None);job=Job('/broken')
    handler.requestStarted(job);spin(app,lambda:job.errors)
    assert job.errors==[QWebEngineUrlRequestJob.Error.RequestFailed]


@pytest.mark.skipif(os.name!='nt',reason='Windows junction path boundary')
def test_junction_and_replaced_root_cannot_read_outside(app,root):
    original_root=root/'assets';original_root.mkdir();outside=root/'outside';outside.mkdir()
    (outside/'sentinel.txt').write_bytes(b'outside stays private')
    canonical=original_root.resolve()
    def junction(link):
        subprocess.run(['powershell.exe','-NoProfile','-Command',
            'New-Item -ItemType Junction -Path $env:PTB_ASSET_TEST_LINK -Target $env:PTB_ASSET_TEST_TARGET | Out-Null'],
            env=dict(os.environ,PTB_ASSET_TEST_LINK=str(link),PTB_ASSET_TEST_TARGET=str(outside)),
            check=True,capture_output=True)
    nested=original_root/'nested'
    junction(nested)
    try:
        with pytest.raises(ValueError):web_assets._read_asset(canonical,'nested/sentinel.txt')
    finally:os.rmdir(nested)  # Remove only this owned junction entry.
    original_root.rename(root/'original-assets')
    junction(original_root)
    try:
        with pytest.raises(ValueError):web_assets._read_asset(canonical,'sentinel.txt')
    finally:os.rmdir(original_root)
    assert (outside/'sentinel.txt').read_bytes()==b'outside stays private'
