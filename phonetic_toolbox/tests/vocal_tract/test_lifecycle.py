import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.parse import urlparse
from urllib.request import urlopen
import pytest
from phonetic_toolbox.services.vocal_tract_service import VocalTractService

ROOT=Path(__file__).resolve().parents[3]


def eventually(predicate,seconds=10):
    end=time.monotonic()+seconds
    while time.monotonic()<end:
        if predicate():return
        time.sleep(.05)
    assert predicate()


def port_closed(url):
    try:
        with socket.create_connection(('127.0.0.1',urlparse(url).port),timeout=.2):return False
    except OSError:return True


def test_owned_worker_reuses_then_shuts_down_without_stopping_other_instance(tmp_path):
    a=VocalTractService(profile_dir=tmp_path/'a',silent=True)
    b=VocalTractService(profile_dir=tmp_path/'b',silent=True)
    try:
        first=a.launch();second=b.launch()
        assert first.success and second.success,(first,second)
        assert first.process_id!=second.process_id
        again=a.launch();assert again.url==first.url and again.process_id==first.process_id
        child=a.process;a.shutdown();assert child.poll() is not None
        eventually(lambda:port_closed(first.url))
        with urlopen(second.url+'/api/meta',timeout=3) as response:
            meta=json.load(response);assert meta['pid']==second.process_id and meta['audio_locked']
        assert not a.launch().success
    finally:a.shutdown();b.shutdown()


@pytest.mark.skipif(os.name!='nt',reason='Windows process-handle ownership')
def test_forced_parent_exit_closes_worker_and_port(tmp_path):
    report=tmp_path/'ready.json'
    code="""
import json,sys,time
from pathlib import Path
from phonetic_toolbox.services.vocal_tract_service import VocalTractService
s=VocalTractService(profile_dir=Path(sys.argv[1])/'profile',silent=True)
r=s.launch()
Path(sys.argv[1],'ready.json').write_text(json.dumps({'success':r.success,'url':r.url,'pid':r.process_id}),encoding='utf-8')
while True:time.sleep(.2)
"""
    with (tmp_path/'host.log').open('w',encoding='utf-8') as log:
        parent=subprocess.Popen([sys.executable,'-c',code,str(tmp_path)],cwd=ROOT,stdout=log,stderr=log)
    try:
        eventually(report.exists,60);ready=json.loads(report.read_text(encoding='utf-8'));assert ready['success']
        assert not port_closed(ready['url'])
        parent.kill();parent.wait(timeout=5)
        eventually(lambda:port_closed(ready['url']))
        from phonetic_toolbox.services.vocal_tract.process_guard import kernel
        k=kernel();handle=k.OpenProcess(0x100000,False,ready['pid'])
        if handle:
            try:assert k.WaitForSingleObject(handle,5000)==0
            finally:k.CloseHandle(handle)
    finally:
        if parent.poll() is None:parent.terminate();parent.wait(timeout=5)


def test_shutdown_during_startup_does_not_leave_orphan(tmp_path):
    import threading
    service=VocalTractService(profile_dir=tmp_path,silent=True)
    result=[];thread=threading.Thread(target=lambda:result.append(service.launch()))
    thread.start()
    eventually(lambda:service.process is not None)
    child=service.process;service.shutdown();thread.join(10)
    assert not thread.is_alive() and child.poll() is not None and not result[0].success
