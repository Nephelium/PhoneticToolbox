"""Pinned delivery safety and progress. Native probe is tested separately."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from ptb_worker.mfa import delivery
from ptb_worker.mfa.components import ComponentManager, digest


def fixture(tmp_path, monkeypatch):
    files = {'env/python.exe': b'fixture only', 'models/mandarin_mfa.zip': b'model',
             'models/mandarin_mfa.dict': b'a a'}
    rows = [dict(path=p, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()) for p, raw in files.items()]
    meta = dict(schema='m11-component/1', id='fixture', platform='windows', arch='x86_64',
                mfa_version='3.3.8', source='test fixture', dependencies=[], files=rows,
                installed_bytes=sum(r['bytes'] for r in rows))
    raw = json.dumps(meta).encode()
    files[delivery.MANIFEST] = raw
    path = tmp_path/'package.zip'
    with zipfile.ZipFile(path,'w') as z:
        for name, content in files.items():z.writestr(name,content)
    pinned = dict(id='fixture',manifest_sha256=hashlib.sha256(raw).hexdigest(),sha256=digest(path),
                  download_bytes=path.stat().st_size,installed_bytes=1000,
                  url='https://www.phonetictoolbox.com/releases/components/mfa/3.3.8-r1/fixture.zip',trusted_host='www.phonetictoolbox.com')
    monkeypatch.setattr(delivery.release_catalog,'RELEASE',pinned)
    return path,files,pinned


def test_archive_uses_app_pin_not_internal_self_claim(tmp_path,monkeypatch):
    path,files,pinned=fixture(tmp_path,monkeypatch)
    trusted=delivery.archive_manifest(path,pinned)
    assert len(trusted['files'])==4
    with zipfile.ZipFile(path,'a') as z:z.writestr('env/evil.pth',b'import bad')
    with pytest.raises(ValueError,match='archive_hash'):delivery.archive_manifest(path,pinned)


@pytest.mark.parametrize('change',['missing','extra','changed','manifest'])
def test_extracted_folder_rejects_untrusted_contents(tmp_path,monkeypatch,change):
    path,files,pinned=fixture(tmp_path,monkeypatch)
    folder=tmp_path/'extracted'
    for name,raw in files.items():
        if change=='missing' and name.endswith('.dict'):continue
        p=folder/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
    if change=='extra':(folder/'env/evil.pth').write_bytes(b'import bad')
    if change=='changed':(folder/'env/python.exe').write_bytes(b'bad')
    if change=='manifest':(folder/delivery.MANIFEST).write_bytes(b'{}')
    with pytest.raises(ValueError):delivery.folder_manifest(folder,pinned)


def test_fresh_install_supplies_bundled_resources_and_failed_probe_preserves_registry(tmp_path,monkeypatch):
    path,_,_=fixture(tmp_path,monkeypatch)
    from ptb_worker.mfa import probe
    root=tmp_path/'components';root.mkdir();state=root/'registry.json';state.write_text('{"old":true}')
    def fail(runtime,model,dictionary,**kwargs):
        assert runtime.name=='env' and model.name=='mandarin_mfa.zip'
        assert dictionary.name=='mandarin_mfa.dict' and kwargs['publish'] is False
        assert json.loads(state.read_text())=={'old':True}
        raise ValueError('m11_self_test_failed')
    monkeypatch.setattr(probe,'register',fail)
    with pytest.raises(ValueError,match='self_test'):delivery.install('install_zip',archive=path,root=root)
    assert json.loads(state.read_text())=={'old':True}
    assert delivery.status()['stage']=='failed'


def test_rate_excludes_resumed_bytes_and_stalls_decay(monkeypatch):
    clock=[100.]
    monkeypatch.setattr(delivery.time,'monotonic',lambda:clock[0])
    tick=delivery.download_progress();tick(500_000_000,900_000_000)
    clock[0]+=1;tick(502_000_000,900_000_000)
    assert delivery.status()['bytes_per_second']==2_000_000
    assert delivery.status()['completed']==502_000_000
    clock[0]+=4
    assert delivery.status()['bytes_per_second']==0


def test_https_download_reports_actual_bytes_and_verifies_hash(tmp_path,monkeypatch):
    path,_,pinned=fixture(tmp_path,monkeypatch)
    import io
    raw=path.read_bytes();calls=[]
    class Response(io.BytesIO):
        status=200
        headers={}
    monkeypatch.setattr('urllib.request.build_opener',lambda *_:SimpleNamespace(open=lambda *a,**k:Response(raw)))
    result=ComponentManager(tmp_path/'components').download(pinned,progress=lambda *v:calls.append(v))
    assert result.read_bytes()==raw and calls[0]==(0,len(raw)) and calls[-1]==(len(raw),len(raw))
    pinned['sha256']='f'*64
    with pytest.raises(ValueError,match='archive_hash'):ComponentManager(tmp_path/'other').download(pinned)
    assert not (tmp_path/'other/downloads'/('f'*64+'.part')).exists()
    assert len(list((tmp_path/'other/downloads').glob('*.rejected-*')))==1


def test_local_api_does_not_require_model_for_install_and_does_not_expose_server_install(tmp_path,monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from ptb_api.m11_components import component_router
    calls=[]
    monkeypatch.setattr(delivery,'install',lambda action,**kw:calls.append((action,kw)) or {'success':True})
    for mode,expected in [('local',200),('server',404)]:
        app=FastAPI();app.include_router(component_router(mode,lambda:None))
        response=TestClient(app).post('/m11/component',json={'action':'install_zip','archive':str(tmp_path/'a.zip')})
        assert response.status_code==expected
    assert len(calls)==1 and calls[0][0]=='install_zip'


def test_interrupted_download_resumes_only_with_exact_range(tmp_path,monkeypatch):
    path,_,pinned=fixture(tmp_path,monkeypatch)
    import io
    raw=path.read_bytes();cut=len(raw)//2;requests=[]
    class Response(io.BytesIO):
        def __init__(self,data,status,headers):super().__init__(data);self.status=status;self.headers=headers
    replies=[Response(raw[:cut],200,{}),Response(raw[cut:],206,{'Content-Range':f'bytes {cut}-{len(raw)-1}/{len(raw)}'})]
    def open_response(request,**kwargs):requests.append(request.get_header('Range'));return replies.pop(0)
    monkeypatch.setattr('urllib.request.build_opener',lambda *_:SimpleNamespace(open=open_response))
    manager=ComponentManager(tmp_path/'components')
    with pytest.raises(ValueError,match='download_incomplete'):manager.download(pinned)
    assert manager.download(pinned).read_bytes()==raw
    assert requests==['bytes=0-',f'bytes={cut}-']


def test_download_404_is_actionable_and_keeps_existing_state(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from urllib.error import HTTPError
    from ptb_api.m11_components import component_router
    def fail(*a,**k):raise HTTPError('https://example.invalid',404,'missing',{},None)
    monkeypatch.setattr(delivery,'install',fail)
    app=FastAPI();app.include_router(component_router('local',lambda:None))
    response=TestClient(app).post('/m11/component',json={'action':'download'})
    assert response.status_code==422 and response.json()['detail']=='m11_download_not_uploaded'
