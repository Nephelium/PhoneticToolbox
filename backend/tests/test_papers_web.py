"""Real managed storage, Qt PDF child and HTTP ownership/expiry boundary."""
import io
import json
import threading
from pathlib import Path
from uuid import uuid4
import pytest
from pypdf import PdfReader, PdfWriter
from pypdf.generic import NameObject, DictionaryObject, DecodedStreamObject
from ptb_api.papers import WebPapers, NAME, run_child, relative
from ptb_api.storage_models import UploadInput
from ptb_api.quota import StorageError
from test_p07_policy_postgres import legacy
from test_web_workspaces_postgres import web, device, expire
from web_postgres_support import cluster


def pdf():
    writer = PdfWriter()
    for text in ('Paper reading fixture', 'Second page phonetics'):
        page = writer.add_blank_page(width=600, height=800)
        font = DictionaryObject({NameObject('/Type'): NameObject('/Font'), NameObject('/Subtype'): NameObject('/Type1'), NameObject('/BaseFont'): NameObject('/Helvetica')})
        page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): writer._add_object(font)})})
        stream = DecodedStreamObject(); stream.set_data(f'BT /F1 18 Tf 50 700 Td ({text}) Tj ET'.encode())
        page[NameObject('/Contents')] = writer._add_object(stream)
    target = io.BytesIO(); writer.write(target); writer.close(); return target.getvalue()


def upload(e, project, raw=None, name='测试论文.pdf'):
    raw = pdf() if raw is None else raw
    a = e.storage.create(e.owner, UploadInput(project_id=project, name=name, expected_bytes=len(raw), idempotency_key=uuid4().hex))
    e.storage.append(e.owner, a['id'], 0, raw)
    return e.storage.finalize(e.owner, a['id'])


def call(client, project, operation, **args):
    response = client.post(f'/api/v1/papers/{project}/request', json={'operation': operation, 'args': args})
    assert response.status_code == 200, response.text
    assert response.headers['cache-control'] == 'no-store'
    events = [json.loads(line) for line in response.text.splitlines()]
    assert events[-1]['ok'], events[-1]
    return events[-1]['value']


def test_complete_pdf_library_notes_export_and_recycle(web):
    e = web; client, _, _, project = device(e)
    state = call(client, project, 'status'); assert state['autoReceive']
    library = call(client, project, 'library'); assert library['root'] == '我的 PDF'
    source = upload(e, project)
    library = call(client, project, 'importLocal', folder='', files=[{'asset_id': source['id'], 'relative': source['name']}])
    assert library['failures'] == [] and len(library['documents']) == 1
    selected = library['documents'][0]['id']
    page = call(client, project, 'render', id=selected, language='original', page=1, width=1020)
    assert page['pages'] == 2 and 'Paper reading fixture' in page['text'] and page['lines']
    assert page['imageWidth'] == 1020 and page['image'].startswith('data:image/png;base64,')
    item = dict(id='note1', page=1, kind='note', color='yellow', text='中文笔记与音标', quote='Paper', rects=[[.1,.1,.2,.04]])
    assert call(client, project, 'saveAnnotation', id=selected, language='original', revision=0, item=item, remove=False)['revision'] == 1
    conflict = client.post(f'/api/v1/papers/{project}/request', json={'operation': 'saveAnnotation', 'args': dict(id=selected, language='original', revision=0, item=item, remove=False)})
    assert '另一窗口更改' in conflict.text
    highlight = dict(item, id='mark1', kind='highlight', text='', color='green')
    call(client, project, 'saveAnnotation', id=selected, language='original', revision=1, item=highlight, remove=False)
    raw_export = call(client, project, 'export', id=selected, language='original', annotated=False)
    assert client.get('/api/v1/assets/'+raw_export['asset_id']+'/content').content == pdf()
    export = call(client, project, 'export', id=selected, language='original', annotated=True)
    exported = PdfReader(io.BytesIO(client.get('/api/v1/assets/'+export['asset_id']+'/content').content))
    annotations = [a.get_object() for a in exported.pages[0]['/Annots']]
    assert {a['/Subtype'] for a in annotations} == {'/Text','/Highlight'}
    assert '中文笔记' in annotations[0]['/Contents']
    call(client, project, 'createFolder', name='语音学')
    moved = call(client, project, 'moveLocal', id=selected, folder='语音学'); selected = moved['selected']
    assert len(call(client, project, 'annotations', id=selected, language='original')['items']) == 2
    relocated = call(client, project, 'deleteFolder', folder='语音学'); selected = relocated['relocated'][selected]
    call(client, project, 'removeLocal', id=selected)
    assert call(client, project, 'library')['documents'] == []
    trash = call(client, project, 'trash'); assert len(trash) == 1
    restored = call(client, project, 'restoreLocal', id=trash[0]['id'])
    selected = restored['documents'][0]['id']
    assert len(call(client, project, 'annotations', id=selected, language='original')['items']) == 2
    assert call(client, project, 'trash') == []
    service = WebPapers(e.storage)
    snapshot = service.read(e.owner, service.current(e.owner, project))
    assert source['id'] in snapshot['files'].values()  # no second copy consumes quota
    for value in snapshot['json'].values(): assert str(e.root) not in json.dumps(value)
    e.storage.recover()  # durable ledger and physical files remain consistent


def test_folder_import_name_conflicts_and_recommendation_settings(web):
    e=web;client,_,_,project=device(e);call(client,project,'status');a=upload(e,project)
    library=call(client,project,'chooseLibrary',name='文献',files=[{'asset_id':a['id'],'relative':'nested/测试论文.pdf'}])
    assert library['root']=='文献' and library['folders']==['nested']
    duplicate=call(client,project,'importLocal',folder='nested',files=[{'asset_id':a['id'],'relative':'测试论文.pdf'}])
    assert len(duplicate['documents'])==1 and '同名文件' in duplicate['failures'][0]
    assert not call(client,project,'autoReceive',enabled=False)['autoReceive']
    assert call(client,project,'interval',hours=.25)['checkIntervalHours']==.25
    assert call(client,project,'createRecommendationFolder',name='经典')['folders']==['经典']
    assert call(client,project,'deleteRecommendationFolder',folder='经典')['folders']==[]
    invalid=client.post(f'/api/v1/papers/{project}/request',json={'operation':'render','args':{'id':library['documents'][0]['id'],'language':'original','page':1,'width':100000}})
    assert '阅读参数无效' in invalid.text


def test_account_workspace_csrf_and_expiry_boundaries(web):
    e=web;a,_,lease,project=device(e);b,_,_,_=device(e)
    call(a,project,'status')
    body={'operation':'status','args':{}}
    assert b.post(f'/api/v1/papers/{project}/request',json=body).status_code==404
    assert a.post(f'/api/v1/papers/{project}/request',json=body,headers={'X-CSRF-Token':'bad'}).status_code==403
    assert a.post(f'/api/v1/papers/{project}/request',json=body,headers={'Origin':'https://other.test'}).status_code==403
    expire(e,lease)
    assert a.post(f'/api/v1/papers/{project}/request',json=body).status_code==410
    e.storage.cleanup()
    assert e.conn.execute("SELECT count(*) n FROM ptb_storage.assets WHERE project_id=%s AND state!='deleted'",(project,)).fetchone()['n']==0


def test_quota_failure_preserves_previous_snapshot_and_revision(web):
    e=web;client,_,_,project=device(e);call(client,project,'status')
    before=WebPapers(e.storage).current(e.owner,project)
    e.conn.execute('UPDATE ptb_storage.quota_accounts SET quota_bytes=used_bytes WHERE owner_id=%s',(e.owner,))
    response=client.post(f'/api/v1/papers/{project}/request',json={'operation':'interval','args':{'hours':2}})
    assert 'quota_' in response.text
    assert WebPapers(e.storage).current(e.owner,project)['id']==before['id']
    assert call(client,project,'status')['checkIntervalHours']==6
    e.storage.recover()


def test_cancelled_request_retires_private_files_without_publishing(web):
    e=web;client,_,_,project=device(e);cancel=threading.Event();roots=[]
    def runner(root,*args):
        roots.append(root);cancel.set();return {'ok':True,'value':{}}
    service=WebPapers(e.storage,runner=runner)
    with pytest.raises(StorageError,match='paper_cancelled'):
        service.call(e.owner,project,'status',{},cancel,lambda _:None,lambda:None)
    assert not roots[0].exists() and service.current(e.owner,project) is None


@pytest.mark.parametrize('value',['../outside','/absolute','C:/data','nested\\path','hidden/.file','a//b'])
def test_untrusted_paths_are_rejected(value):
    with pytest.raises(StorageError):relative(value)


def test_real_browser_flow(web, cluster):
    import os, socket, subprocess, time
    if os.environ.get('PTB_M18_BROWSER')!='1': pytest.skip('explicit standalone browser opt-in required')
    from starlette.staticfiles import StaticFiles
    from ptb_api.main import create_app
    from ptb_api.auth import AuthSettings
    import uvicorn
    e=web;client,_,_,project=device(e)
    call(client,project,'status');call(client,project,'autoReceive',enabled=False)
    root=Path(__file__).resolve().parents[2]
    out=root/'output/validation/web-paper-reading'
    _,scratch=cluster
    source=scratch/'测试论文.pdf';source.write_bytes(pdf())
    export=scratch/'browser-export.pdf'
    sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    origin=f'http://127.0.0.1:{port}'
    app=create_app(account_store=e.accounts,auth_settings=AuthSettings(origin=origin,signing_key='m18-browser-test-'*4,allow_insecure_loopback=True),job_store=e.jobs,storage=e.storage,workspace_service=e.workspaces)
    app.mount('/app',StaticFiles(directory=out/'build',html=True),name='m18-test-ui')
    server=uvicorn.Server(uvicorn.Config(app,log_level='error'))
    thread=threading.Thread(target=lambda:server.run(sockets=[sock]));thread.start()
    try:
        until=time.monotonic()+15
        while not server.started and time.monotonic()<until:time.sleep(.05)
        assert server.started
        env={**os.environ,'PTB_M18_ORIGIN':origin,'PTB_M18_OUTPUT':str(out),'PTB_M18_PDF':str(source),'PTB_M18_EXPORT':str(export),'PTB_M18_COOKIE':client.cookies.get('__Host-ptb-session')}
        result=subprocess.run(['node',str(root/'scripts/verify_web_papers.cjs')],env=env,capture_output=True,text=True,encoding='utf8',timeout=240)
        (out/'browser.log').write_text(result.stdout+result.stderr,encoding='utf8')
        assert result.returncode==0,result.stdout+result.stderr
        reader=PdfReader(export);assert {a.get_object()['/Subtype'] for a in reader.pages[0]['/Annots']}=={'/Text','/Highlight'}
    finally:
        server.should_exit=True;thread.join(timeout=15);sock.close();assert not thread.is_alive()


def test_live_recommendations_and_three_pdf_channels(web):
    import os
    if os.environ.get('PTB_M18_NETWORK')!='1':pytest.skip('explicit public paper download opt-in required')
    e=web;client,_,_,project=device(e);call(client,project,'status')
    status=call(client,project,'refresh');assert status['papers']
    candidates=[p for p in status['papers'] if p.get('analysisPdf')]
    selected=min(candidates,key=lambda p:p['original']['size']+p['translation']['size']+p['analysisPdf']['size'])
    status=call(client,project,'downloadSelected',ids=[selected['id']])
    assert next(p for p in status['papers'] if p['id']==selected['id'])['downloaded']
    for language in ('original','translation','analysis'):
        page=call(client,project,'render',id=selected['id'],language=language,page=1,width=1020)
        assert page['pages']>0 and page['text'].strip() and page['lines']
    call(client,project,'removeDownload',id=selected['id'])
    assert next(p for p in call(client,project,'status')['papers'] if p['id']==selected['id'])['dismissed']
    restored=call(client,project,'downloadSelected',ids=[selected['id']])
    assert next(p for p in restored['papers'] if p['id']==selected['id'])['downloaded']
    if status.get('notice'):
        notice=status['notice'];call(client,project,'removeDownload',id=notice['id'])
        assert call(client,project,'status')['notice']['dismissed']
        assert not call(client,project,'downloadSelected',ids=[notice['id']])['notice']['dismissed']
    out=Path(__file__).resolve().parents[2]/'output/validation/web-paper-reading'
    out.mkdir(parents=True,exist_ok=True)
    (out/'network.json').write_text(json.dumps({'success':True,'catalogCount':len(status['papers']),'id':selected['id'],'channels':['original','translation','analysis'],'removeRestore':True,'noticeRemoveRestore':bool(status.get('notice'))},ensure_ascii=False,indent=2),encoding='utf8')


def test_partial_cancel_expiry_and_orphan_scratch_lifecycle(web):
    import hashlib, os, tempfile
    e=web;client,key,lease,project=device(e);service=WebPapers(e.storage)
    marker=json.loads((e.root/'.ptb-storage.json').read_text('utf8'))['instance_id']
    with tempfile.TemporaryDirectory(prefix='ptb-papers-',dir=e.root.parent) as old, tempfile.TemporaryDirectory(prefix='ptb-papers-',dir=e.root.parent) as active:
        old=Path(old);active=Path(active)
        for path,pid in ((old,2147483647),(active,os.getpid())):
            (path/'owner.json').write_text(json.dumps({'schema':'ptb-paper-scratch/1','instance':marker,'pid':pid}),encoding='utf8')
        (old/'synthetic.pdf').write_bytes(pdf())
        service.cleanup();assert not old.exists() and active.exists()
    cancel=threading.Event();raw=pdf();digest=hashlib.sha256(raw).hexdigest()
    def partial(root,operation,args,stop,progress):
        if operation=='downloadSelected':
            target=root/'cache/files';target.mkdir(parents=True);(target/(digest+'.pdf')).write_bytes(raw)
            cancel.set();raise StorageError('paper_cancelled',409)
        return {'ok':True,'value':{'papers':[]}}
    service.runner=partial
    assert service.call(e.owner,project,'downloadSelected',{'ids':[]},cancel,lambda _:None,lambda:None)['ok']
    snapshot=service.read(e.owner,service.current(e.owner,project))
    asset_id=snapshot['files']['cache/files/'+digest+'.pdf']
    assert e.storage.metadata(e.owner,asset_id)['sha256']==digest
    def late(*args):expire(e,lease);return {'ok':True,'value':{}}
    service.runner=late
    with pytest.raises(StorageError,match='workspace_expired'):
        service.call(e.owner,project,'interval',{'hours':2},threading.Event(),lambda _:None,lambda:e.workspaces.current(e.owner,key))
    assert not list(e.root.parent.glob('ptb-papers-*'))
