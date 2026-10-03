"""M01-R3 user exports: real pairs, readable collisions, internal JSON retained."""
import hashlib
import io
import json
import sqlite3
from urllib.parse import urlsplit,parse_qs
from uuid import uuid4
import pytest
from openpyxl import load_workbook
from ptb_desktop.file_provider import FileProvider
from ptb_desktop.task_bridge import TaskBridge
from ptb_worker.io.parameter_exports import _build_pair
from ptb_worker.io.limits import Limits


def fixture(tmp_path,operation='acoustic_analysis'):
    table={'columns':['Time_s','F0 - REAPER','word'], 'kinds':['number','number','text'],
           'rows':[[0.,123.456789012345,'井井'],[.005,None,'=literal'],[.01,'+Infinity',None]]}
    pair=_build_pair(table,Limits())
    base='result' if operation=='acoustic_analysis' else '声音_word_井井_0.000_0.020_0001'
    payloads={base+'.xlsx':pair.xlsx,base+'.ptb.sqlite':pair.sqlite,
              ('result.ptb.json' if operation=='acoustic_analysis' else 'segments.ptb.json'):json.dumps({'computation_revision':'acoustic/2','inputs':['internal provenance']}).encode()}
    if operation=='textgrid_segment':payloads[base+'.wav']=b'segment wav bytes'
    files=[dict(id=str(uuid4()),name=name,size_bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()) for name,raw in payloads.items()]
    job=dict(id=str(uuid4()),state='succeeded',operation=operation,result_manifest={'files':files})
    batch=dict(id=str(uuid4()),audio_names=['声音.wav'],summary={'items':[dict(index=0,state='succeeded',job_id=job['id'])]})
    class Service:
        downloads=[]
        fail=None
        def get(self,path):return batch if '/batches/' in path else job
        def binary(self,path):
            url=urlsplit(path);query=parse_qs(url.query);file=next(f for f in files if f['id']==url.path.rsplit('/',1)[-1]);name=file['name']
            self.downloads.append(name)
            if name==self.fail:raise RuntimeError('controlled download failure')
            offset=int(query['offset'][0]);return payloads[name][offset:offset+int(query['size'][0])]
    provider=FileProvider();grant=provider.choose('output',lambda:tmp_path)
    service=Service();service.downloads=[]
    return TaskBridge(provider,service),grant['id'],batch,job,table,payloads,base


def test_user_saves_two_real_formats_and_preserves_internal_json(tmp_path):
    bridge,grant,batch,job,table,payloads,_=fixture(tmp_path)
    result=bridge.save(batch['id'],grant)
    assert set(result['saved'])=={'声音.xlsx','声音.ptb.sqlite'} and result['count']==2
    assert not list(tmp_path.glob('*.json'))
    assert 'result.ptb.json' not in bridge.service.downloads
    assert any(f['name']=='result.ptb.json' for f in job['result_manifest']['files'])
    assert json.loads(payloads['result.ptb.json'])['computation_revision']=='acoustic/2'
    wb=load_workbook(tmp_path/'声音.xlsx',read_only=True,data_only=False)
    try:
        rows=list(wb.active.values);assert list(rows[0])==table['columns']
        assert rows[1][0]==0. and rows[1][1]==pytest.approx(123.456789012345,abs=1e-12) and rows[1][2]=='井井'
        assert rows[2]==(.005,None,'=literal') and rows[3]==(.01,'inf',None)
    finally:wb.close()
    conn=sqlite3.connect(':memory:')
    try:
        conn.deserialize((tmp_path/'声音.ptb.sqlite').read_bytes())
        assert conn.execute('PRAGMA integrity_check').fetchone()==('ok',)
        rows=conn.execute('SELECT * FROM params ORDER BY rowid').fetchall()
        assert rows[0]==(0.,123.456789012345,'井井') and rows[1]==(.005,None,'=literal') and rows[2]==(.01,float('inf'),None)
    finally:conn.close()
    assert bridge.save(batch['id'],grant)==result
    assert len(list(tmp_path.iterdir()))==2


def test_conflicting_pairs_use_one_suffix_and_repeated_save_reuses_pair(tmp_path):
    bridge,grant,batch,*_=fixture(tmp_path)
    (tmp_path/'声音.xlsx').write_bytes(b'previous result')
    (tmp_path/'声音 (2).ptb.sqlite').write_bytes(b'another previous result')
    result=bridge.save(batch['id'],grant)
    assert set(result['saved'])=={'声音 (3).xlsx','声音 (3).ptb.sqlite'}
    assert (tmp_path/'声音.xlsx').read_bytes()==b'previous result'
    assert (tmp_path/'声音 (2).ptb.sqlite').read_bytes()==b'another previous result'
    assert bridge.save(batch['id'],grant)==result
    assert len(list(tmp_path.iterdir()))==4


def test_segment_wav_pair_share_suffix_without_saving_manifest(tmp_path):
    bridge,grant,batch,job,table,payloads,base=fixture(tmp_path,'textgrid_segment')
    (tmp_path/(base+'.wav')).write_bytes(b'previous segment')
    result=bridge.save(batch['id'],grant)
    assert set(result['saved'])=={base+' (2).wav',base+' (2).xlsx',base+' (2).ptb.sqlite'}
    assert (tmp_path/(base+'.wav')).read_bytes()==b'previous segment'
    assert not list(tmp_path.glob('*.json'))
    assert bridge.save(batch['id'],grant)==result
    assert any(f['name']=='segments.ptb.json' for f in job['result_manifest']['files'])


def test_failed_pair_download_rolls_back_only_new_outputs(tmp_path):
    bridge,grant,batch,*_=fixture(tmp_path)
    (tmp_path/'声音.xlsx').write_bytes(b'previous result')
    bridge.service.fail='result.ptb.sqlite'
    with pytest.raises(RuntimeError,match='controlled download failure'):bridge.save(batch['id'],grant)
    assert [p.name for p in tmp_path.iterdir()]==['声音.xlsx']
    assert (tmp_path/'声音.xlsx').read_bytes()==b'previous result'
