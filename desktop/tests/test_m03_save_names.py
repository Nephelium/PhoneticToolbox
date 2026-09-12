"""Actual native writes for M03 save suggestions; server lifecycle tested by UI QA."""
import hashlib
import json
from urllib.parse import urlsplit,parse_qs
from uuid import uuid4
import pytest
from ptb_desktop.file_provider import FileProvider,FileAccessError
from ptb_desktop.task_bridge import TaskBridge


def bridge_for(folder,names):
    data={'egg.ptb.json':json.dumps({'export_names':names}).encode(),'egg_DATA.csv':b'Time (s),CQ\n0.1,0.5\n'}
    records=[dict(id=str(uuid4()),name=name,size_bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()) for name,raw in data.items()]
    blobs={record['id']:data[record['name']] for record in records}
    job=dict(id=str(uuid4()),state='succeeded',operation='egg_analysis',result_manifest=dict(files=records))
    class Service:
        def get(self,path):return job
        def binary(self,path):
            query=parse_qs(urlsplit(path).query);raw=blobs[urlsplit(path).path.rsplit('/',1)[1]]
            offset=int(query['offset'][0]);return raw[offset:offset+int(query['size'][0])]
    provider=FileProvider();grant=provider.choose('output',lambda:folder)
    return provider,TaskBridge(provider,Service()),job['id'],grant['id']


def test_named_save_conflict_repeat_and_old_metadata_fallback(tmp_path):
    for mapping in ({},{'egg_DATA.csv':'ɑ̃˥_0_10s_0_50s_DATA.csv','egg.ptb.json':'ɑ̃˥_0_10s_0_50s.ptb.json'}):
        folder=tmp_path/str(uuid4());folder.mkdir();target=folder/mapping.get('egg_DATA.csv','egg_DATA.csv');target.write_bytes(b'existing result')
        provider,bridge,job,grant=bridge_for(folder,mapping)
        try:
            result=bridge.save(job,grant,single=True)
            assert result['count']==2 and target.read_bytes()==b'existing result'
            assert (folder/(job[:8]+'-'+target.name)).read_bytes()==b'Time (s),CQ\n0.1,0.5\n'
            assert bridge.save(job,grant,single=True)['saved']==result['saved']
            assert len(list(folder.iterdir()))==3
        finally:provider.close()


def test_unsafe_suggestion_rolls_back_only_new_files(tmp_path):
    existing=tmp_path/'keep.csv';existing.write_bytes(b'keep')
    provider,bridge,job,grant=bridge_for(tmp_path,{'egg_DATA.csv':'../escaped.csv'})
    try:
        with pytest.raises(FileAccessError):bridge.save(job,grant,single=True)
        assert list(tmp_path.iterdir())==[existing] and existing.read_bytes()==b'keep'
    finally:provider.close()
