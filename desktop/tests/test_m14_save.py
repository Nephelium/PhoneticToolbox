"""Owned-directory failure injection; production save transaction unchanged."""
import base64,hashlib,json,os
from types import SimpleNamespace
from pathlib import Path
from uuid import uuid4
from unittest.mock import patch
import pytest
from ptb_desktop.file_provider import FileProvider,FileAccessError
from ptb_desktop.m14_bridge import M14Bridge
from phonetic_core.transcription.phonology.models import NAMES


def test_native_save_failure_rolls_back_only_new_outputs(tmp_path):
    provider=FileProvider();ref=provider.choose('output',lambda:str(tmp_path))
    sentinel=tmp_path/'unrelated.txt';sentinel.write_text('keep',encoding='utf8')
    files=[dict(id=str(i),name=n) for i,n in enumerate(NAMES)];payload=b'public synthetic save-failure probe'
    receipts=[]
    bridge=SimpleNamespace(provider=provider,service=SimpleNamespace(get=lambda _:dict(state='succeeded',operation='phonology_induction',result_manifest=dict(files=files))),invoke=lambda _:dict(base64=base64.b64encode(payload).decode()),record_export=lambda ids:receipts.append(ids))
    save=M14Bridge(bridge);real=os.rename;calls=[]
    def fail_second(source,target):
        calls.append(1)
        if len(calls)==2:raise PermissionError('controlled write failure')
        return real(source,target)
    with patch('ptb_desktop.m14_bridge.os.rename',side_effect=fail_second):
        with pytest.raises(OSError):save.save(str(uuid4()),ref['id'])
    assert list(tmp_path.iterdir())==[sentinel] and sentinel.read_text()=='keep'
    assert not receipts
    assert save.save(str(uuid4()),ref['id'])['count']==3
    assert receipts==[[item['id'] for item in files]]
    before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.iterdir()}
    with pytest.raises(FileAccessError,match='同名'):save.save(str(uuid4()),ref['id'])
    assert before=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.iterdir()}
