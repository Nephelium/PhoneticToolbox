"""Disposable SQLite ownership receipts and actual Windows path/link boundaries."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from uuid import uuid4
import pytest

from test_local_retention import storage,add,NOW
from ptb_worker.store import canonical,JobError
from ptb_worker.local_retention import DAY,LocalRetention
from ptb_worker.local_acoustic_files import LocalAcousticFiles,initialize_local_files
from ptb_worker.mfa_retention import MARKER


@pytest.fixture
def registry(tmp_path,monkeypatch):
    root=tmp_path/'mfa-components';root.mkdir()
    monkeypatch.setenv('PTB_M11_COMPONENT_ROOT',str(root))
    for name in ('registered-model.zip','registered-dictionary.dict','kernel-cache/keep.bin'):
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'protected-'+name.encode())
    return root


def job(storage,*,state='running',snapshot=None):
    store,files,manager=storage
    key,_=add(storage,state='running',operation='mfa_alignment',age=0)
    snapshot=snapshot or dict(operation='mfa_alignment',execution_route='desktop-local',input_assets=[])
    with files.batch_transaction() as tx:
        tx.execute('UPDATE {jobs} SET state=?,generation=1,worker_id=?,lease_until=?,snapshot=? WHERE id=?',
                   (state,'mfa-owner',NOW+1000,canonical(snapshot),key))
    return key,'mfa-owner',1


def terminal(storage,identity,state='failed'):
    with storage[1].batch_transaction() as tx:tx.execute('UPDATE {jobs} SET state=? WHERE id=?',(state,identity[0]))


def attempt(storage,registry,*,finished=True,start=NOW,finish=NOW,identity=None):
    identity=identity or job(storage)
    path=registry/'attempts'/f'{identity[0]}-{identity[2]}-{uuid4().hex}';path.mkdir(parents=True)
    storage[2].register_mfa_attempt(path,identity,now=start)
    (path/'corpus').mkdir();(path/'corpus/input.wav').write_bytes(b'owned audio duplicate')
    (path/'output').mkdir();(path/'output/result.TextGrid').write_bytes(b'owned diagnostic output')
    if finished:storage[2].complete_mfa_attempt(path,identity,now=finish)
    terminal(storage,identity)
    return path,identity


def policy(storage):return json.loads((storage[1].root/'.ptb-retention.json').read_text('utf-8'))


def test_seven_days_after_completion_with_full_grace_and_sentinels(storage,registry):
    path,_=attempt(storage,registry,start=NOW-20*DAY,finish=NOW)
    before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in registry.rglob('*') if p.is_file() and 'attempts' not in p.parts}
    result=storage[2].sweep(now=NOW+7*DAY-1,force=True)
    assert result['diagnostic_count']==0 and path.exists()
    result=storage[2].sweep(now=NOW+7*DAY,force=True)
    assert result['diagnostic_count']==1 and result['diagnostic_bytes']>0 and result['complete'] and not path.exists()
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==sha for p,sha in before.items())


def test_unregistered_historical_attempts_ignored_even_with_old_mtime(storage,registry):
    path=registry/'attempts'/f'{uuid4()}-1-{uuid4().hex}';path.mkdir(parents=True);(path/'old.wav').write_bytes(b'unknown')
    os.utime(path,(1,1));os.utime(path/'old.wav',(1,1))
    result=storage[2].sweep(now=NOW+100*DAY,force=True)
    assert result['diagnostic_count']==0 and (path/'old.wav').read_bytes()==b'unknown'
    assert not policy(storage).get('mfa_attempts')


@pytest.mark.parametrize('state',['queued','running','cancel_requested'])
def test_any_active_generation_protects_completed_old_attempt(storage,registry,state):
    path,identity=attempt(storage,registry)
    with storage[1].batch_transaction() as tx:tx.execute('UPDATE {jobs} SET state=?,generation=9 WHERE id=?',(state,identity[0]))
    result=storage[2].sweep(now=NOW+8*DAY,force=True)
    assert result['diagnostic_count']==0 and result['diagnostic_protected_count']==1 and path.exists()


def test_incomplete_attempt_stays_protected_after_terminal_job(storage,registry):
    path,_=attempt(storage,registry,finished=False)
    result=storage[2].sweep(now=NOW+100*DAY,force=True)
    assert result['diagnostic_protected_count']==1 and path.exists()
    assert policy(storage)['mfa_attempts'][path.name]['completed_at'] is None


def test_enabled_controls_diagnostics_without_resetting_completion(storage,registry):
    path,_=attempt(storage,registry)
    storage[2].configure(enabled=False,days=30)
    assert storage[2].sweep(now=NOW+8*DAY,force=True)['skipped'] and path.exists()
    storage[2].configure(enabled=True,days=30)
    assert storage[2].sweep(now=NOW+8*DAY,force=True)['diagnostic_count']==1


def test_other_instance_same_registry_remains_unowned(storage,registry,tmp_path):
    foreign=tmp_path/'foreign-cache';foreign.mkdir();initialize_local_files(foreign)
    other_files=LocalAcousticFiles(storage[0],foreign)
    other=(storage[0],other_files,LocalRetention(other_files))
    path,_=attempt(other,registry)
    assert storage[2].sweep(now=NOW+8*DAY,force=True)['diagnostic_count']==0 and path.exists()
    assert other[2].sweep(now=NOW+8*DAY,force=True)['diagnostic_count']==1


def test_hardlink_is_rejected_before_any_prefix_deletion(storage,registry,tmp_path):
    path,_=attempt(storage,registry);original=tmp_path/'protected-user.wav'
    os.link(path/'corpus/input.wav',original)
    result=storage[2].sweep(now=NOW+8*DAY,force=True)
    assert not result['complete'] and result['diagnostic_failed_count']==1
    assert original.read_bytes()==b'owned audio duplicate' and (path/'output/result.TextGrid').exists() and (path/MARKER).exists()


def test_junction_target_not_touched_and_retry_after_link_removed(storage,registry,tmp_path):
    path,_=attempt(storage,registry);outside=tmp_path/'user-originals';outside.mkdir();(outside/'input.wav').write_bytes(b'original')
    link=path/'external'
    try:link.symlink_to(outside,target_is_directory=True)
    except OSError:
        if sys.platform!='win32':pytest.skip('symlink unavailable')
        made=subprocess.run(['cmd','/c','mklink','/J',str(link),str(outside)],capture_output=True)
        if made.returncode:pytest.skip('junction unavailable')
    result=storage[2].sweep(now=NOW+8*DAY,force=True)
    assert result['diagnostic_failed_count']==1 and not result['complete'] and (outside/'input.wav').read_bytes()==b'original'
    assert (path/'corpus/input.wav').exists()
    if link.is_symlink():link.unlink()
    else:link.rmdir()
    assert storage[2].sweep(now=NOW+8*DAY,force=True)['diagnostic_count']==1
    assert (outside/'input.wav').read_bytes()==b'original'


def test_partial_delete_retains_marker_registration_and_can_retry(storage,registry,monkeypatch):
    path,_=attempt(storage,registry);blocked=path/'output/result.TextGrid';original=Path.unlink
    def deny(item,*args,**kwargs):
        if item==blocked:raise PermissionError('owned fixture denial')
        return original(item,*args,**kwargs)
    monkeypatch.setattr(Path,'unlink',deny)
    result=storage[2].sweep(now=NOW+8*DAY,force=True)
    assert result['diagnostic_failed_count']==1 and not result['complete'] and blocked.exists() and (path/MARKER).exists()
    assert policy(storage)['mfa_attempts'][path.name]['removed'] is False
    monkeypatch.setattr(Path,'unlink',original)
    assert storage[2].sweep(now=NOW+8*DAY,force=True)['diagnostic_count']==1


def test_replaced_root_marker_or_registry_fails_closed(storage,registry,monkeypatch):
    path,_=attempt(storage,registry)
    marker=path/MARKER;value=json.loads(marker.read_text('utf-8'));value['instance_id']=str(uuid4());marker.write_text(json.dumps(value),'utf-8')
    assert storage[2].sweep(now=NOW+8*DAY,force=True)['diagnostic_failed_count']==1 and path.exists()
    monkeypatch.setenv('PTB_M11_COMPONENT_ROOT',str(registry/'new-root'))
    assert storage[2].sweep(now=NOW+8*DAY,force=True)['diagnostic_failed_count']==1 and path.exists()


def test_path_escape_and_non_mfa_job_never_get_registered(storage,registry):
    identity=job(storage)
    outside=registry/f'{identity[0]}-1-{uuid4().hex}';outside.mkdir()
    with pytest.raises(JobError):storage[2].register_mfa_attempt(outside,identity,now=NOW)
    path=registry/'attempts'/f'{identity[0]}-1-{uuid4().hex}';path.mkdir(parents=True)
    with storage[1].batch_transaction() as tx:tx.execute('UPDATE {jobs} SET snapshot=? WHERE id=?',(canonical({'operation':'acoustic_analysis','execution_route':'desktop-local','input_assets':[]}),identity[0]))
    with pytest.raises(JobError):storage[2].register_mfa_attempt(path,identity,now=NOW)
    assert not (outside/MARKER).exists() and not (path/MARKER).exists()


def test_completed_receipt_does_not_restart_grace(storage,registry):
    path,identity=attempt(storage,registry)
    storage[2].complete_mfa_attempt(path,identity,now=NOW+6*DAY)
    assert storage[2].sweep(now=NOW+7*DAY,force=True)['diagnostic_count']==1


@pytest.mark.parametrize('cleanup',[True,False,None])
def test_executor_registers_failure_but_only_completes_known_cleaned_attempt(storage,registry,monkeypatch,cleanup):
    import ptb_worker.m11_executor as executor
    model=registry/'registered-model.zip';dictionary=registry/'registered-dictionary.dict'
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    snapshot=dict(operation='mfa_alignment',execution_route='desktop-local',request=dict(runtime_id='runtime',model_id='model',corpus=[],config={}),
                  runtime_fingerprint='f'*64,model_sha256=sha(model),dictionary_sha256=sha(dictionary),input_assets=[])
    identity=job(storage,snapshot=snapshot)
    monkeypatch.setattr(executor,'select',lambda *args,**kwargs:({'path':str(registry),'fingerprint':'f'*64},
                          {'model':str(model),'dictionary':str(dictionary),'model_sha256':sha(model),'dictionary_sha256':sha(dictionary)}))
    def failed_run(*args,**kwargs):
        if cleanup is not None:kwargs['evidence'].update(main_pid=424242,group_cleaned=cleanup)
        raise ValueError('m11_timeout')
    monkeypatch.setattr(executor,'run',failed_run)
    with storage[0].transaction() as tx:claim=dict(storage[0]._row(tx,identity[0]))
    executor.execute_claim(storage[0],claim,identity[1],threading.Event())
    records=policy(storage)['mfa_attempts'];assert len(records)==1
    record=next(iter(records.values()))
    assert (record['completed_at'] is not None)==(cleanup is True)
    result=storage[2].sweep(now=(record['completed_at'] or NOW)+7*DAY,force=True)
    assert result['diagnostic_count']==int(cleanup is True)
    if cleanup is not True:assert result['diagnostic_protected_count']==1
    assert model.exists() and dictionary.exists()


def test_final_rmdir_failure_restores_exact_marker_and_retries(storage,registry,monkeypatch):
    path,_=attempt(storage,registry);receipt=(path/MARKER).read_bytes();original=Path.rmdir
    def deny(item,*args,**kwargs):
        if item==path:raise PermissionError('owned root denial')
        return original(item,*args,**kwargs)
    monkeypatch.setattr(Path,'rmdir',deny)
    result=storage[2].sweep(now=NOW+8*DAY,force=True)
    assert not result['complete'] and result['failed_count']==1 and result['diagnostic_failed_count']==1
    assert (path/MARKER).read_bytes()==receipt and not policy(storage)['mfa_attempts'][path.name]['removed']
    monkeypatch.setattr(Path,'rmdir',original)
    assert storage[2].sweep(now=NOW+8*DAY,force=True)['diagnostic_count']==1


def test_directory_replacement_with_copied_marker_is_rejected(storage,registry):
    path,_=attempt(storage,registry);former=path.with_name(path.name+'-original');path.rename(former)
    path.mkdir();(path/MARKER).write_bytes((former/MARKER).read_bytes());(path/'new.wav').write_bytes(b'foreign replacement')
    result=storage[2].sweep(now=NOW+8*DAY,force=True)
    assert not result['complete'] and result['diagnostic_failed_count']==1
    assert (path/'new.wav').read_bytes()==b'foreign replacement' and (former/'corpus/input.wav').exists()
