"""Host configuration failures and dispatch preservation; no fake Linux success."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from ptb_worker.native import linux_runtime, capabilities
from ptb_worker.io.limits import FormatError, Limits


def test_unknown_entry_rejected_without_configuration():
    with pytest.raises(ValueError,match='Unsupported fixed'):
        linux_runtime.command('os','request')


def test_missing_profile_rejected(monkeypatch):
    monkeypatch.delenv('PTB_LINUX_RUNTIME_PROFILE',raising=False)
    with pytest.raises(FormatError,match='linux_runtime_unavailable'):
        linux_runtime.load_profile()


@pytest.mark.parametrize('value',['[]','{}','{"schema":"p11-runtime/1"}','invalid json'])
def test_malformed_profile_is_bounded_task_error(tmp_path,value):
    path=tmp_path/'bad.json';path.write_text(value)
    with pytest.raises(FormatError):linux_runtime.load_profile(path)


def profile(tmp_path):
    binary=tmp_path/'python';binary.write_bytes(b'fixture interpreter, never executed')
    bootstrap=Path(linux_runtime.__file__).parents[1]/'linux_bootstrap.py'
    value=dict(schema='p11-runtime/1',python=str(binary),cache=str(tmp_path),sys_paths=[str(tmp_path)],
               hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [binary,bootstrap]})
    path=tmp_path/'runtime.json';path.write_text(json.dumps(value),encoding='utf-8')
    return path,binary


def test_changed_binary_invalidates_hash_profile(tmp_path):
    path,binary=profile(tmp_path)
    linux_runtime.load_profile(path)
    binary.write_bytes(b'changed')
    with pytest.raises(FormatError,match='linux_runtime_mismatch'):linux_runtime.load_profile(path)


def test_profile_without_current_bootstrap_rejected(tmp_path):
    path,_=profile(tmp_path);value=json.loads(path.read_text())
    value['hashes']={value['python']:value['hashes'][value['python']]}
    path.write_text(json.dumps(value))
    with pytest.raises(FormatError,match='linux_runtime_mismatch'):linux_runtime.load_profile(path)


def test_profile_alone_cannot_advertise_capability(tmp_path,monkeypatch):
    path,_=profile(tmp_path)
    monkeypatch.setenv('PTB_LINUX_RUNTIME_PROFILE',str(path))
    monkeypatch.delenv('PTB_LINUX_VALIDATION_RECEIPT',raising=False)
    assert capabilities.linux_capabilities(SimpleNamespace(files=object(),batches=object()))[0]==[]


@pytest.mark.parametrize('profile_name,expected', [('server-small',1073741824),('desktop-local',3_000_000_000)])
def test_linux_dispatch_clamps_group_budget_and_preserves_hooks(monkeypatch,tmp_path,profile_name,expected):
    from ptb_worker.native import posix
    from ptb_worker import resource_profiles
    monkeypatch.setattr(resource_profiles,'selected_profile',lambda:resource_profiles.PROFILES[profile_name])
    seen={};hook=lambda pid:None;stop=lambda:False;evidence={}
    monkeypatch.setattr(linux_runtime,'command',lambda entry,request:['trusted',entry,str(request)])
    def capture(argv,payload,cwd,limits,**kwargs):
        seen.update(argv=argv,payload=payload,cwd=cwd,limits=limits,**kwargs)
        return b'bounded bytes'
    monkeypatch.setattr(posix,'run_bounded',capture)
    assert linux_runtime.run('lpc',tmp_path/'request',tmp_path,Limits(process_bytes=3_000_000_000),
                             on_started=hook,stop=stop,evidence=evidence)==b'bounded bytes'
    assert seen['limits'].process_bytes==expected
    assert seen['on_started'] is hook and seen['stop'] is stop and seen['evidence'] is evidence
