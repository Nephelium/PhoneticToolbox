"""M11 safety/behaviour regressions. Synthetic execution is explicitly separate."""
import hashlib
import json
from pathlib import Path
import zipfile
import pytest
import os
import platform
import sysconfig


def test_missing_transcript_is_not_asr(tmp_path):
    from ptb_worker.mfa.corpus import validate_corpus
    (tmp_path / '音频.wav').write_bytes(b'RIFF')
    with pytest.raises(ValueError, match='m11_missing_transcript'):
        validate_corpus(tmp_path)


def test_pairing_names_and_ambiguous_text(tmp_path):
    from ptb_worker.mfa.corpus import validate_corpus
    (tmp_path / '音频 1.wav').write_bytes(b'RIFF')
    (tmp_path / '音频 1.lab').write_text('a', encoding='utf8')
    pairs = validate_corpus(tmp_path)
    assert pairs[0]['audio'] == '音频 1.wav'
    (tmp_path / '音频 1.TextGrid').write_text('a', encoding='utf8')
    with pytest.raises(ValueError, match='m11_ambiguous_transcript'):
        validate_corpus(tmp_path)


@pytest.mark.parametrize('name', ['../outside', '/absolute', 'C:/escape', 'a\\escape', 'a/../../x', 'AUX.txt', 'x:stream', 'x./z'])
def test_component_rejects_unsafe_names(name):
    from ptb_worker.mfa.components import safe_name
    with pytest.raises(ValueError):
        safe_name(name)


def test_install_failure_keeps_current_and_partial(tmp_path):
    from ptb_worker.mfa.components import ComponentManager
    root = tmp_path / 'components'
    root.mkdir()
    (root / 'current.json').write_text('{"id":"old"}', encoding='utf8')
    archive = tmp_path / 'pack.zip'
    raw = b'not an executable'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('python.exe', raw)
    trusted = dict(schema='m11-component/1', id='candidate', platform='windows' if os.name=='nt' else 'linux', arch=platform.machine().lower().replace('amd64','x86_64') or 'x86_64',
                   mfa_version='3.3.8', source='local-reviewed', sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                   download_bytes=archive.stat().st_size, installed_bytes=len(raw), dependencies=[],
                   files=[dict(path='python.exe', bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())])
    mgr = ComponentManager(root)
    def fail(_):
        raise ValueError('m11_self_test_failed')
    with pytest.raises(ValueError, match='self_test'):
        mgr.import_archive(archive, trusted, fail)
    assert json.loads((root / 'current.json').read_text()) == {'id': 'old'}
    assert list((root / 'versions').iterdir())


def test_no_published_download_is_explicit(tmp_path):
    from ptb_worker.mfa.components import ComponentManager
    with pytest.raises(ValueError, match='m11_download_not_published'):
        ComponentManager(tmp_path).download({})


def package(tmp_path,name='python.exe',raw=b'fixture'):
    archive=tmp_path/'candidate.zip'
    with zipfile.ZipFile(archive,'w') as z:z.writestr(name,raw)
    trusted=dict(schema='m11-component/1',id='candidate',platform='windows' if os.name=='nt' else 'linux',
                 arch=platform.machine().lower().replace('amd64','x86_64') or 'x86_64',mfa_version='3.3.8',source='local test fixture',
                 dependencies=[],sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),download_bytes=archive.stat().st_size,
                 installed_bytes=len(raw),files=[dict(path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())])
    return archive,trusted


def test_install_switches_only_after_successful_probe(tmp_path):
    from ptb_worker.mfa.components import ComponentManager
    archive,trusted=package(tmp_path);root=tmp_path/'components';root.mkdir()
    current=root/'current.json';current.write_text('{"id":"old"}')
    def probe(path):
        assert json.loads(current.read_text())['id']=='old'
        assert (path/'python.exe').read_bytes()==b'fixture'
        return dict(success=True)
    target=ComponentManager(root).import_archive(archive,trusted,probe)
    assert Path(json.loads(current.read_text())['path'])==target


@pytest.mark.parametrize('case',['archive_hash','file_hash','platform','traversal'])
def test_invalid_component_never_activates(tmp_path,case):
    from ptb_worker.mfa.components import ComponentManager
    archive,trusted=package(tmp_path,'../escaped' if case=='traversal' else 'python.exe')
    if case=='archive_hash':trusted['sha256']='0'*64
    if case=='file_hash':trusted['files'][0]['sha256']='0'*64
    if case=='platform':trusted['platform']='invalid'
    root=tmp_path/'components';root.mkdir();current=root/'current.json';current.write_text('{"id":"old"}')
    with pytest.raises(ValueError):ComponentManager(root).import_archive(archive,trusted,lambda _:dict(success=True))
    assert json.loads(current.read_text())==dict(id='old')
    assert not (tmp_path/'escaped').exists()


def test_mfa_result_uses_current_policy_without_changing_legacy_reads():
    from ptb_worker.acoustic_files import AcousticFiles
    from ptb_api.m11_models import M11Manifest
    from uuid import uuid4
    files=[dict(id=str(uuid4()),name=name,kind='result',size_bytes=1,sha256='a'*64,expires_at=259300.) for name in ('x.TextGrid','m11-provenance.json')]
    publisher=object.__new__(AcousticFiles)
    manifest=publisher.manifest('mfa_alignment',files)
    assert M11Manifest.model_validate(manifest).policy_version==2
    assert publisher.independent_expiry(dict(operation='mfa_alignment'))


def test_no_linux_unrestricted_fallback(tmp_path,monkeypatch):
    import ptb_worker.mfa.runtime as runtime
    monkeypatch.setattr(runtime,'resolve_runtime',lambda _: (tmp_path,tmp_path/'python'))
    from types import SimpleNamespace
    monkeypatch.setattr(runtime,'os',SimpleNamespace(name='posix'))
    with pytest.raises(ValueError,match='m11_linux_runtime_unverified'):
        runtime.run(tmp_path,tmp_path/'attempt',{})


def test_log_redaction_and_bounded_tail(tmp_path):
    from ptb_worker.mfa.logs import redact,read_log
    text='File "C:\\Users\\private\\env\\mfa.py", line 12\ntoken=very-secret\nMFCC finished'
    value=redact(text)
    assert 'private' not in value and 'very-secret' not in value and 'MFCC finished' in value
    (tmp_path/'native.log').write_bytes(b'x'*300000+b'\nfinished')
    result=read_log(tmp_path)
    assert result['truncated'] and len(result['text'])<=262144 and result['text'].endswith('finished')


def test_main_bundle_excludes_optional_runtime_without_executing_build():
    import importlib.util
    root=Path(__file__).resolve().parents[2]
    spec=importlib.util.spec_from_file_location('m11_bundle',root/'scripts/m11_bundle.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    args=module.arguments(root)
    assert 'montreal_forced_aligner' in args and '_kalpy' in args
    assert sum(v.endswith(';resources/mfa') for v in args)==1
    assert all('auto_alignment' not in v and '.venv' not in v for v in args)


def test_runtime_content_identity_detects_native_drift_even_with_unchanged_metadata(tmp_path):
    from ptb_worker.mfa.runtime import fingerprint
    root=tmp_path/'env';(root/'conda-meta').mkdir(parents=True)
    executable=root/('python.exe' if os.name=='nt' else 'bin/python');executable.parent.mkdir(exist_ok=True);executable.write_bytes(b'fixture')
    (root/'conda-meta/montreal-forced-aligner-3.3.8.json').write_text('{"version":"3.3.8"}')
    native=root/'native.dat';native.write_bytes(b'old');stamp=native.stat()
    before=fingerprint(root);native.write_bytes(b'new');os.utime(native,ns=(stamp.st_atime_ns,stamp.st_mtime_ns))
    assert fingerprint(root)!=before
    with pytest.raises(ValueError,match='cancelled'):fingerprint(root,stop=lambda:True)


def test_m11_r1_new_defaults_do_not_rewrite_explicit_historical_values():
    from ptb_api.m11_models import M11Config
    assert M11Config().model_dump()==dict(beam=100,retry_beam=400)
    assert M11Config(beam=10,retry_beam=40).model_dump()==dict(beam=10,retry_beam=40)


def test_m11_r1_parallel_fingerprint_is_canonical_full_content_hash(tmp_path):
    from ptb_worker.mfa.runtime import fingerprint
    root=tmp_path/'env';(root/'conda-meta').mkdir(parents=True)
    executable=root/('python.exe' if os.name=='nt' else 'bin/python');executable.parent.mkdir(exist_ok=True);executable.write_bytes(b'fixture')
    (root/'conda-meta/montreal-forced-aligner-3.3.8.json').write_text('{"version":"3.3.8"}')
    for i in range(130):(root/f'{i:03d}.dat').write_bytes(bytes([i])*20)
    reference=hashlib.sha256()
    for path in sorted(p for p in root.rglob('*') if p.is_file()):
        reference.update(path.relative_to(root).as_posix().encode())
        reference.update(hashlib.sha256(path.read_bytes()).digest())
    assert fingerprint(root)==reference.hexdigest()


@pytest.mark.skipif(os.name!='nt',reason='Windows Job Object orchestration')
def test_m11_r1_kernel_cache_is_host_owned_and_isolated_by_content(tmp_path,monkeypatch):
    import ptb_worker.mfa.runtime as runtime
    from ptb_worker.native import windows
    monkeypatch.setenv('PTB_M11_COMPONENT_ROOT',str(tmp_path/'components'))
    monkeypatch.setattr(runtime,'resolve_runtime',lambda _: (tmp_path,tmp_path/'python.exe'))
    class FixtureProcess:
        pid=1
        group_cleaned=False
        def __init__(self,argv,root,memory):
            (root/'response.json').write_text('{"success":true}',encoding='utf8')
        def poll(self):return 0
        def memory_peak(self):return 0
        def close(self):self.group_cleaned=True
    monkeypatch.setattr(windows,'OwnedProcess',FixtureProcess)
    paths=[]
    for i,key in enumerate(('a'*64,'b'*64,'a'*64)):
        attempt=tmp_path/f'attempt{i}'
        result=runtime.run(tmp_path,attempt,dict(kernel_cache_key=key,kernel_cache='untrusted-input-path'))
        assert result['success']
        request=json.loads((attempt/'request.json').read_text('utf8'))
        paths.append(request['kernel_cache'])
        assert Path(paths[-1])==tmp_path/'components/kernel-cache'/key
    assert paths[0]==paths[2] and paths[0]!=paths[1]
    with pytest.raises(ValueError,match='m11_runtime_changed'):
        runtime.run(tmp_path,tmp_path/'invalid',dict(kernel_cache_key='../escape'))


@pytest.mark.skipif(os.name!='nt',reason='Windows Job Object orchestration')
def test_m11_r1_shared_kernel_cache_remains_within_monitored_budget(tmp_path,monkeypatch):
    import ptb_worker.mfa.runtime as runtime
    from ptb_worker.native import windows
    monkeypatch.setenv('PTB_M11_COMPONENT_ROOT',str(tmp_path/'components'))
    monkeypatch.setattr(runtime,'resolve_runtime',lambda _: (tmp_path,tmp_path/'python.exe'))
    cache=tmp_path/'components/kernel-cache'/('a'*64);cache.mkdir(parents=True)
    (cache/'kernel.nbc').write_bytes(b'k'*2000)
    class FixtureProcess:
        pid=1
        group_cleaned=False
        def __init__(self,*args):pass
        def poll(self):return None
        def memory_peak(self):return 0
        def close(self):self.group_cleaned=True
    monkeypatch.setattr(windows,'OwnedProcess',FixtureProcess)
    evidence={}
    with pytest.raises(ValueError,match='m11_temp_budget'):
        runtime.run(tmp_path,tmp_path/'attempt',dict(kernel_cache_key='a'*64),disk=1500,evidence=evidence)
    assert evidence['group_cleaned'] and evidence['temp_peak_sampled_bytes']>=2000
