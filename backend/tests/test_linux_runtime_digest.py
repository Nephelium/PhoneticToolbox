import concurrent.futures
import hashlib
import os
import json
from pathlib import Path
import pytest
from ptb_worker.io.limits import FormatError

from ptb_worker.native import linux_runtime as runtime


def test_concurrent_verification_reads_unchanged_file_once(tmp_path, monkeypatch):
    path=tmp_path/'library.so';path.write_bytes(b'trusted runtime'*1000)
    original=Path.open;opens=[]
    def opened(self,*args,**kwargs):
        if self==path:opens.append(args)
        return original(self,*args,**kwargs)
    monkeypatch.setattr(Path,'open',opened)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        actual=list(pool.map(lambda _:runtime._verified_digest(path),range(24)))
    assert actual==[hashlib.sha256(b'trusted runtime'*1000).hexdigest()]*24
    assert len(opens)==1


def test_file_edit_and_atomic_replacement_invalidate_digest(tmp_path):
    path=tmp_path/'library.so';path.write_bytes(b'original')
    original=runtime._verified_digest(path)
    path.write_bytes(b'changed content')
    assert runtime._verified_digest(path)!=original
    prior=path.stat();replacement=tmp_path/'replacement'
    replacement.write_bytes(b'altered content');os.utime(replacement,ns=(prior.st_atime_ns,prior.st_mtime_ns))
    replacement.replace(path)
    assert runtime._verified_digest(path)==hashlib.sha256(b'altered content').hexdigest()


def test_profile_expected_hash_is_checked_after_cache_hit(tmp_path):
    path=tmp_path/'library.so';path.write_bytes(b'trusted')
    bootstrap=Path(runtime.__file__).parents[1]/'linux_bootstrap.py'
    profile=dict(schema='p11-runtime/1',python=str(path),cache=str(tmp_path),sys_paths=[],hashes={str(path):hashlib.sha256(b'trusted').hexdigest(),str(bootstrap):hashlib.sha256(bootstrap.read_bytes()).hexdigest()})
    manifest=tmp_path/'runtime.json';manifest.write_text(json.dumps(profile))
    runtime.load_profile(manifest)
    profile['hashes'][str(path)]=hashlib.sha256(b'untrusted').hexdigest();manifest.write_text(json.dumps(profile))
    with pytest.raises(FormatError,match='linux_runtime_mismatch'):runtime.load_profile(manifest)
