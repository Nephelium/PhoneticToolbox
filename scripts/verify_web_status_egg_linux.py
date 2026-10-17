"""Owned host-cache safety, one-second aggregate HTTP and 77-second EGG regression."""
from pathlib import Path
import hashlib,io,json,os,subprocess,sys,tempfile,time
from types import SimpleNamespace


def main():
    out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
    from ptb_worker.native.linux_runtime import load_profile,_verified_digest
    from ptb_worker.native.digest_cache import shared_digests
    from ptb_worker.io.limits import FormatError
    profile_path,profile=load_profile();records=[]
    with tempfile.TemporaryDirectory(prefix='digest-safety-',dir=out) as directory:
        root=Path(directory);root.chmod(0o700);library=root/'synthetic.so';library.write_bytes(b'original');expected=hashlib.sha256(b'original').hexdigest()
        first={}
        with shared_digests(first,root=root):first[str(library)]=((1,2,3,4,5,6),expected)
        second={}
        with shared_digests(second,root=root):assert second==first
        child="from pathlib import Path;from ptb_worker.native.digest_cache import shared_digests;import sys;m={};ctx=shared_digests(m,root=Path(sys.argv[1]));ctx.__enter__();assert m[str(Path(sys.argv[1])/'synthetic.so')][1]==sys.argv[2];ctx.__exit__(None,None,None)"
        subprocess.run([sys.executable,'-B','-c',child,str(root),expected],check=True,timeout=10)
        _verified_digest(library,expected);info=library.stat();time.sleep(.01);library.write_bytes(b'modified');os.utime(library,ns=(info.st_atime_ns,info.st_mtime_ns));assert _verified_digest(library,expected)!=expected
        bad=root/'ptb-runtime-digests-v1/digests.json';bad.chmod(0o644)
        try:
            with shared_digests({},root=root):raise AssertionError('Unsafe cache accepted')
        except FormatError as exc:assert str(exc)=='linux_runtime_cache_unsafe'
        bad.chmod(0o600)
    from ptb_worker.egg_memory_cache import save,restore,source_bytes
    from ptb_worker.egg_interactive_child import Session
    from ptb_api.egg_models import EggTaskConfig
    import numpy as np
    from scipy.io import wavfile
    short=io.BytesIO();axis=np.arange(16000)/16000;wavfile.write(short,16000,np.column_stack([np.sin(2*np.pi*120*axis),np.sin(2*np.pi*120*axis+.3)]).astype('float32'));short=short.getvalue()
    from ptb_worker.native.anonymous_memory import create
    state_fd=create('ptb-egg-state-test');os.fchmod(state_fd,0o600);state_path=f'/proc/{os.getpid()}/fd/{state_fd}'
    try:
        config=EggTaskConfig(mode='preview',roi_start=.1,roi_end=.4,keep_praat_f0=False,keep_gci_f0=False).model_dump();fresh=Session(short);fresh.update(config);assert save(fresh,state_path)
        restored=Session(short);assert restore(restored,state_path)
        other={**config,'micro_width_ms':40};expected=Session(short).update(other)
        import phonetic_core.egg as egg_core
        original=egg_core.prepare
        def forbidden(*args,**kwargs):raise AssertionError('Display update repeated full preprocessing')
        egg_core.prepare=forbidden
        try:actual=restored.update(other)
        finally:egg_core.prepare=original
        assert json.loads(json.dumps(actual))==json.loads(json.dumps(expected))
        # A damaged cache must fall back to raw-source analysis, never partial state.
        os.ftruncate(state_fd,20);assert not restore(restored,state_path) and restored.result is None
    finally:os.close(state_fd)
    # No user identity or project is used. SQL reads only current aggregate counts.
    import psycopg
    from psycopg.rows import dict_row
    from contextlib import contextmanager
    from ptb_api.web_status import PublicLoad,host_counters
    from ptb_api.workspaces import create_workspace_router
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    settings=json.loads(Path('/var/lib/ptb-web/private.json').read_text());reads=[];probes=[]
    @contextmanager
    def read():
        reads.append(time.monotonic())
        with psycopg.connect(settings['dsn'],autocommit=True,row_factory=dict_row,options='-c default_transaction_read_only=on') as c:yield c
    service=SimpleNamespace(_read=read,now=lambda c:c.execute('SELECT extract(epoch FROM clock_timestamp())::float8 AS t').fetchone()['t'])
    def probe():probes.append(time.monotonic());return host_counters()
    load=PublicLoad(service,SimpleNamespace(max_running=1),probe=probe)
    for i in range(10):load.sample(details=False)
    assert len(reads)==1 and not probes
    time.sleep(1.05);load.sample(details=False);assert len(reads)==2 and not probes
    load.sample();time.sleep(1.05);assert load.sample().cpu_percent is not None and len(probes)==2
    ctx=SimpleNamespace(available=lambda:None,session=lambda:{'id':'isolated-synthetic'},mutation=lambda:{'id':'isolated-synthetic'})
    app=FastAPI();app.include_router(create_workspace_router(ctx,service,SimpleNamespace(max_running=1)))
    with TestClient(app) as client:
        for details in (False,True):
            response=client.get('/api/v1/workspaces/server-load',params={'details':str(details).lower()});assert response.status_code==200,response.text
            if not details:assert response.json()['cpu_percent'] is None and response.json()['memory_percent'] is None
    import numpy as np
    from scipy.io import wavfile
    from ptb_worker.egg_interactive import InteractivePreview
    from ptb_api.egg_models import EggTaskConfig
    fs=44100;t=np.arange(round(fs*77.249))/fs;raw=io.BytesIO();wavfile.write(raw,fs,np.column_stack([.4*np.sin(2*np.pi*120*t)+.03*np.sin(2*np.pi*240*t),.3*np.sin(2*np.pi*120*t+.3)]).astype('float32'));source=raw.getvalue()
    service=InteractivePreview(memory_bytes=1_000_000_000)
    try:
        started=time.monotonic();opened=service.open('synthetic-performance',source);records.append(dict(case='open',seconds=time.monotonic()-started))
        config=EggTaskConfig(mode='preview',roi_start=4.85,roi_end=5.30,micro_center=5.075,micro_width_ms=73.21,keep_praat_f0=False,keep_reaper_f0=False,keep_gci_f0=False).model_dump()
        initial_audio=None
        for name,changes in [('initial',{}),('micro_wheel',{'micro_width_ms':60}),('main_wheel',{'roi_start':4.9,'roi_end':5.25}),('swap',{'flip_channels':True}),('micro_after_swap',{'micro_width_ms':50})]:
            config.update(changes);started=time.monotonic();value=service.update('synthetic-performance',opened['session_id'],config,auto_db=True)
            if name in ('initial','swap'):assert value['audio_base64']
            else:assert value['audio_base64'] is None
            lo,hi=value['preview']['suggested_db_range'];assert (value['config']['spec_vmin'],value['config']['spec_vmax'])==(lo,hi)
            assert value['input_sha256']==opened['sha256'] and value['sample_count']==3406681
            records.append(dict(case=name,seconds=time.monotonic()-started,audio_base64_bytes=len(value.get('audio_base64') or ''),sample_count=value['sample_count']))
        cache_fd,source_fd=service.linux_cache_fd,service.linux_source_fd
        assert os.fstat(cache_fd).st_size>0 and os.fstat(source_fd).st_size==len(source)
    finally:service.close()
    for fd in (cache_fd,source_fd):
        try:os.fstat(fd);raise AssertionError('Session memory not released')
        except OSError:pass
    report=dict(success=True,profile_sha256=hashlib.sha256(profile_path.read_bytes()).hexdigest(),checks=['host_cache_cross_process_metadata','restored_mtime_invalidation','unsafe_cache_permissions_rejected','ram_checkpoint_scientific_parity','display_reuses_preprocessing','damaged_ram_cache_falls_back','anonymous_memory_released','task_only_no_host_probe','one_second_shared_counts','one_second_detailed_cpu','real_pg_readonly_aggregate_http','77_second_audio_transfer_once','auto_db_one_rpc','channel_invalidation'],records=records)
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))


if __name__=='__main__':main()
