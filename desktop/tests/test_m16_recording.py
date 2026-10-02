"""Deterministic device/storage tests. No physical microphone is opened."""
import copy
import hashlib
import time
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
from ptb_desktop.recording.capture import Capture
from ptb_desktop.recording.storage import Project,save_pcm,read_range,read_json,atomic_json,validate_span
from ptb_desktop.recording.service import RecordingService
from ptb_desktop.recording.jobs import process_audio,export_audio


def config():return {'sample_rate':48000,'channels':2,'roles':['microphone','egg'],'gain_db':18,'device_index':0,'device':'test','device_name':'Synthetic test device'}
def signal(n=8000):
    t=np.arange(n)/48000
    return np.column_stack((.15*np.sin(2*np.pi*137*t),.7*np.sin(2*np.pi*95*t))).astype(np.float32)


def recorded(tmp_path,n=10000):
    p=Project(tmp_path/'project',True);c=Capture(p.root,config(),chunk_frames=2048);c.start(open_stream=False);x=signal(n)
    for start in range(0,n,512):c.submit(x[start:start+512])
    result=c.stop();assert not result['error'];return p,c,x,result


def test_raw_gain_not_baked_and_cross_segments(tmp_path):
    p,c,x,r=recorded(tmp_path)
    assert len(r['spans'])==5
    assert np.array_equal(read_range(p.root,r['spans'],0,len(x)),x)
    assert r['quality']['peak'][0]<.16
    for span in r['spans']:validate_span(p.root,span,hash_check=True)
    p.close()


def test_queue_overflow_cannot_report_success(tmp_path):
    p=Project(tmp_path/'p',True);c=Capture(p.root,config(),queue_blocks=1)
    c.submit(signal(100));c.submit(signal(100));assert c.error and '满' in c.error
    with pytest.raises(RuntimeError,match='队列'):c.start(open_stream=False)
    result=c.stop();assert result['status']=='recovered_partial';assert result['frames']==100;p.close()


def test_nonfinite_stops_and_keeps_confirmed_prefix(tmp_path):
    p=Project(tmp_path/'p',True);c=Capture(p.root,config(),chunk_frames=512);c.start(open_stream=False);c.submit(signal(512))
    end=time.monotonic()+3
    while c.written<512 and time.monotonic()<end:time.sleep(.01)
    x=signal(512);x[7,0]=np.nan;c.submit(x);result=c.stop();assert result['error'];assert result['frames']==512;p.close()


def test_disk_write_failure_is_visible_and_recoverable(tmp_path,monkeypatch):
    import ptb_desktop.recording.capture as module
    p=Project(tmp_path/'p',True)
    monkeypatch.setattr(module,'save_pcm',lambda *args:(_ for _ in ()).throw(OSError('injected disk full')))
    c=Capture(p.root,config(),chunk_frames=100);c.start(open_stream=False);c.submit(signal(100));result=c.stop();assert 'injected disk full' in result['error'];assert result['frames']==0;p.close()


def test_single_writer_and_release(tmp_path):
    p=Project(tmp_path/'p',True)
    with pytest.raises(ValueError,match='第二写者'):Project(p.root)
    p.close();q=Project(tmp_path/'p');q.close()


def test_new_project_rejects_nonempty_directory(tmp_path):
    (tmp_path/'original.txt').write_text('preserve')
    with pytest.raises(ValueError,match='空目录'):Project(tmp_path,True)
    assert list(tmp_path.iterdir())==[tmp_path/'original.txt']


def test_manifest_replace_failure_preserves_previous(tmp_path,monkeypatch):
    import ptb_desktop.recording.storage as module
    p=Project(tmp_path/'p',True);old=(p.root/'project.json').read_bytes();original=module.os.replace
    def fail(src,dst):
        if Path(dst).name=='project.json':raise PermissionError('injected occupied file')
        return original(src,dst)
    monkeypatch.setattr(module.os,'replace',fail)
    data=copy.deepcopy(p.data);data['tasks']=[{'id':'a'}]
    with pytest.raises(PermissionError):p.commit(data)
    assert (p.root/'project.json').read_bytes()==old;assert p.data['tasks']==[];p.close()


def test_recovery_checks_hash_and_truncates_bad_tail(tmp_path):
    p,c,x,r=recorded(tmp_path);root=p.root;p.close();tail=root/r['spans'][-1]['file'];tail.write_bytes(b'corrupt')
    q=Project(root);assert len(q.recoveries)==1;assert len(q.recoveries[0]['spans'])==4;assert q.recoveries[0]['status']=='recovered_partial';q.close()


def test_edits_undo_restore_persist_and_raw_hashes(tmp_path):
    p,c,x,r=recorded(tmp_path);service=RecordingService();service.project=p;service.commit_capture(copy.deepcopy(r));raw_hashes={s['file']:s['sha256'] for s in r['spans']};id=r['id']
    service.dispatch({'op':'edit','id':id,'action':'cut','start':100,'end':101});take=service.take(id)
    assert np.array_equal(read_range(p.root,take['versions'][take['head']]['spans'],0,len(x)-1),np.delete(x,100,axis=0))
    service.dispatch({'op':'edit','id':id,'action':'paste','start':100,'end':100});take=service.take(id);assert np.array_equal(read_range(p.root,take['versions'][take['head']]['spans'],0,len(x)),x)
    service.dispatch({'op':'undo','id':id});assert service.view()['takes'][0]['versions'][service.take(id)['head']]['frames']==len(x)-1
    service.dispatch({'op':'redo','id':id});service.dispatch({'op':'restore','id':id})
    for file,digest in raw_hashes.items():assert hashlib.sha256((p.root/file).read_bytes()).hexdigest()==digest
    service.close();q=Project(tmp_path/'project');assert len(q.data['takes'][0]['versions'])==4;q.close()


def test_task_snapshots_and_rerecord_not_overwritten(tmp_path):
    p,c,x,r=recorded(tmp_path);service=RecordingService();service.project=p;r['task_snapshot']={'id':'T01','prompt':'old'};service.commit_capture(copy.deepcopy(r));old_id=r['id']
    service.dispatch({'op':'tasks','tasks':[{'id':'T01','prompt':'new','title':'','filename_stem':'one','enabled':True,'skipped':False}]})
    second=copy.deepcopy(r);second['id']='second';second['task_snapshot']['prompt']='new';service.commit_capture(second)
    assert service.take(old_id)['task_snapshot']['prompt']=='old';assert len(service.view()['takes'])==2;service.close()


def test_export_formats_segments_no_overwrite(tmp_path):
    p,c,x,r=recorded(tmp_path);item={**r,'name':'中文 ɑ̃','version':{'id':'v','kind':'raw','spans':r['spans']}}
    result=tmp_path/'result.json'
    for subtype in ('FLOAT','PCM_24','PCM_16'):
        export_audio(p.root,[item],tmp_path/'exports',subtype,result,tmp_path/'cancel',max_frames=3000)
        report=read_json(result);assert report['success_count']==1;files=report['items'][0]['files'];assert len(files)==4
        actual=np.concatenate([sf.read(tmp_path/'exports'/report['directory_name']/f['name'],dtype='float32',always_2d=True)[0] for f in files])
        assert len(actual)==len(x);assert np.max(np.abs(actual-x))<= (0 if subtype=='FLOAT' else 2**(-15 if subtype=='PCM_16' else -23))
    p.close()


def test_export_pcm_overflow_rejected_float_preserved(tmp_path):
    p=Project(tmp_path/'p',True);x=signal();x[0,0]=1.1;span=save_pcm(p.root,'takes/a/raw/000.f32',x)
    item={'id':'a','config':config(),'version':{'id':'v','kind':'raw','spans':[span]}}
    result=tmp_path/'result.json';export_audio(p.root,[item],tmp_path/'exports','PCM_16',result,tmp_path/'cancel');assert read_json(result)['success_count']==0
    export_audio(p.root,[item],tmp_path/'exports','FLOAT',result,tmp_path/'cancel');assert read_json(result)['success_count']==1;p.close()


def test_processing_egg_unchanged_and_cancel(tmp_path):
    p=Project(tmp_path/'p',True);rng=np.random.default_rng(16);x=signal(300000);x[:,0]+=rng.normal(0,.015,len(x)).astype(np.float32);x[:8000,0]=rng.normal(0,.015,8000)
    spans=[save_pcm(p.root,f'takes/a/raw/{i}.f32',x[start:start+131072]) for i,start in enumerate(range(0,len(x),131072))]
    request={'id':'derive','kind':'denoise','config':config(),'spans':spans,'strength':1.0,'noise':{'spans':spans,'start':0,'end':8000,'channel':0}}
    result=tmp_path/'result.json';process_audio(p.root,request,result,tmp_path/'cancel');out=read_json(result);assert out['state']=='complete';y=read_range(p.root,out['spans'],0,len(x));assert np.array_equal(y[:,1],x[:,1]);assert y.shape==x.shape;assert np.std(y[:8000,0])<np.std(x[:8000,0])
    from phonetic_core.recording import noise_profile,denoise
    profile,_=noise_profile(x[:8000,0]);whole=denoise(x,profile,[0],1);assert np.max(np.abs(y-whole))<2e-6
    (tmp_path/'cancel').touch();request['id']='cancelled';process_audio(p.root,request,result,tmp_path/'cancel');assert read_json(result)['state']=='cancelled';p.close()


def test_path_escape_rejected(tmp_path):
    p=Project(tmp_path/'p',True)
    with pytest.raises(ValueError,match='路径'):save_pcm(p.root,'../escape.f32',signal())
    p.close()


def test_live_spectrum_covers_same_five_seconds(tmp_path):
    p=Project(tmp_path/'p',True);c=Capture(p.root,config(),probe=True,queue_blocks=512);c.start(open_stream=False)
    x=signal(240000)
    for start in range(0,len(x),1024):c.submit(x[start:start+1024])
    c.stop();v=c.preview(True,0);assert v['window_frames']==240000;assert v['spectrum']['times'][-1]>4.9;assert len(v['spectrum']['rows'])<=128
    off=c.preview(False);assert off['spectrum'] is None;p.close()


def test_stream_stop_failure_still_closes_owned_stream(tmp_path):
    p=Project(tmp_path/'p',True);c=Capture(p.root,config());c.start(open_stream=False)
    class Stream:
        closed=False
        def stop(self):raise RuntimeError('injected stop failure')
        def close(self):self.closed=True
    stream=Stream();c.stream=stream;result=c.stop();assert stream.closed;assert c.stream is None;assert 'stop failure' in result['error'];p.close()


def test_stream_close_failure_retains_reference_for_retry(tmp_path):
    p=Project(tmp_path/'p',True);c=Capture(p.root,config());c.start(open_stream=False)
    class Stream:
        fail=True
        def stop(self):pass
        def close(self):
            if self.fail:raise RuntimeError('injected close failure')
    stream=Stream();c.stream=stream
    with pytest.raises(RuntimeError,match='仍未释放'):c.stop()
    assert c.stream is stream;stream.fail=False;c.stop();assert c.stream is None;p.close()


def test_sleep_gap_marked_incomplete_not_silently_concatenated(tmp_path):
    p=Project(tmp_path/'p',True);c=Capture(p.root,config());c.start(open_stream=False);c.submit(signal(512));c.last_callback-=4;c.submit(signal(512));result=c.stop();assert '中断' in result['error'];assert result['frames']==512;p.close()


def test_native_device_defaults_two_audio_and_single_input_fallback():
    from ptb_desktop.recording.devices import devices,validate_config
    class Backend:
        channels=2
        def query_hostapis(self):return [{'name':'test'}]
        def query_devices(self):return [{'name':'audio','hostapi':0,'max_input_channels':self.channels,'max_output_channels':2,'default_samplerate':48000}]
        def check_input_settings(self,**kwargs):pass
    backend=Backend();device=devices(backend)[0]['id'];cfg=validate_config({'device':device},backend);assert cfg['roles']==['microphone','microphone'];backend.channels=1;device=devices(backend)[0]['id'];cfg=validate_config({'device':device},backend);assert cfg['roles']==['microphone']


def test_residual_playback_is_source_minus_processed_on_same_frames(tmp_path):
    from ptb_desktop.recording.devices import Playback,devices
    class Stream:
        def start(self):pass
        def abort(self):pass
        def close(self):pass
    class Backend:
        def query_hostapis(self):return [{'name':'test'}]
        def query_devices(self):return [{'name':'out','hostapi':0,'max_input_channels':2,'max_output_channels':2,'default_samplerate':48000}]
        def check_output_settings(self,**kwargs):pass
        def OutputStream(self,**kwargs):return Stream()
    p=Project(tmp_path/'p',True);x=signal(1000);source=save_pcm(p.root,'takes/a/raw/0.f32',x);derived=save_pcm(p.root,'derived/a/0.f32',x*.5);backend=Backend()
    player=Playback(p.root,[derived],config(),devices(backend)[0]['id'],0,1,backend,reference_spans=[source]);assert np.array_equal(player.current,x[:,0]*.5);player.stop();p.close()


@pytest.mark.parametrize('fail_at',['construct','start'])
def test_output_start_failure_reclaims_reader_and_stream(tmp_path,fail_at):
    import threading
    from ptb_desktop.recording.devices import Playback,devices
    class Stream:
        closed=False
        def start(self):raise RuntimeError('injected start')
        def abort(self):raise RuntimeError('injected abort')
        def close(self):self.closed=True
    class Backend:
        stream=Stream()
        def query_hostapis(self):return [{'name':'test'}]
        def query_devices(self):return [{'name':'out','hostapi':0,'max_input_channels':2,'max_output_channels':2,'default_samplerate':48000}]
        def check_output_settings(self,**kwargs):pass
        def OutputStream(self,**kwargs):
            if fail_at=='construct':raise RuntimeError('injected construct')
            return self.stream
    p=Project(tmp_path/'p',True);span=save_pcm(p.root,'takes/a/raw/0.f32',signal(200000));backend=Backend();existing={t.ident for t in threading.enumerate()}
    with pytest.raises(RuntimeError,match='injected'):Playback(p.root,[span],config(),devices(backend)[0]['id'],0,1,backend)
    assert not [t for t in threading.enumerate() if t.name=='ptb-m16-play-reader' and t.ident not in existing]
    if fail_at=='start':assert backend.stream.closed
    p.close()


def test_output_close_failure_stays_owned_until_retry(tmp_path):
    from ptb_desktop.recording.devices import devices
    class Stream:
        fail=True
        def start(self):raise RuntimeError('injected start')
        def abort(self):raise RuntimeError('injected abort')
        def close(self):
            if self.fail:raise RuntimeError('injected close')
    class Backend:
        stream=Stream()
        def query_hostapis(self):return [{'name':'test'}]
        def query_devices(self):return [{'name':'out','hostapi':0,'max_input_channels':2,'max_output_channels':2,'default_samplerate':48000}]
        def check_output_settings(self,**kwargs):pass
        def OutputStream(self,**kwargs):return self.stream
    p,c,x,r=recorded(tmp_path);backend=Backend();service=RecordingService(backend=backend);service.project=p;service.commit_capture(copy.deepcopy(r))
    with pytest.raises(RuntimeError,match='尚未释放'):service.dispatch({'op':'play','id':r['id'],'device':devices(backend)[0]['id']})
    assert service.player is not None and service.player.stream is backend.stream
    with pytest.raises(RuntimeError,match='仍未释放'):service.dispatch({'op':'play_stop'})
    assert service.player is not None;backend.stream.fail=False;service.dispatch({'op':'play_stop'});assert service.player is None;assert service.close()
