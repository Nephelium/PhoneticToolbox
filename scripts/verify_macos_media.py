"""Actual FaceMesh long-video and repeated-processing verification on macOS.

Inputs are generated from the project's public NASA portrait fixture. Every
decoded frame is analyzed. Reports distinguish synthetic media from devices.
"""
from pathlib import Path
from fractions import Fraction
import csv,ctypes,gc,hashlib,json,os,shutil,sys,time,traceback,wave

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for raw in iter(lambda:stream.read(1048576),b''):h.update(raw)
    return h.hexdigest()

def prepare(directory):
    import av,cv2,numpy as np
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    image=cv2.imread(str(directory/'astronaut.png'));assert image is not None
    portrait=cv2.resize(image,(720,720));base=np.zeros((720,1280,3),np.uint8);base[:,280:1000]=portrait
    images=[np.ascontiguousarray(np.roll(base,shift,axis=1)) for shift in (-2,0,2)]
    cases=[]
    for name,seconds,fps in [('long-12min-10fps',720,10),('long-5min-25fps',300,25),('short-visible-4s-25fps',4,25)]:
        target=directory/(name+'.mp4');count=seconds*fps
        if not target.exists():
            with av.open(str(target),'w') as media:
                video=media.add_stream('libx264',rate=fps);video.width=1280;video.height=720;video.pix_fmt='yuv420p';video.options={'preset':'ultrafast','crf':'32'}
                audio=media.add_stream('aac',rate=48000);audio.layout='mono';cursor=0
                for i in range(count):
                    while cursor<round((i+1)*48000/fps):
                        size=min(1024,seconds*48000-cursor)
                        if not size:break
                        values=(.12*np.sin((np.arange(size)+cursor)*2*np.pi*330/48000)).astype(np.float32).reshape(1,-1)
                        af=av.AudioFrame.from_ndarray(values,format='fltp',layout='mono');af.sample_rate=48000;af.time_base=Fraction(1,48000);af.pts=cursor;cursor+=size
                        for packet in audio.encode(af):media.mux(packet)
                    # The unshifted NASA portrait is a positive control. The
                    # translated long clips also exercise explicit missing faces.
                    image=images[1] if name.startswith('short-visible') else images[(i//fps)%3]
                    frame=av.VideoFrame.from_ndarray(image,format='bgr24');frame.time_base=Fraction(1,fps);frame.pts=i
                    for packet in video.encode(frame):media.mux(packet)
                for stream in (audio,video):
                    for packet in stream.encode():media.mux(packet)
        assert target.stat().st_size<128_000_000
        cases.append(dict(name=name,file=target.name,seconds=seconds,fps=fps,frames=count,bytes=target.stat().st_size,sha256=sha(target)))
        print('MEDIA INPUT READY',name,count,flush=True)
    report=dict(source='ASSET-M05-NASA; task-generated H264/AAC, small deterministic image translations; not natural long recording',portrait_sha256=sha(directory/'astronaut.png'),resolution=[1280,720],audio_rate=48000,cases=cases)
    (directory/'inputs.json').write_text(json.dumps(report,indent=2)+'\n');return report

def main(output,inputs):
    if sys.platform!='darwin':raise RuntimeError('macOS required')
    import av,cv2,numpy as np,psutil
    from ptb_worker.m05_executor import run_child
    from ptb_worker.m05_video import analyze_video,JsonLinesSink,VideoLimits
    from phonetic_core.lip.sequence import LipConfig
    output=Path(output);inputs=Path(inputs);output.mkdir(parents=True,exist_ok=False);scratch=output/'scratch';scratch.mkdir()
    spec=json.loads((inputs/'inputs.json').read_text());process=psutil.Process();results=[];cleanups=[]
    report=dict(success=False,platform=sys.platform,frozen=bool(getattr(sys,'frozen',False)),input_spec=spec,results=results,cleanup=cleanups)
    def save():(output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    def cleanup(directory):
        assert directory.resolve().is_relative_to(scratch.resolve()) and directory!=scratch
        files=[]
        for path in directory.rglob('*'):
            if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):raise RuntimeError('Unexpected scratch link')
            if path.is_file():files.append(dict(path=str(path.relative_to(output)),bytes=path.stat().st_size,sha256=sha(path)));path.unlink()
        for path in sorted(directory.rglob('*'),key=lambda p:len(p.parts),reverse=True):
            if path.is_dir():path.rmdir()
        directory.rmdir();cleanups.append(dict(success=True,files=files))
    def namespace():
        objc=ctypes.CDLL('/usr/lib/libobjc.A.dylib');objc.objc_getClass.argtypes=[ctypes.c_char_p];objc.objc_getClass.restype=ctypes.c_void_p
        values=[objc.objc_getClass(name.encode()) for name in ('AVFFrameReceiver','AVFAudioReceiver','PTBPyAV8AVFFrameReceiver','PTBPyAV8AVFAudioReceiver')]
        assert all(values) and len(set(values))==4
        assert all(name in av.formats_available for name in ('avfoundation','lavfi','audiotoolbox'))
        return dict(check='objective_c_private_classes',success=True,all_four_distinct=True,devices_preserved=True)
    config=dict(filter_enabled=True,cutoff_hz=15.,animation='none',quality='standard',offset=0.)
    def owned(case,name,cancel_at=None,broken=False):
        directory=scratch/name;directory.mkdir();target=directory/'input.mp4'
        if broken:target.write_bytes(b'not a valid video')
        else:subprocess_copy(inputs/case['file'],target)
        progress_count=[0];evidence={};started=time.monotonic();last_notice=[0.]
        def progress(value):
            progress_count[0]=value.get('frames',0)
            if time.monotonic()-last_notice[0]>15:
                print('MEDIA PROGRESS',name,progress_count[0],flush=True);last_notice[0]=time.monotonic()
        try:
            response=run_child(directory,dict(input=target.name,original_name=target.name,config=config),lambda:cancel_at is not None and progress_count[0]>=cancel_at,progress,evidence)
            if cancel_at or broken:raise AssertionError('Expected bounded failure')
            assert response['success'] and response['frames']==case['frames']
            folder=directory/'results';meta=json.loads((folder/'manifest.json').read_text());assert meta['complete'] and meta['validity']['detected']>=.99*case['frames']
            assert meta['timing']['decoded_frames']==case['frames'] and meta['validity']['imputed_measurements']==0
            count=0
            with (folder/'frames.jsonl').open() as stream:
                for count,line in enumerate(stream,1):
                    row=json.loads(line);assert row['index']==count-1 and abs(row['time_s']-(count-1)/case['fps'])<1e-9
                    if row['detected']:
                        assert len(row['points'])==len(row['raw_points'])==478
                        assert all(np.isfinite(list(row['metrics'].values())))
            assert count==case['frames']
            with (folder/'measurements.csv').open(encoding='utf-8-sig') as stream:assert sum(1 for _ in csv.DictReader(stream))==case['frames']
            with wave.open(str(folder/'audio_recording.wav'),'rb') as audio:
                assert audio.getframerate()==48000 and audio.getnchannels()==1
                assert case['seconds']*48000<=audio.getnframes()<=case['seconds']*48000+1024
                values=np.frombuffer(audio.readframes(48000),'<i2').astype(float)
                correlation=float(np.corrcoef(values,np.sin(np.arange(len(values))*2*np.pi*330/48000))[0,1]);assert correlation>.99
            for item in meta['files']:assert sha(folder/item['name'])==item['sha256']
            preview=json.loads((folder/'preview.json').read_text());assert preview['full_frame_count']==case['frames'] and len(preview['rows'])<=241
            results.append(dict(check=name,success=True,frames=count,detected=meta['validity']['detected'],fps=case['fps'],media_seconds=case['seconds'],wall_seconds=time.monotonic()-started,audio_correlation=correlation,resources=evidence,exchange_available=preview['exchange_available'],files=meta['files']))
        except ValueError as exc:
            assert cancel_at is not None or broken,str(exc)
            if cancel_at is not None:assert str(exc)=='cancelled' and progress_count[0]>=cancel_at
            else:
                response=json.loads((directory/'response.json').read_text());assert response['success'] is False
            assert not (directory/'results/manifest.json').exists()
            results.append(dict(check=name,success=True,expected_failure=str(exc),resources=evidence,frames_before_stop=progress_count[0]))
        assert evidence.get('group_cleaned') is True,evidence
        cleanup(directory);save();print('MEDIA CHECK PASS',name,flush=True)
    try:
        results.append(namespace());save()
        for case in spec['cases'][:2]:owned(case,case['name'])
        short=spec['cases'][2];samples=[]
        for index in range(24):
            directory=scratch/('repeat-'+str(index));directory.mkdir()
            with (directory/'frames.jsonl').open('xb') as stream:
                meta=analyze_video(inputs/short['file'],JsonLinesSink(stream,400_000_000),LipConfig(),limits=VideoLimits(input_bytes=128_000_000,output_bytes=400_000_000))
            assert meta['timing']['decoded_frames']==short['frames'] and meta['validity']['detected']>=.99*short['frames']
            gc.collect();samples.append(dict(index=index,rss_bytes=process.memory_info().rss,open_fds=process.num_fds()))
            cleanup(directory)
            if index%4==3:print('CONTINUOUS RUNS',index+1,'rss',samples[-1]['rss_bytes'],flush=True)
        before=np.median([r['rss_bytes'] for r in samples[4:10]]);after=np.median([r['rss_bytes'] for r in samples[-6:]])
        assert after-before<128*1024**2,(before,after,samples)
        assert max(r['open_fds'] for r in samples[4:])-min(r['open_fds'] for r in samples[4:])<=8
        results.append(dict(check='same_process_24_runs',success=True,samples=samples,post_warmup_rss_growth_bytes=float(after-before),frames_per_run=short['frames']));save()
        owned(spec['cases'][0],'cancel-long',cancel_at=50)
        owned(short,'recover-after-cancel')
        owned(short,'corrupt-input',broken=True)
        owned(short,'recover-after-corrupt')
        report['success']=True;save();print('MACOS MEDIA LONG AND CONTINUOUS PASSED',flush=True);return 0
    except BaseException:
        report['failure']=traceback.format_exc();save();raise

def subprocess_copy(source,target):
    import subprocess
    subprocess.run(['/bin/cp','-c',str(source),str(target)],check=True)

if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare(sys.argv[2])
    else:raise SystemExit(main(*sys.argv[1:]))
