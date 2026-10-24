"""Exercise actual fixed scientific children, native libraries and media on Mac.

Fixtures are public repository fixtures. Outputs use a fresh owned directory.
This same harness runs from the source entry and from the final frozen app.
"""
import hashlib,io,json,struct,sys,time
from pathlib import Path
from uuid import uuid4

def main(output,fixtures):
    import numpy as np
    import soundfile as sf
    from ptb_worker.native.darwin import run,bundle_root,fingerprint
    from ptb_worker.io.limits import Limits
    from ptb_worker.m06_executor import unpack as m06_unpack
    root=bundle_root();output=Path(output);fixtures=Path(fixtures);output.mkdir(parents=True,exist_ok=False)
    work=output/'scratch';work.mkdir();checks=[];processes=[]
    from macos_validation_identity import runtime_hashes,require_identity
    identity=runtime_hashes(root)
    report=dict(source_sha256=identity,success=False,platform=sys.platform,frozen=bool(getattr(sys,'frozen',False)),runtime=fingerprint(),checks=checks,processes=processes)
    def save():
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        (output/'cleanup.json').write_text(json.dumps(dict(state='retained',scratch=str(work),reason='Mac owned scratch cleanup must follow verification reads; no shared directory deletion'),indent=2)+'\n')
    def digest(raw):return hashlib.sha256(raw).hexdigest()
    def wav(x,rate):
        f=io.BytesIO();sf.write(f,x,rate,format='WAV',subtype='FLOAT');return f.getvalue()
    def child(entry,header,raw=b''):
        request=work/(entry+'-request.json');request.write_bytes(json.dumps(header).encode()+b'\n'+raw);e={}
        value=run(entry,request,work,Limits(input_bytes=1,output_bytes=128_000_000,process_bytes=2_147_483_648,timeout_seconds=180),evidence=e)
        processes.append(dict(entry=entry,**e));return value
    def unpack(raw,width=8):
        n=struct.unpack_from('<Q' if width==8 else '<I',raw)[0];head=json.loads(raw[width:width+n]);assert 'error' not in head,head
        offset=width+n;files={}
        for item in head['files']:
            size=item.get('size',item.get('size_bytes'));blob=raw[offset:offset+size];offset+=size
            assert len(blob)==size and digest(blob)==item['sha256'];files[item['name']]=blob
        assert offset==len(raw);return files
    def streamed(raw):
        files={}
        while raw:
            n=struct.unpack_from('<I',raw)[0];head=json.loads(raw[4:4+n]);raw=raw[4+n:];assert head['kind']!='error',head
            if head['kind']=='complete':assert not raw and head['count']==len(files);return files
            size=head['size'];blob=raw[:size];raw=raw[size:];assert digest(blob)==head['sha256'];files[head['name']]=blob
        raise AssertionError('Missing complete frame')
    try:
        from phonetic_core.synthesis.klatt.api import defaults,generate,export_parameters
        c=defaults();c['sequence']='aiu';c['duration']=.3
        for curve in c['curves'].values():curve['points'][-1][0]=.3
        generated=generate(c);raw=wav(.2*np.sin(np.arange(16000)*2*np.pi*150/16000),16000)
        for action,params,data in [('generate',export_parameters(c),b''),('synthesize',export_parameters(generated),b''),('extract',export_parameters(c),raw)]:
            files=dict(m06_unpack(child('m06',dict(parameters=params,action=action,seed=42,input_sha256=digest(data)),data),digest(data),action))
            assert 'm06.ptb.json' in files
            if action=='synthesize':
                values,sr=sf.read(io.BytesIO(files['synthesis.wav']));assert sr==16000 and len(values)==4800 and np.isfinite(values).all()
            checks.append('m06_'+action)
        source=work/'source.wav';source.write_bytes(raw);native=work/'native.wav';native.touch()
        preview=None
        for config in [dict(action='preview'),dict(action='transform',pitch_ratio=1.2,pitch_hz=20),dict(action='linear',start=0,end=1,points=[dict(time=.1,freqs=[120,180],mode='full'),dict(time=.8,freqs=[140,200],mode='full')])]:
            # M08 uses a plain JSON request rather than a JSON + binary line.
            request=work/'m08.json';request.write_text(json.dumps(dict(config=config,sha256=digest(raw),audio_path=str(source),native_path=str(native),input_name='synthetic.wav')))
            e={};files=streamed(run('m08',request,work,Limits(input_bytes=1,output_bytes=64_000_000,process_bytes=1_000_000_000,timeout_seconds=120),evidence=e));processes.append(dict(entry='m08',**e))
            meta=json.loads(files['m08.ptb.json']);assert len(meta['original_f0'])>10 and native.stat().st_size==0
            checks.append('m08_'+config['action'])
        from ptb_worker.store import LOCAL_PROJECT
        data=np.load(fixtures/'m07/v2.npz');spec=json.loads((fixtures/'m07/v2.json').read_text())
        blobs={role:wav(data['input'+str(i)].astype(float)/32768,spec['cases'][i]['fs']) for i,role in enumerate(('source','target'))}
        refs={role:dict(asset_id=str(uuid4()),sha256=digest(blob)) for role,blob in blobs.items()}
        request=dict(project_id=LOCAL_PROJECT,idempotency_key='macos-runtime-test',action='analyze',**refs)
        def m07(req,inputs):
            h=dict(request=req,request_hash='a'*64,inputs=[dict(role=r,size_bytes=len(v),sha256=digest(v)) for r,v in inputs.items()])
            return unpack(child('m07',h,b''.join(inputs.values())),4)
        files=m07(request,blobs);analysis=files['analysis.m07.json'];checks.append('m07_analysis')
        files=m07(dict(request,action='generate',analysis_job_id=str(uuid4()),generation=dict(step_count=3)),dict(blobs,analysis=analysis));assert len([n for n in files if n.endswith('.wav')])==4;checks.append('m07_generation')
        egg=np.load(fixtures/'m03/EGG-SYN-PCM16.npz');audio=wav(egg['load.audio_signal'],44100);stereo=wav(np.column_stack([egg['load.egg_signal_raw'],egg['load.audio_signal']]),44100)
        font=dict(schema_version='font/1',zh='PingFang SC',latin='DejaVu Sans',ipa='Doulos SIL',size_px=14,fallback_policy='portable')
        for mode in ('preview','batch','inverse'):
            cfg=dict(mode=mode,font=font,keep_reaper_f0=False)
            if mode!='batch':cfg.update(roi_start=.1,roi_end=.4)
            if mode=='preview':cfg['micro_center']=.2
            files=unpack(child('egg',dict(sha256=digest(stereo),input_name='egg.wav',config=cfg),stereo))
            assert 'egg.ptb.json' in files;checks.append('egg_'+mode)
        files=unpack(child('lpc',dict(sha256=digest(audio),audio_size=len(audio),textgrid_sha256=None,input_name='audio.wav',config=dict(roi_start=.1,roi_end=.2,font=font)),audio))
        assert set(files)=={'lpc.ptb.json','lpc_SPECTRUM.png','lpc_AUDIO.wav'};checks.append('lpc_plot_audio')
        # M01 runs both the Python fallback policy and the actual arm64 REAPER.
        for policy in ('disabled','native_required'):
            request=dict(project_id=LOCAL_PROJECT,idempotency_key='macos-acoustic-'+policy,inputs=dict(audio=dict(asset_id=str(uuid4()),sha256=digest(audio))),config=dict(settings=dict(only_voiced=False),backend_policy=dict(reaper=policy)))
            files=unpack(child('acoustic',dict(request=request,inputs=[dict(id=request['inputs']['audio']['asset_id'],role='audio',size_bytes=len(audio),sha256=digest(audio),expires_at=4102444800)],native_scratch=str(native),reaper_binary=str(root/'resources/research/reaper')),audio))
            assert set(files)=={'result.xlsx','result.ptb.sqlite','result.ptb.json'};checks.append('m01_'+policy)
        from ptb_worker.native.darwin import OwnedProcess,group_members
        from ptb_worker.process_entry import command
        streamed_root=work/'acoustic-stream';streamed_root.mkdir();(streamed_root/'input.wav').write_bytes(audio)
        cfg=dict(settings=dict(only_voiced=False),backend_policy=dict(reaper='native_required'),extended={})
        (streamed_root/'request.json').write_text(json.dumps(dict(config=cfg,index='0',operation='acoustic_analysis',audio_name='public.wav',audio_sha256=digest(audio),reaper_binary=str(root/'resources/research/reaper'))))
        process=OwnedProcess(command('ptb_worker.acoustic_stream_child',str(streamed_root)),streamed_root,2_147_483_648)
        def finish_file_child(process,response):
            try:
                deadline=time.monotonic()+180
                while process.poll() is None:
                    if time.monotonic()>deadline:raise AssertionError('File child timeout')
                    time.sleep(.05)
                value=json.loads(response.read_text());assert process.poll()==0,value;return value
            finally:process.close();assert process.group_cleaned
        value=finish_file_child(process,streamed_root/'response.json');assert value['success'],value;checks.append('m01_streamed_local_files')
        table_path=streamed_root/'result.ptb.sqlite';parameter_root=work/'parameter-view';parameter_root.mkdir()
        (parameter_root/'request.json').write_text(json.dumps(dict(source=str(table_path),cache=str(work/'view-cache.sqlite'),name='result.ptb.sqlite',sha256=digest(table_path.read_bytes()),view=None)))
        process=OwnedProcess(command('ptb_worker.parameter_bundle_child',str(parameter_root)),parameter_root,1_073_741_824)
        value=finish_file_child(process,parameter_root/'response.json');assert 'result' in value,value;checks.append('m02_parameter_bundle_view')
        from ptb_worker.native.darwin_egg import InteractivePreview
        manager=InteractivePreview(reaper_binary=root/'resources/research/reaper')
        egg_path=work/'interactive-egg.wav';egg_path.write_bytes(stereo)
        try:
            opened=manager.open('macos-qa',b'',source_path=egg_path);token=opened['session_id'];pid=manager.process.pid
            for flip in (False,True,False):
                view=manager.update('macos-qa',token,dict(mode='preview',roi_start=.1,roi_end=.4,micro_center=.2,flip_channels=flip,f0_policy='audio-f0/2',keep_reaper_f0=True))
                assert view['sample_rate_hz']==44100 and view['sample_count']==len(egg['load.audio_signal'])
            manager.release('macos-qa',token);assert manager.process is None and not group_members(pid)
        finally:manager.close()
        checks.append('egg_persistent_file_roi_swap_reaper_and_close')
        long_path=work/'long-egg.wav';rate=16000
        with sf.SoundFile(str(long_path),'w',samplerate=rate,channels=2,subtype='FLOAT') as stream:
            t=np.arange(rate)/rate;block=np.column_stack([.2*np.sin(2*np.pi*150*t),.1*np.sin(2*np.pi*150*t+.2)]).astype('float32')
            for _ in range(125):stream.write(block)
        manager=InteractivePreview(reaper_binary=root/'resources/research/reaper')
        try:
            opened=manager.open('macos-qa-long',b'',source_path=long_path);assert opened['overview_base64']
            for start in (.1,100.):
                view=manager.update('macos-qa-long',opened['session_id'],dict(mode='preview',roi_start=start,roi_end=start+.3,micro_center=start+.1,keep_gci_f0=False,keep_praat_f0=False))
                assert view['sample_count']==125*rate and abs(view['audio_start_s']-start)<1/rate
        finally:manager.close()
        checks.append('egg_125_second_file_backed_two_distant_rois')
        from dataclasses import asdict
        from ptb_worker.segmentation import SEGMENT_LIMITS
        grid=('File type = "ooTextFile short"\n"TextGrid"\n0\n1\n<exists>\n1\n"IntervalTier"\n"syllable"\n0\n1\n3\n0\n0.2\n"a"\n0.2\n0.5\n"sil"\n0.5\n1\n"b"\n').encode()
        segment_header=dict(audio_size=len(raw),grid_size=len(grid),parent_size=0,audio_sha256=digest(raw),textgrid_sha256=digest(grid),layer='syllable',audio_name='synthetic.wav',legacy_name=None,limits=asdict(SEGMENT_LIMITS))
        files=unpack(child('segment',segment_header,raw+grid));assert len(files)==2
        assert sorted(len(sf.read(io.BytesIO(value))[0]) for value in files.values())==[3200,8000];checks.append('textgrid_exact_sample_slices')
        from ptb_desktop.recording.devices import devices
        found=devices();assert isinstance(found,list);report['audio_devices']=found;checks.append('coreaudio_device_enumeration')
        from ptb_worker.m14_jobs import execute
        table=(fixtures/'m14/public.xlsx').read_bytes();cfg=dict(action='preview',skip_first_row=True,consonant_only_as_zero_initial=True)
        files=unpack(child('m14',dict(sha256=digest(table),name='public.xlsx',config=cfg),table),4)
        preview=json.loads(files['m14-preview.json']);files=unpack(child('m14',dict(sha256=digest(table),name='public.xlsx',config=dict(cfg,action='export',settings=preview['config'],font=font)),table),4)
        assert len(files)==3;checks.append('m14_preview_three_exports')
        from ptb_worker.spec2wav_preview import render
        from ptb_worker.segmentation import unpack_bundle
        import cv2
        y,x=np.indices((96,160));gray=np.uint8(np.clip(255-150*np.exp(-((y-35-5*np.sin(x/18))/5)**2),0,255));png=cv2.imencode('.png',gray)[1].tobytes()
        config=dict(mode='image',time_start=0,time_end=.3,freq_start=0,freq_end=5000,n_iter=32,reconstruction_profile='calibrated',phase_initialization='auto',frequency_axis='linear',auto_parameters=False,win_length_ms=10,min_db=-60,max_db=0,target_sr=0,seed=0)
        assert render(png,dict(config,auto_parameters=True))['width']==160;checks.append('m09_parameter_preview')
        bundle=unpack_bundle(child('spec2wav',dict(config=config,sha256=digest(png)),png),64_000_000)
        files=dict(zip([f['name'] for f in bundle.manifest['files']],bundle.payloads));values,sr=sf.read(io.BytesIO(files['reconstructed.wav']));assert sr==10000 and len(values)==3000 and np.isfinite(values).all();checks.append('m09_image_reconstruction')
        from phonetic_core.vocal_tract.engine import Engine
        engine=Engine(resource_dir=root/'resources/vocal_tract/native')
        try:
            for count in (65,129,257):
                engine.set_section_count(count)
                for name in ('a','i','u','n','l','s'):
                    state=engine.snapshot(engine.presets[name]);assert len(state['centerline'])==count and np.isfinite(state['tube_areas']).all()
        finally:engine.close()
        checks.append('vtl_six_presets_three_resolutions')
        from fractions import Fraction
        import av
        webm=work/'mp4-source.webm'
        with av.open(str(webm),'w') as container:
            vs=container.add_stream('libvpx',rate=30);vs.width=320;vs.height=240;vs.pix_fmt='yuv420p'
            aus=container.add_stream('libopus',rate=48000);aus.layout='mono'
            for i in range(3):
                frame=av.VideoFrame.from_ndarray(np.zeros((240,320,3),dtype='uint8'),format='rgb24');frame.pts=i
                for packet in vs.encode(frame):container.mux(packet)
            pcm=av.AudioFrame.from_ndarray(np.zeros((1,4800),dtype='float32'),format='flt',layout='mono');pcm.sample_rate=48000;pcm.pts=0;pcm.time_base=Fraction(1,48000)
            for packet in aus.encode(pcm):container.mux(packet)
            for packet in aus.encode():container.mux(packet)
            for packet in vs.encode():container.mux(packet)
        mp4_root=work/'mp4';mp4_root.mkdir();(mp4_root/'request.json').write_text(json.dumps(dict(source=str(webm),target=str(mp4_root/'result.mp4'),config=dict(width=320,height=240,duration=.1))))
        from ptb_worker import vocal_mp4
        process=OwnedProcess([sys.executable,'-I','-B',vocal_mp4.__file__,str(mp4_root/'request.json')],mp4_root,1_073_741_824)
        value=finish_file_child(process,mp4_root/'response.json');assert value['ok'],value;checks.append('mp4_h264_aac_decode_frame_timestamps')
        # Actual decoder and FaceMesh run. Black frames intentionally have no face.
        video=work/'synthetic.mp4'
        with av.open(str(video),'w') as container:
            stream=container.add_stream('libx264',rate=30);stream.width=320;stream.height=240;stream.pix_fmt='yuv420p'
            for i in range(5):
                frame=av.VideoFrame.from_ndarray(np.zeros((240,320,3),dtype='uint8'),format='rgb24');frame.pts=i
                for packet in stream.encode(frame):container.mux(packet)
            for packet in stream.encode():container.mux(packet)
        from ptb_worker.m05_executor import run_child
        lip=work/'lip';lip.mkdir();(lip/'input.mp4').write_bytes(video.read_bytes());e={}
        response=run_child(lip,dict(input='input.mp4',config=dict(filter_enabled=False,cutoff_hz=10,animation='none')),lambda:False,lambda _:None,e)
        assert response['frames']==5 and e['group_cleaned'];processes.append(dict(entry='lip',**e));checks.append('lip_actual_facemesh_no_face_missing_values')
        require_identity(runtime_hashes(root),identity,'Runtime inputs changed during verification')
        report['success']=True;save();print(json.dumps(dict(success=True,checks=checks)),flush=True);return 0
    except BaseException as error:
        import traceback
        report.update(error=str(error),traceback=traceback.format_exc());save();raise

if __name__=='__main__':raise SystemExit(main(*sys.argv[1:]))
