"""Public, reproducible listening examples using the application's scientific core.

User-selected real recordings for M07/M08; native M10 model output for the articulatory demo. Originals are read-only.
"""
from pathlib import Path
import argparse,hashlib,json,sys,shutil
from dataclasses import asdict
import numpy as np
from scipy.io import wavfile
ROOT=Path(__file__).resolve().parents[1]
for part in ('packages/phonetic_core/src','backend/src'):sys.path.insert(0,str(ROOT/part))
from phonetic_core.vocal_tract.engine import Engine
from phonetic_core.manipulation import m07_api
from phonetic_core.manipulation.m07_models import PhonationAnalysisConfig,PhonationGenerationConfig,F0Backend,ContinuumType
from phonetic_core.manipulation.m08_synthesis import synthesize_from_pitch
from ptb_worker.m07_science import estimator
from ptb_worker.native.reaper import Reaper
from ptb_worker.io.scratch import Scratch
import parselmouth

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phonation-only',action='store_true',help='Rebuild only the two phonation endpoints and seven continuum steps.')
    parser.add_argument('--phonation-export',type=Path,help='Import a user-approved M07 complete export without re-synthesis.')
    args=parser.parse_args()
    out=ROOT/'temporary-site/assets/audio';out.mkdir(exist_ok=True)
    previous=json.loads((out/'provenance.json').read_text('utf-8'))
    if previous.get('phonation_delivery')=='user_export' and not args.phonation_export:
        raise ValueError('Approved manual audio is active; refresh it with --phonation-export, without automatic re-synthesis.')
    rows=[r for r in previous['files'] if not r['file'].startswith('phonation-')] if args.phonation_only or args.phonation_export else [r for r in previous['files'] if r['file'].startswith('spectrum-')]
    retained={r['file']:hashlib.sha256((out/r['file']).read_bytes()).hexdigest() for r in rows}
    def save(name,a,rate,description,parameters,f0_port=None,original_path=None):
        a=np.asarray(a,dtype=float).reshape(-1)
        assert np.isfinite(a).all() and np.std(a)>.00001
        # Same peak ceiling for comfortable short demonstrations; not loudness research.
        if original_path is None:a=a/max(1.,np.max(np.abs(a))/.65)
        path=out/(name+'.wav')
        if original_path is not None:
            shutil.copyfile(original_path,path)
        else:
            wavfile.write(path,int(rate),np.int16(np.clip(a,-1,1)*32767))
        if f0_port is not None:
            delivered=parselmouth.Sound(str(path));pitch=f0_port(delivered.values[0],delivered.sampling_frequency)
        else:
            pitch=parselmouth.Sound(a,rate).to_pitch(pitch_floor=60,pitch_ceiling=400).selected_array['frequency']
        voiced=pitch[pitch>0]
        rows.append(dict(file=path.name,description=description,parameters=parameters,sample_rate=int(rate),duration=round(len(a)/rate,4),peak=float(np.max(np.abs(a))),rms=float(np.sqrt(np.mean(a*a))),median_f0=float(np.median(voiced)) if len(voiced) else None,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),waveform=[round(float(np.max(np.abs(c))),4) for c in np.array_split(a,72)]))
        return a
    inputs=Path(r'C:\Users\13680\Desktop\project\音频数据\test')
    if args.phonation_export:
        export=args.phonation_export.resolve();metadata_path=export/'m07.ptb.json'
        metadata_raw=metadata_path.read_bytes();metadata=json.loads(metadata_raw)
        assert metadata['schema_version']=='m07/1' and metadata['status']=='complete' and metadata['action']=='generate'
        count=metadata['generation']['step_count'];assert isinstance(count,int) and 2<=count<=50
        assert metadata['continuum_type']==3 and metadata['reverse_direction'] is False
        assert metadata['analysis']['f0_backend']=='parselmouth'
        listed={r['name']:r for r in metadata['outputs']};export_before={}
        allowed={f'step{i:02d}.wav' for i in range(1,count+1)}|{'combined_steps.wav','edited_f0.csv'}
        assert set(listed)==allowed
        for name,row in listed.items():
            path=export/name;assert path.resolve().is_relative_to(export)
            raw=path.read_bytes();assert len(raw)==row['size_bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
            export_before[name]=row['sha256']
        selected=[inputs/'dl_m02_0558_ma.wav',inputs/'dl_m05_0558_ma.wav']
        originals=[]
        for role,path in zip(('source','target'),selected):
            digest=hashlib.sha256(path.read_bytes()).hexdigest();assert digest==metadata['inputs'][role]['sha256']
            originals.append(dict(file=path.name,sha256=digest,source_label='SCSC 录音',origin='SCSC'))
        config=metadata['analysis']
        def measure(audio,rate):
            return parselmouth.Sound(audio,float(rate)).to_pitch(time_step=config['f0_frame_interval_ms']/1000,pitch_floor=config['min_f0_hz'],pitch_ceiling=config['max_f0_hz']).selected_array['frequency']
        measurement=dict(backend='parselmouth',min_f0_hz=config['min_f0_hz'],max_f0_hz=config['max_f0_hz'],frame_interval_ms=config['f0_frame_interval_ms'],measured='Delivered WAV PCM')
        for i in range(1,count+1):
            path=export/f'step{i:02d}.wav';sound=parselmouth.Sound(str(path))
            parameters=dict(step=i,steps=count,kind='f0_and_phonation',delivery='user_export',source_output=path.name,source_output_sha256=listed[path.name]['sha256'],export_metadata_sha256=hashlib.sha256(metadata_raw).hexdigest(),inputs=originals,engine='M07 application export',analysis=config,generation=metadata['generation'],alignment=metadata['alignment'],f0_measurement=measurement)
            save(f'phonation-{i}',sound.values[0],sound.sampling_frequency,f'应用导出连续统，第 {i} 步',parameters,f0_port=measure,original_path=path)
            assert hashlib.sha256((out/f'phonation-{i}.wav').read_bytes()).hexdigest()==listed[path.name]['sha256']
        for i,path in enumerate(selected,1):
            sound=parselmouth.Sound(str(path));save(f'phonation-endpoint-{i}',sound.values[0],sound.sampling_frequency,f'SCSC 原声 {i}',dict(input=originals[i-1],processing='Unchanged original WAV',f0_measurement=measurement),f0_port=measure,original_path=path)
            assert hashlib.sha256((out/f'phonation-endpoint-{i}.wav').read_bytes()).hexdigest()==originals[i-1]['sha256']
        assert metadata_raw==metadata_path.read_bytes()
        assert all(hashlib.sha256((export/name).read_bytes()).hexdigest()==digest for name,digest in export_before.items())
        assert all(hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256'] for path,r in zip(selected,originals))
        assert all(hashlib.sha256((out/name).read_bytes()).hexdigest()==digest for name,digest in retained.items())
        manifest=dict(scope='User-approved M07 application export copied unchanged; both input recordings are SCSC. Other eleven audio examples retained.',phonation_delivery='user_export',files=rows)
        (out/'provenance.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        audit=ROOT/'output/validation/website/manual-phonation-m02-m05-20261013';audit.mkdir(exist_ok=True)
        report=dict(success=True,steps=count,analysis=config,generation=metadata['generation'],metadata_sha256=hashlib.sha256(metadata_raw).hexdigest(),inputs=originals,export_files_unchanged=export_before,outputs=[r for r in rows if r['file'].startswith('phonation-')],other_audio_unchanged=retained,no_resynthesis=True)
        (audit/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(dict(imported_steps=count,audio_count=len(rows),f0=[r['median_f0'] for r in report['outputs']]),ensure_ascii=False));return

    if not args.phonation_only:
        e=Engine(resource_dir=ROOT/'resources/vocal_tract/native')
        try:
            vowels={v:e.synthesize(e.presets[v],duration=1.2,f0=160) for v in ('a','i','u')}
            scale=.6/max(np.max(np.abs(a)) for a in vowels.values())
            for v,a in vowels.items():save('vowel-'+v,a*scale,e.sr,'生理参数合成 /'+v+'/',dict(preset=v,f0_hz=160,duration=1.2,engine='VocalTractLab / current PhoneticToolbox core'))
            chosen=[inputs/'dl_m02_0558_ma.wav']
            originals=[dict(file=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in chosen]
            source=parselmouth.Sound(str(chosen[0]));base=source.values[0];rate=source.sampling_frequency;duration=source.duration
            save('pitch-original',base,rate,'真实原声 /ma/',dict(input=originals[0],speed=1,processing='PCM16 export; no pitch or duration change'))
            track=source.to_pitch(time_step=.005,pitch_floor=50,pitch_ceiling=400)
            times=track.xs();f0=track.selected_array['frequency']
            for name,delta,speed in [('lower',-20.,1),('higher',20.,1),('slower',0.,.75),('faster',0.,1.25)]:
                modified=np.where(f0>0,f0+delta,0.)
                snd=synthesize_from_pitch(source,times,modified,0,duration,speed)
                save('pitch-'+name,snd.values[0],snd.sampling_frequency,'真实录音变速变调 '+name,dict(input=originals[0],f0_offset_hz=delta,speed=speed,engine='M08 Praat overlap-add'))
            assert all(hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'] for p,row in zip(chosen,originals))
        finally:e.close()
    chosen=[inputs/'dl_m02_0558_ma.wav',inputs/'webedit_m02_0135_ma.wav']
    originals=[dict(file=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in chosen]
    originals[0].update(source_label='SCSC 录音',origin='SCSC')
    originals[1].update(source_label='自有音频',origin='User-owned audio, not SCSC')
    cfg=PhonationAnalysisConfig(f0_backend=F0Backend.REAPER)
    endpoints=[parselmouth.Sound(str(p)) for p in chosen]
    audit=ROOT/'output/validation/website/phonation-reaper-m02-webedit-20261013';audit.mkdir(exist_ok=True)
    before={r['file']:hashlib.sha256((out/r['file']).read_bytes()).hexdigest() for r in rows}
    with Scratch(audit,64_000_000) as scratch:
        native=Reaper(ROOT/'phonetic_toolbox/core/acoustic/reaper.exe',scratch)
        f0_estimator=estimator(native)
        def measure(a,rate):return f0_estimator(a,int(rate),cfg)
        analyses=[m07_api.analyze(s.values[0],s.sampling_frequency,cfg,f0_estimator) for s in endpoints]
        generation=PhonationGenerationConfig(step_count=7)
        result=m07_api.generate(*analyses,ContinuumType.F0_AND_PHONATION,generation)
        measurement=dict(backend='REAPER',binary_sha256=native.sha256,min_f0_hz=cfg.min_f0_hz,max_f0_hz=cfg.max_f0_hz,frame_interval_ms=cfg.f0_frame_interval_ms,measured='Delivered WAV PCM')
        for i,a in enumerate(result.audio_steps.T):
            save('phonation-'+str(i+1),a,result.sample_rate,'发声类型与基频连续统，第 '+str(i+1)+' 步',dict(step=i+1,steps=7,kind='f0_and_phonation',inputs=originals,engine='M07 LPC residual continuum',analysis=asdict(cfg),generation=asdict(generation),f0_measurement=measurement),f0_port=measure)
        for i,snd in enumerate(endpoints):
            save('phonation-endpoint-'+str(i+1),snd.values[0],snd.sampling_frequency,originals[i]['source_label']+'端点 '+str(i+1),dict(input=originals[i],processing='Unchanged original WAV',f0_measurement=measurement),f0_port=measure,original_path=chosen[i])
        assert all(hashlib.sha256((out/('phonation-endpoint-'+str(i+1)+'.wav')).read_bytes()).hexdigest()==r['sha256'] for i,r in enumerate(originals))
    verification=dict(success=True,inputs=originals,analysis=asdict(cfg),generation=asdict(generation),kind='f0_and_phonation',reaper_sha256=native.sha256,outputs=[r for r in rows if r['file'].startswith('phonation-')],other_audio_unchanged=before,native_scratch_removed=not scratch.root.exists())
    assert all(hashlib.sha256((out/name).read_bytes()).hexdigest()==digest for name,digest in before.items())
    (audit/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'] for p,row in zip(chosen,originals))
    assert all(hashlib.sha256((out/name).read_bytes()).hexdigest()==digest for name,digest in retained.items())
    manifest=dict(scope='User-authorized SCSC recording A and user-owned audio B for M07; the selected recording for M08. M10 sounds are actual articulatory-model outputs, explicitly labeled. Input identity retained by filename/hash; no private local paths.',files=rows)
    (out/'provenance.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n','utf-8')
    print(json.dumps(dict(count=len(rows),durations=[r['duration'] for r in rows],f0=[r.get('median_f0') for r in rows]),ensure_ascii=False))
if __name__=='__main__':main()
