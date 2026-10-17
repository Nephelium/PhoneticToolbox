"""Explicit, matched three-way export. Existing website WAVs are immutable inputs."""
from pathlib import Path
import argparse, hashlib, io, json, sys
from website_audio_baseline import SITE, ROOT, POLICY, SETTINGS, verify as verify_basic

LOCK = 'assets/spectrogram-examples/three-way-baseline.json'
REPAIR_SHA = 'a60a3587139ffdc79b69978615b2079c66a43282cb2b77ab99d276717b433ab2'
VOCODER_SHA = 'e2a93f92971e1b6750d8f9b701121597cd271e473e6c189c8072be7080047735'

def sha(data): return hashlib.sha256(data).hexdigest()
def read(path): return json.loads(path.read_text('utf-8'))
def write(path, obj): path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', 'utf-8')

def verify(site=SITE, *, expected_settings=None):
    settings = SETTINGS if expected_settings is None else expected_settings
    if read(site/'assets/spectrogram-examples/comparison.json').get('comparison_schema') == 'ptb-four-way/1':
        assert settings == SETTINGS, 'Four-way comparison must use current frozen settings'
        from build_website_four_comparison import verify as verify_four
        return verify_four(site)
    basic = verify_basic(site, expected_settings=settings)
    lock = read(site/LOCK)
    rows = {r['file']:r for r in read(site/'assets/audio/provenance.json')['files']}
    comparison = read(site/'assets/spectrogram-examples/comparison.json')
    pairs = comparison['pairs']
    if comparison.get('comparison_schema') == 'ptb-matched-three-way/2':
        assert comparison['source_report_sha256'] == sha(json.dumps(lock,sort_keys=True).encode())
    assert lock['settings'] == dict(settings, momentum=.99)
    assert lock['repair']['sha256'] == REPAIR_SHA and lock['repair']['step'] == 6000
    assert lock['vocoder']['sha256'] == VOCODER_SHA
    for name, digest in lock['preserved_audio'].items():
        assert sha((site/'assets/audio'/name).read_bytes()) == digest, name
    for pair in pairs:
        case = lock['cases'][pair['id']]
        assert case['input_sha256'] == basic['cases'][pair['id']]['input_sha256']
        assert pair['accelerated_settings'] == dict(pair['before_defaults'], momentum=.99)
        assert 'parameter_prediction' not in pair and pair['applied_window_ms'] == 10
        for kind in ['accelerated','after']:
            row = rows[pair[kind]]
            assert sha((site/'assets/audio'/pair[kind]).read_bytes()) == pair[kind+'_sha256'] == row['sha256'] == case[kind]['sha256']
            assert row['sample_rate'] == 24000 and abs(row['duration']-pair['seconds']) < 1/24000
            params = row['parameters']
            assert params['gl'] == dict(settings, momentum=.99)
            assert params['pulse_initialization'] is False and params['parameter_prediction'] is False
            assert params['input_sha256'] == case['input_sha256']
        neural = rows[pair['after']]['parameters']
        assert neural['vocoder_input_file'] == pair['accelerated']
        assert neural['vocoder_input_sha256'] == pair['accelerated_sha256']
        assert neural['repair_model_sha256'] == REPAIR_SHA and neural['repair_model_step'] == 6000
        assert neural['vocoder_model_sha256'] == VOCODER_SHA
        levels = [rows[pair[k]]['rms']*pair['playback_volumes'][pair[k][:-4]] for k in ['before','accelerated','after']]
        assert max(levels)-min(levels)<1e-10
        assert all(0<v<=1 for v in pair['playback_volumes'].values())
    return lock

def generate(args):
    import numpy as np
    from PIL import Image
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    import torch
    migration = args.migrate_to_32
    if migration:
        # Only this explicit, user-requested 256 -> 32 migration accepts old settings.
        assert SETTINGS['iterations'] == 32
        old_settings = dict(SETTINGS, iterations=256)
        basic = verify_basic(expected_settings=old_settings)
        verify(expected_settings=old_settings)
        previous_lock_sha = sha((SITE/LOCK).read_bytes())
    else:
        assert not (SITE/LOCK).exists(), 'Three-way exports are frozen; check them instead of regenerating'
        basic = verify_basic()
    assert sha(args.repair.read_bytes()) == REPAIR_SHA
    assert sha((args.vocoder/'bigvgan_generator.pt').read_bytes()) == VOCODER_SHA
    assert torch.load(args.repair, map_location='cpu', weights_only=True)['step'] == 6000
    assert torch.cuda.is_available(), 'CUDA required; no silent CPU fallback'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sys.path.insert(0, str(args.research_scripts.resolve()))
    sys.path.insert(0, str(ROOT/'backend/src/ptb_worker'))
    from aoki_phase_reconstruction import reconstruct
    from compare_five_pulse_vocoders import calibrated_spectrum
    from diagnose_aoki_image import decode
    from m09_pipeline_child import repair
    from vocoder_child import synthesize
    # Same implementation, calibrated grid and RNG as the frozen original GL.
    for name,digest in basic['implementation'].items():
        assert sha((args.research_scripts/name).read_bytes()) == digest, name
    comparison = read(SITE/'assets/spectrogram-examples/comparison.json')
    provenance = read(SITE/'assets/audio/provenance.json')
    rows = {r['file']:r for r in provenance['files']}
    preserved = {name:sha((SITE/'assets/audio'/name).read_bytes()) for name in rows}
    iterations = SETTINGS['iterations']
    new_basic = dict(basic, schema='ptb-basic-gl-baseline/2', settings=dict(SETTINGS), cases={},
                     previous_baseline=basic, authorization='2026-10-16 user explicitly requested 32 iterations instead of 256') if migration else basic
    lock = dict(schema='ptb-three-way-comparison/1', established_on='2026-10-16',
                settings=dict(SETTINGS,momentum=.99), preserved_audio=preserved,
                repair=dict(sha256=REPAIR_SHA,step=6000),vocoder=dict(sha256=VOCODER_SHA),
                device=torch.cuda.get_device_name(0),implementation=dict(basic['implementation']),cases={})
    if migration:
        lock['migration'] = dict(from_iterations=256,to_iterations=32,previous_lock_sha256=previous_lock_sha,
                                 authorization='User requested 32-iteration matched three-way comparison; previous WAVs retained unchanged')
    for name in ['m09_pipeline_child.py','vocoder_child.py','m09_neural_models.py']:
        lock['implementation'][name] = sha((ROOT/'backend/src/ptb_worker'/name).read_bytes())
    pending = []
    rms = lambda x:float(np.sqrt(np.mean(np.asarray(x,dtype=np.float64)**2)))
    def encode(wave, count, *, normalize=True):
        assert np.isfinite(wave).all() and abs(len(wave)-count)<=2 and rms(wave)>1e-7
        wave = np.pad(wave,(0,max(0,count-len(wave))),mode='edge')[:count]
        gain = min(.08/rms(wave),.95/max(float(np.abs(wave).max()),1e-12)) if normalize else 1.
        buf=io.BytesIO();wavfile.write(buf,24000,np.int16(np.clip(wave*gain,-1,1)*32767))
        payload=buf.getvalue();sr,pcm=wavfile.read(io.BytesIO(payload));assert sr==24000 and len(pcm)==count and pcm.ndim==1
        values=pcm.astype(np.float64)/32768
        return payload,values,dict(sample_rate=sr,duration=count/sr,peak=float(np.abs(values).max()),rms=rms(values),
                      waveform=[round(float(np.abs(c).max()),4) for c in np.array_split(values,72)],presentation_gain=gain)
    for pair in comparison['pairs']:
        case=basic['cases'][pair['id']]
        gray=np.asarray(Image.open(SITE/case['input_file']).convert('L'))
        magnitude,_,length=calibrated_spectrum(decode(gray,60),pair['seconds'],*pair['frequency_range'],hop=24)
        tensor=torch.from_numpy(magnitude[None]).to('cuda')
        count=round(pair['seconds']*24000)
        if migration:
            # Keep the old unaccelerated algorithm and presentation normalization.
            original,original_curve=reconstruct(tensor,160,24,length,iterations=iterations,momentum=0.,sources=None,seed=20261015)
            original_raw=original[iterations][0]
            original_gain=min(.08/rms(original_raw),.95/float(np.abs(original_raw).max()))
            original_payload,_,original_stats=encode(resample_poly(original_raw*original_gain,3,2).astype(np.float32),count,normalize=False)
            original_name='spectrum-before-32i-'+pair['id']+'.wav'
            assert original_name not in rows and not (SITE/'assets/audio'/original_name).exists()
            original_params=json.loads(json.dumps(rows[pair['before']]['parameters']))
            original_params['gl']['iterations']=iterations
            original_params.update(source_output_sha256=sha(original_payload),presentation_gain=original_gain)
            original_stats.pop('presentation_gain')
            original_row=dict(rows[pair['before']],file=original_name,sha256=sha(original_payload),
                              description=pair['id']+' · 原始 GL，32次，10 ms / 60 dB，无动量加速',parameters=original_params,**original_stats)
            provenance['files'].append(original_row);rows[original_name]=original_row
            pending.append((SITE/'assets/audio'/original_name,original_payload))
            pair['comparison_256_history']={k:pair[k] for k in ['before','before_sha256','accelerated','accelerated_sha256','after','after_sha256','before_defaults','accelerated_settings']}
            pair.update(before=original_name,before_sha256=sha(original_payload),before_defaults=dict(pair['before_defaults'],iterations=iterations))
            new_basic['cases'][pair['id']]=dict(case,file=original_name,sha256=sha(original_payload),
                                               previous_sha256=case['sha256'],convergence=original_curve)
        waves,curve=reconstruct(tensor,160,24,length,iterations=iterations,momentum=.99,sources=None,seed=20261015)
        raw=waves[iterations][0]
        fast,fast_pcm,fast_meta=encode(resample_poly(raw,3,2),count)
        suffix=('32i-' if migration else '')+pair['id']+'.wav'
        fast_name='spectrum-fast-'+suffix
        neural_name='spectrum-vocoder-comparison-'+suffix
        config=dict(freq_start=pair['frequency_range'][0],freq_end=pair['frequency_range'][1])
        # Decode the actual delivered PCM. The third route consumes these exact samples.
        neural,neural_meta=synthesize(fast_pcm,24000,args.vocoder,
                 feature_transform=lambda features,h:repair(features,config,h,checkpoint=args.repair),device='cuda')
        assert neural_meta['inference_device']=='cuda'
        neural_payload,_,neural_stats=encode(neural,count)
        common=dict(gl=dict(SETTINGS,momentum=.99),input_sha256=case['input_sha256'],
                    pulse_initialization=False,parameter_prediction=False,
                    magnitude_sha256=sha(magnitude.tobytes()),source=rows[pair['before']]['parameters']['source'])
        for name,payload,stats,params,desc in [
            (fast_name,fast,fast_meta,dict(common,route='动量加速 Griffin–Lim',mel_repair=False,vocoder=False,convergence=curve),'动量加速 GL，10 ms / 60 dB，动量0.99'),
            (neural_name,neural_payload,neural_stats,dict(common,route='同一动量 GL → Mel 修复 → BigVGAN',mel_repair=True,vocoder=True,
             vocoder_input_file=fast_name,vocoder_input_sha256=sha(fast),repair_model_sha256=REPAIR_SHA,repair_model_step=6000,
             vocoder_model_sha256=VOCODER_SHA,inference=neural_meta),'同一动量 GL 接 Mel 修复与 BigVGAN')]:
            destination=SITE/'assets/audio'/name
            assert not destination.exists() and name not in rows, 'Refusing to replace existing audio: '+name
            row=dict(rows[pair['before']],file=name,sha256=sha(payload),description=pair['id']+' · '+desc,parameters=params,**stats)
            params['presentation_gain']=row.pop('presentation_gain')
            provenance['files'].append(row);rows[name]=row;pending.append((destination,payload))
        historical={k:pair[k] for k in ['after','after_sha256','parameter_prediction','applied_window_ms','parameter_model_sha256'] if k in pair}
        if not migration:pair['previous_comparison']=historical
        for k in ['parameter_prediction','parameter_model_sha256']:pair.pop(k,None)
        pair.update(accelerated=fast_name,accelerated_sha256=sha(fast),after=neural_name,after_sha256=sha(neural_payload),
                    applied_window_ms=10,accelerated_settings=dict(pair['before_defaults'],momentum=.99))
        target=min(rows[pair[k]]['rms'] for k in ['before','accelerated','after'])
        pair['playback_volumes']={pair[k][:-4]:target/rows[pair[k]]['rms'] for k in ['before','accelerated','after']}
        lock['cases'][pair['id']]=dict(input_sha256=case['input_sha256'],magnitude_sha256=common['magnitude_sha256'],
                                     accelerated=dict(file=fast_name,sha256=sha(fast)),after=dict(file=neural_name,sha256=sha(neural_payload)))
        print('THREE_WAY '+pair['id'],flush=True)
    assert len(pending)==(18 if migration else 12)
    assert all(sha((SITE/'assets/audio'/n).read_bytes())==digest for n,digest in preserved.items())
    for path,payload in pending:path.write_bytes(payload)
    comparison.update(processing=f'同图同参数三路：固定10ms Hann/60dB、随机相位、{iterations}次；原始GL动量0，动量GL为0.99，第三路直接取第二路PCM接6000步80带NAFNet与BigVGAN。均不使用Aoki初始化或参数预测。播放只做等RMS衰减。',comparison_schema='ptb-matched-three-way/2')
    comparison['previous_source_report_sha256']=comparison.get('source_report_sha256')
    comparison['source_report_sha256']=sha(json.dumps(lock,sort_keys=True).encode())
    if migration:comparison['before_source_report_sha256']=sha(json.dumps(new_basic,sort_keys=True).encode())
    write(SITE/'assets/audio/provenance.json',provenance)
    write(SITE/'assets/spectrogram-examples/comparison.json',comparison)
    write(SITE/LOCK,lock)
    if migration:write(SITE/POLICY,new_basic)
    verify();print(json.dumps(dict(success=True,new_audio=len(pending),iterations=iterations,preserved_audio=len(preserved),device=lock['device'])))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--generate',action='store_true');p.add_argument('--research-scripts',type=Path)
    p.add_argument('--repair',type=Path);p.add_argument('--vocoder',type=Path);p.add_argument('--site',type=Path,default=SITE)
    p.add_argument('--migrate-to-32',action='store_true',help='Explicit authorized migration from locked 256-iteration tracks; new filenames, old WAVs unchanged')
    a=p.parse_args()
    if a.generate or a.migrate_to_32:
        assert a.research_scripts and a.repair and a.vocoder and a.site==SITE
        generate(a)
    else:
        verify(a.site);print(json.dumps(dict(success=True,matched_cases=6,tracks=18,original_controls_unchanged=True)))
