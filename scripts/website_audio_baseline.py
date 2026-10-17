"""Frozen, unaccelerated website GL controls; this is separate from model demos."""
from pathlib import Path
import argparse, hashlib, importlib.util, io, json, sys

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / 'temporary-site'
POLICY = 'assets/spectrogram-examples/basic-gl-baseline.json'
SETTINGS = dict(window_ms=10, window='Hann', dynamic_range_db=60,
                phase_initialization='random', momentum=0, iterations=32,
                sampling_rate=16000, nfft=512, hop=24, seed=20261015)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(path.read_text('utf-8'))


def verify(site=SITE, *, expected_settings=None):
    settings = SETTINGS if expected_settings is None else expected_settings
    policy = read_json(site / POLICY)
    assert policy['settings'] == settings, 'Website baseline algorithm or parameters changed'
    rows = {r['file']: r for r in read_json(site/'assets/audio/provenance.json')['files']}
    pairs = read_json(site/'assets/spectrogram-examples/comparison.json')['pairs']
    assert len(pairs) == len(policy['cases']) == 6
    assert {p['id'] for p in pairs} == set(policy['cases'])
    for pair in pairs:
        locked = policy['cases'][pair['id']]
        assert pair['before'] == locked['file']
        assert sha((site/'assets/audio'/pair['before']).read_bytes()) == locked['sha256'] == pair['before_sha256'] == rows[pair['before']]['sha256'], pair['id']
        assert sha((site/locked['input_file']).read_bytes()) == locked['input_sha256'] == pair['input_sha256']
        params = rows[pair['before']]['parameters']
        for key in ['window_ms', 'window', 'dynamic_range_db']:
            assert params['assumptions'][key] == settings[key]
        for key in ['phase_initialization','momentum','iterations','sampling_rate','nfft','hop','seed']:
            assert params['gl'][key] == settings[key], (pair['id'],key)
        assert params['pulse_initialization'] is False and params['parameter_prediction'] is False
        assert params['mel_repair'] is False and params['vocoder'] is False
        assert pair['before_defaults']['momentum'] == 0
        assert pair['before_defaults']['phase_initialization'] == 'random'
        assert pair['before_defaults']['window_ms'] == 10 and pair['before_defaults']['dynamic_range_db'] == 60
        assert pair['before_defaults']['iterations'] == settings['iterations']
    return policy


def protect_pending(pending, site=SITE):
    """Reject a changed control before any importer writes its pending batch."""
    policy = verify(site)
    locked = {r['file']:r['sha256'] for r in policy['cases'].values()}
    three_way=site/'assets/spectrogram-examples/three-way-baseline.json'
    if three_way.is_file():
        comparison_lock=read_json(three_way)
        locked.update(comparison_lock['preserved_audio'])
        for case in comparison_lock['cases'].values():
            for kind in ['accelerated','after']:
                locked[case[kind]['file']]=case[kind]['sha256']
    four_way=site/'assets/spectrogram-examples/four-way-baseline.json'
    if four_way.is_file():
        four_lock=read_json(four_way)
        locked.update(four_lock['preserved_audio'])
        for case in four_lock['cases'].values():
            for kind in ['predicted','after']:
                locked[case[kind]['file']]=case[kind]['sha256']
    for destination, payload in pending:
        if destination.name in locked:
            assert sha(payload) == locked[destination.name], 'Frozen basic GL control cannot be replaced: '+destination.name


def generate(research_scripts):
    """One explicit baseline migration. Once frozen, routine regeneration is refused."""
    assert not (SITE/POLICY).exists(), 'Basic GL controls are already frozen; use --check, do not regenerate them'
    import numpy as np
    from PIL import Image
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    import torch
    assert torch.cuda.is_available(), 'CUDA is required; no implicit CPU fallback'
    research_scripts = research_scripts.resolve()
    sys.path.insert(0,str(research_scripts))
    from aoki_phase_reconstruction import reconstruct
    from compare_five_pulse_vocoders import calibrated_spectrum
    from diagnose_aoki_image import decode
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    comparison = read_json(SITE/'assets/spectrogram-examples/comparison.json')
    provenance = read_json(SITE/'assets/audio/provenance.json')
    rows = {r['file']:r for r in provenance['files']}
    after_before = {p['after']:sha((SITE/'assets/audio'/p['after']).read_bytes()) for p in comparison['pairs']}
    all_before = {r['file']:sha((SITE/'assets/audio'/r['file']).read_bytes()) for r in provenance['files']}
    policy = dict(schema='ptb-basic-gl-baseline/1', established_on='2026-10-16', settings=SETTINGS,
                  mapping='20log10 magnitude; -60 to 0 dB; gray >=252 zero; DC zero; linear calibrated time/frequency grid',
                  output=dict(sampling_rate=24000,format='PCM16',gain='presentation only: fixed RMS 0.08 target with 0.95 peak cap'),
                  forbidden=['Aoki/pulse initialization','momentum acceleration','parameter prediction','Mel repair','vocoder','denoise or filtering','automatic quality tuning or reselection'],
                  regeneration='Frozen files are reused byte for byte. A new user request is required for changes.',
                  implementation={name:sha((research_scripts/name).read_bytes()) for name in ['aoki_phase_reconstruction.py','compare_five_pulse_vocoders.py','diagnose_aoki_image.py']}, cases={})
    pending=[]
    with torch.inference_mode():
        for pair in comparison['pairs']:
            candidates=[SITE/'assets/spectrogram-examples'/(pair['id']+suffix) for suffix in ['-input.png','.png']]
            image=next(p for p in candidates if p.is_file() and sha(p.read_bytes())==pair['input_sha256'])
            gray=np.asarray(Image.open(image).convert('L'))
            magnitude,_,length=calibrated_spectrum(decode(gray,60),pair['seconds'],*pair['frequency_range'],hop=24)
            tensor=torch.from_numpy(magnitude[None]).to('cuda')
            waves,curve=reconstruct(tensor,160,24,length,iterations=SETTINGS['iterations'],momentum=0.,sources=None,seed=20261015)
            raw=waves[SETTINGS['iterations']][0];assert np.isfinite(raw).all() and np.std(raw)>1e-7
            rms=lambda v:float(np.sqrt(np.mean(np.asarray(v,dtype=np.float64)**2)))
            gain=min(.08/rms(raw),.95/float(np.abs(raw).max()))
            wave=resample_poly(raw*gain,3,2).astype(np.float32);count=round(pair['seconds']*24000)
            assert abs(len(wave)-count)<=2
            wave=np.pad(wave,(0,max(0,count-len(wave))),mode='edge')[:count]
            assert np.isfinite(wave).all() and float(np.abs(wave).max())<1
            stream=io.BytesIO();wavfile.write(stream,24000,np.int16(np.clip(wave,-1,1)*32767));payload=stream.getvalue()
            sr,pcm=wavfile.read(io.BytesIO(payload));assert sr==24000 and len(pcm)==count and pcm.ndim==1
            values=pcm.astype(np.float64)/32768;digest=sha(payload);name=pair['before']
            params=dict(route='基础 Griffin–Lim（随机相位、无动量加速）',
                        assumptions=dict(window_ms=10,window='Hann',dynamic_range_db=60,mapping='20log10 magnitude',preemphasis=0,white_threshold=252,no_filter=True),
                        gl={k:SETTINGS[k] for k in ['sampling_rate','nfft','hop','iterations','momentum','seed','phase_initialization']},
                        pulse_initialization=False,parameter_prediction=False,mel_repair=False,vocoder=False,
                        source=rows[name]['parameters']['source'],source_output_sha256=digest,delivery='frozen_basic_gl_control',presentation_gain=gain)
            row=rows[name];row.update(description=f'同图处理前：{pair["id"]} · 基础 GL，10 ms / 60 dB，无动量加速',parameters=params,sha256=digest,
                 sample_rate=sr,duration=count/sr,peak=float(np.abs(values).max()),rms=rms(values),waveform=[round(float(np.abs(c).max()),4) for c in np.array_split(values,72)])
            pair.update(before_sha256=digest,before_defaults=dict(SETTINGS,mapping='20log10 magnitude',preemphasis=0,no_filter=True,pulse_initialization=False,parameter_prediction=False,mel_repair=False,vocoder=False))
            after=rows[pair['after']];target=min(row['rms'],after['rms']);pair['playback_volumes']={name[:-4]:target/row['rms'],pair['after'][:-4]:target/after['rms']}
            policy['cases'][pair['id']]=dict(file=name,sha256=digest,input_file=image.relative_to(SITE).as_posix(),input_sha256=pair['input_sha256'],previous_sha256=all_before[name],convergence=curve)
            pending.append((SITE/'assets/audio'/name,payload));print('BASIC_GL '+pair['id'],flush=True)
    # Validate the full set before replacing any control. Other tracks remain untouched.
    assert len(pending)==6 and all(sha((SITE/'assets/audio'/name).read_bytes())==digest for name,digest in all_before.items())
    for path,payload in pending:path.write_bytes(payload)
    comparison['before_source_report_sha256']=sha(json.dumps(policy,sort_keys=True).encode())
    comparison['processing']=f"处理前冻结为10ms Hann/60dB、固定种子随机相位、{SETTINGS['iterations']}次基础GL，无脉冲与动量加速。播放仅衰减较响侧。"
    for path,obj in [(SITE/POLICY,policy),(SITE/'assets/audio/provenance.json',provenance),(SITE/'assets/spectrogram-examples/comparison.json',comparison)]:
        path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n','utf-8')
    verify();assert all(sha((SITE/'assets/audio'/name).read_bytes())==digest for name,digest in after_before.items())
    assert all(sha((SITE/'assets/audio'/name).read_bytes())==digest for name,digest in all_before.items() if name not in {p.name for p,_ in pending})
    print(json.dumps(dict(success=True,device=torch.cuda.get_device_name(0),controls=6,other_audio_unchanged=len(all_before)-6)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--generate',action='store_true');parser.add_argument('--research-scripts',type=Path);parser.add_argument('--site',type=Path,default=SITE);args=parser.parse_args()
    if args.generate:
        assert args.research_scripts and args.site==SITE
        generate(args.research_scripts)
    else:
        policy=verify(args.site)
        # Exercise the overwrite guard without writing any file.
        name=next(iter(policy['cases'].values()))['file']
        try:protect_pending([(args.site/'assets/audio'/name,b'not the frozen control')],args.site)
        except AssertionError as exc:assert 'cannot be replaced' in str(exc)
        else:raise AssertionError('Overwrite protection did not reject a changed control')
        print(json.dumps(dict(success=True,frozen_controls=6,changed_control_rejected=True)))
