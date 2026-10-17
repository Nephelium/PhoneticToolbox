"""Explicit four-route export: frozen controls, predicted GL, then its actual PCM to vocoder."""
from pathlib import Path
import argparse, hashlib, io, json, math, sys
from website_audio_baseline import SITE, ROOT, SETTINGS, verify as verify_basic
from build_website_gl_comparison import REPAIR_SHA, VOCODER_SHA

LOCK = 'assets/spectrogram-examples/four-way-baseline.json'
PARAMETER_SHA = '7ac61f075259099e0c58e743818b2b61a1ec42d5a125eab6ef9fc9216182a86e'
KINDS = ('before', 'accelerated', 'predicted', 'after')

def sha(data): return hashlib.sha256(data).hexdigest()
def read(path): return json.loads(path.read_text('utf-8'))
def write(path, obj): path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', 'utf-8')

def verify(site=SITE):
    basic = verify_basic(site)
    comparison = read(site/'assets/spectrogram-examples/comparison.json')
    assert comparison['comparison_schema'] == 'ptb-four-way/1'
    lock = read(site/LOCK)
    assert comparison['source_report_sha256'] == sha(json.dumps(lock,sort_keys=True).encode())
    assert lock['settings'] == dict(SETTINGS,momentum=.99)
    assert lock['parameter_model']['sha256'] == PARAMETER_SHA
    assert lock['repair'] == dict(sha256=REPAIR_SHA,step=6000)
    assert lock['vocoder']['sha256'] == VOCODER_SHA
    assert sha((site/'assets/spectrogram-examples/three-way-baseline.json').read_bytes()) == lock['previous_three_way_lock_sha256']
    rows = {r['file']:r for r in read(site/'assets/audio/provenance.json')['files']}
    for name,digest in lock['preserved_audio'].items():
        assert sha((site/'assets/audio'/name).read_bytes()) == digest == rows[name]['sha256'], name
    assert len(comparison['pairs']) == len(lock['cases']) == 6
    for pair in comparison['pairs']:
        case = lock['cases'][pair['id']]
        assert case['input_sha256'] == basic['cases'][pair['id']]['input_sha256'] == pair['input_sha256']
        prediction = pair['parameter_prediction']
        assert prediction == case['parameter_prediction']
        assert prediction['checkpoint_sha256'] == PARAMETER_SHA and prediction['image_sha256'] == pair['input_sha256']
        assert prediction['applied_to_reconstruction'] is True
        assert pair['applied_window_ms'] == math.floor(prediction['window_ms']+.5) == prediction['applied_window_ms']
        expected = dict(SETTINGS,momentum=.99,window_ms=pair['applied_window_ms'],dynamic_range_db=prediction['dynamic_range_db'])
        assert pair['predicted_settings'] == expected
        assert pair['accelerated_settings'] == dict(pair['before_defaults'],momentum=.99)
        for kind in KINDS:
            name = pair[kind];row = rows[name];params = row['parameters']
            assert sha((site/'assets/audio'/name).read_bytes()) == row['sha256'] == pair[kind+'_sha256'] == case[kind]['sha256']
            assert name == case[kind]['file'] and row['sample_rate']==24000
            assert abs(row['duration']-pair['seconds']) < 1/24000
            assert params['pulse_initialization'] is False
            assert params['gl']['iterations']==32 and params['gl']['seed']==20261015 and params['gl']['hop']==24
            assert params['gl']['momentum']==(0 if kind=='before' else .99)
            if kind=='accelerated':
                assert params['gl']==dict(SETTINGS,momentum=.99) and params['parameter_prediction'] is False
            if kind in ('predicted','after'):
                assert params['gl']==expected and params['parameter_prediction']==prediction
                assert params['parameter_model_sha256']==PARAMETER_SHA
                assert params['input_sha256']==case['input_sha256']
                assert params['magnitude_sha256']==case['predicted_magnitude_sha256']
                assert params['mel_repair'] is (kind=='after') and params['vocoder'] is (kind=='after')
        neural = rows[pair['after']]['parameters']
        assert neural['vocoder_input_file']==pair['predicted'] and neural['vocoder_input_sha256']==pair['predicted_sha256']
        assert neural['repair_model_sha256']==REPAIR_SHA and neural['repair_model_step']==6000
        assert neural['vocoder_model_sha256']==VOCODER_SHA
        assert set(pair['playback_volumes']) == {pair[k][:-4] for k in KINDS}
        levels=[rows[pair[k]]['rms']*pair['playback_volumes'][pair[k][:-4]] for k in KINDS]
        assert max(levels)-min(levels)<1e-10 and all(0<v<=1 for v in pair['playback_volumes'].values())
    return lock

def generate(args):
    assert not (SITE/LOCK).exists(), 'Four-way exports are frozen; verify instead of regenerating'
    import numpy as np
    from PIL import Image
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    import torch
    from build_website_gl_comparison import verify as verify_previous
    previous = verify_previous()
    assert SETTINGS['iterations']==32
    for path,digest in [(args.parameter,PARAMETER_SHA),(args.repair,REPAIR_SHA),(args.vocoder/'bigvgan_generator.pt',VOCODER_SHA)]:
        assert sha(path.read_bytes())==digest,str(path)
    assert torch.load(args.repair,map_location='cpu',weights_only=True)['step']==6000
    assert torch.cuda.is_available(), 'CUDA required; no implicit CPU fallback'
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    sys.path.insert(0,str(args.research_scripts.resolve()))
    sys.path.insert(0,str(ROOT/'backend/src/ptb_worker'))
    from predict_continuous_parameters import predict
    from aoki_phase_reconstruction import reconstruct
    from compare_five_pulse_vocoders import calibrated_spectrum
    from diagnose_aoki_image import decode
    from m09_pipeline_child import repair
    from vocoder_child import synthesize
    basic=verify_basic();comparison=read(SITE/'assets/spectrogram-examples/comparison.json')
    provenance=read(SITE/'assets/audio/provenance.json');rows={r['file']:r for r in provenance['files']}
    preserved={n:sha((SITE/'assets/audio'/n).read_bytes()) for n in rows}
    lock=dict(schema='ptb-four-way-baseline/1',settings=dict(SETTINGS,momentum=.99),
              parameter_model=dict(sha256=PARAMETER_SHA,name='parameter-regressor-stream-v5'),
              repair=dict(sha256=REPAIR_SHA,step=6000),vocoder=dict(sha256=VOCODER_SHA),
              previous_three_way_lock_sha256=sha((SITE/'assets/spectrogram-examples/three-way-baseline.json').read_bytes()),
              preserved_audio=preserved,device=torch.cuda.get_device_name(0),cases={},
              authorization='User requested four routes, predicted window/dB, 32 iterations, no pulse initialization, same predicted GL PCM into repair/vocoder',implementation={})
    for name in ['predict_continuous_parameters.py','train_online_parameters.py','aoki_phase_reconstruction.py','compare_five_pulse_vocoders.py','diagnose_aoki_image.py']:
        lock['implementation'][name]=sha((args.research_scripts/name).read_bytes())
    for name,digest in basic['implementation'].items():assert lock['implementation'][name]==digest
    for name in ['m09_pipeline_child.py','vocoder_child.py','m09_neural_models.py']:
        lock['implementation'][name]=sha((ROOT/'backend/src/ptb_worker'/name).read_bytes())
    rms=lambda a:float(np.sqrt(np.mean(np.asarray(a,dtype=np.float64)**2)))
    def encode(wave,count):
        assert np.isfinite(wave).all() and abs(len(wave)-count)<=2 and rms(wave)>1e-7
        wave=np.pad(wave,(0,max(0,count-len(wave))),mode='edge')[:count]
        gain=min(.08/rms(wave),.95/max(float(np.abs(wave).max()),1e-12))
        stream=io.BytesIO();wavfile.write(stream,24000,np.int16(np.clip(wave*gain,-1,1)*32767));payload=stream.getvalue()
        sr,pcm=wavfile.read(io.BytesIO(payload));values=pcm.astype(np.float64)/32768
        assert sr==24000 and len(pcm)==count and pcm.ndim==1 and np.isfinite(values).all()
        return payload,values,dict(sample_rate=sr,duration=count/sr,peak=float(np.abs(values).max()),rms=rms(values),
            waveform=[round(float(np.abs(c).max()),4) for c in np.array_split(values,72)],presentation_gain=gain)
    pending=[]
    for pair in comparison['pairs']:
        case=basic['cases'][pair['id']];source=SITE/case['input_file']
        prediction=predict(args.parameter,source,pair['seconds'],*pair['frequency_range'],axis='linear',device='cuda')
        historical=pair['previous_comparison']['parameter_prediction']
        for key in ('window_ms','dynamic_range_db'):assert abs(prediction[key]-historical[key])<1e-4,(pair['id'],key)
        window_ms=math.floor(prediction['window_ms']+.5);win=window_ms*16
        assert 24<win<=512
        prediction.update(applied_to_reconstruction=True,applied_window_ms=window_ms,window_rounding='nearest integer ms, half up')
        gray=np.asarray(Image.open(source).convert('L'))
        magnitude,_,length=calibrated_spectrum(decode(gray,prediction['dynamic_range_db']),pair['seconds'],*pair['frequency_range'],hop=24)
        waves,curve=reconstruct(torch.from_numpy(magnitude[None]).to('cuda'),win,24,length,iterations=32,momentum=.99,sources=None,seed=20261015)
        count=round(pair['seconds']*24000);gl_payload,gl_pcm,gl_stats=encode(resample_poly(waves[32][0],3,2),count)
        gl_name='spectrum-predicted-fast-32i-'+pair['id']+'.wav'
        after_name='spectrum-predicted-vocoder-32i-'+pair['id']+'.wav'
        neural,inference=synthesize(gl_pcm,24000,args.vocoder,feature_transform=lambda features,h:repair(features,dict(freq_start=pair['frequency_range'][0],freq_end=pair['frequency_range'][1]),h,checkpoint=args.repair),device='cuda')
        assert inference['inference_device']=='cuda'
        neural_payload,_,neural_stats=encode(neural,count)
        settings=dict(SETTINGS,momentum=.99,window_ms=window_ms,dynamic_range_db=prediction['dynamic_range_db'])
        common=dict(gl=settings,input_sha256=case['input_sha256'],parameter_prediction=prediction,parameter_model_sha256=PARAMETER_SHA,
                    pulse_initialization=False,magnitude_sha256=sha(magnitude.tobytes()),source=rows[pair['before']]['parameters']['source'])
        for name,payload,stats,params,desc in [
            (gl_name,gl_payload,gl_stats,dict(common,route='预测窗宽/dB → 动量 Griffin–Lim',mel_repair=False,vocoder=False,convergence=curve),'预测参数 + 动量 GL'),
            (after_name,neural_payload,neural_stats,dict(common,route='同一预测参数动量 GL PCM → NAFNet → BigVGAN',mel_repair=True,vocoder=True,
                vocoder_input_file=gl_name,vocoder_input_sha256=sha(gl_payload),repair_model_sha256=REPAIR_SHA,repair_model_step=6000,vocoder_model_sha256=VOCODER_SHA,inference=inference),'预测参数 + 动量 GL + NAFNet + BigVGAN')]:
            destination=SITE/'assets/audio'/name
            assert name not in rows and not destination.exists(),'Refusing to replace existing audio: '+name
            params['presentation_gain']=stats.pop('presentation_gain')
            row=dict(rows[pair['before']],file=name,sha256=sha(payload),parameters=params,description=pair['id']+' · '+desc,**stats)
            provenance['files'].append(row);rows[name]=row;pending.append((destination,payload))
        pair['three_way_32_history']={k:pair[k] for k in ('after','after_sha256','playback_volumes','applied_window_ms')}
        pair.update(predicted=gl_name,predicted_sha256=sha(gl_payload),after=after_name,after_sha256=sha(neural_payload),
                    parameter_prediction=prediction,parameter_model_sha256=PARAMETER_SHA,applied_window_ms=window_ms,predicted_settings=settings)
        target=min(rows[pair[k]]['rms'] for k in KINDS)
        pair['playback_volumes']={pair[k][:-4]:target/rows[pair[k]]['rms'] for k in KINDS}
        lock['cases'][pair['id']]=dict(input_sha256=case['input_sha256'],parameter_prediction=prediction,predicted_magnitude_sha256=common['magnitude_sha256'],
                                    **{k:dict(file=pair[k],sha256=pair[k+'_sha256']) for k in KINDS})
        print('FOUR_WAY '+pair['id']+' '+str(window_ms)+' ms / '+str(round(prediction['dynamic_range_db'],3))+' dB',flush=True)
    assert len(pending)==12 and all(sha((SITE/'assets/audio'/n).read_bytes())==d for n,d in preserved.items())
    for path,payload in pending:path.write_bytes(payload)
    comparison.update(comparison_schema='ptb-four-way/1',previous_three_way_source_report_sha256=comparison['source_report_sha256'],
                      source_report_sha256=sha(json.dumps(lock,sort_keys=True).encode()),
                      processing='四路32次：固定10ms/60dB原始GL、固定参数动量GL、预测窗宽/dB的动量GL、同一预测GL的PCM接NAFNet与BigVGAN。随机相位/seed/帧移固定，无Aoki，等RMS播放衰减。')
    for path,obj in [(SITE/'assets/audio/provenance.json',provenance),(SITE/'assets/spectrogram-examples/comparison.json',comparison),(SITE/LOCK,lock)]:write(path,obj)
    verify();print(json.dumps(dict(success=True,routes=4,cases=6,new_audio=12,preserved_audio=len(preserved),device=lock['device'])))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--generate',action='store_true');p.add_argument('--site',type=Path,default=SITE)
    for flag in ['research-scripts','parameter','repair','vocoder']:p.add_argument('--'+flag,type=Path)
    args=p.parse_args()
    if args.generate:
        assert args.site==SITE and all(getattr(args,k) for k in ['research_scripts','parameter','repair','vocoder'])
        generate(args)
    else:verify(args.site);print(json.dumps(dict(success=True,cases=6,tracks=24)))
