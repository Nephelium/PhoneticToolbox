"""Offline image reconstruction protocol, frozen inside the optional component."""
import io
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))


def repair(features, config, model_config, checkpoint=None):
    import numpy as np
    import torch
    from librosa.filters import mel
    from m09_neural_models import VisibleMelNAFNet
    net=VisibleMelNAFNet(80).eval().to(features.device)
    net.load_state_dict(torch.load(checkpoint or ROOT/'models/repair80.pt',map_location='cpu',weights_only=True)['model'],strict=True)
    h=model_config
    fb=mel(sr=h.sampling_rate,n_fft=h.n_fft,n_mels=h.num_mels,fmin=h.fmin,fmax=h.fmax)
    freq=np.arange(h.n_fft//2+1)*h.sampling_rate/h.n_fft
    cov=(fb[:,(freq>=config['freq_start'])&(freq<=config['freq_end'])].sum(1)/fb.sum(1)).astype(np.float32)
    coverage=torch.from_numpy(cov[None]).to(features.device)
    initial=torch.where(coverage[:,:,None]>0,features,features.new_tensor(1e-5).log())
    return net(initial,features.new_zeros(1,6),coverage)


def enhance(audio,sr,config):
    from vocoder_child import synthesize
    return synthesize(audio,sr,ROOT/'model',
        feature_transform=lambda features,h:repair(features,config,h),device=config.get('inference_device','auto'))


def reconstruct(gray, config):
    from phonetic_core.spec2wav.adaptive import reconstruct_calibrated, auto_gain
    from vocoder_child import synthesize
    applied=dict(config)
    # Prediction is supplied by the application's built-in NumPy model.
    if config.get('auto_parameters',False):raise ValueError('invalid_spectrogram_config')
    keys=('time_start','time_end','freq_start','freq_end','min_db','max_db','win_length_ms','n_iter','target_sr','seed','frequency_axis','phase_initialization')
    result=reconstruct_calibrated(gray,**{k:applied[k] for k in keys if k in applied})
    audio,meta=enhance(result['audio'],result['sr'],applied)
    result['audio'],gain=auto_gain(audio)
    result['metadata'].update(applied_config=applied,vocoder=meta,
                              phase_method='griffin-lim+repair80+bigvgan',**gain)
    return result


def main():
    global ROOT
    import numpy as np
    import torch
    torch.set_num_threads(4)
    if len(sys.argv)==3 and sys.argv[1]=='--component-root':ROOT=Path(sys.argv[2])
    if sys.argv[1:]==['--check']:
        config=dict(time_start=0.,time_end=.2,freq_start=0.,freq_end=8000.,n_iter=2,target_sr=24000,
                    frequency_axis='linear',seed=0,auto_parameters=False,win_length_ms=10,min_db=-60,max_db=0)
        gray=np.full((64,96),230,np.uint8);gray[20:24,::8]=30
        result=reconstruct(gray,config)
        if len(result['audio'])!=4800 or not np.isfinite(result['audio']).all():raise ValueError('m09_vocoder_self_test_failed')
        print(json.dumps(dict(success=True,samples=4800)))
        return
    raw=sys.stdin.buffer.read(32_000_001)
    if not 0<len(raw)<=32_000_000:raise ValueError('invalid_spectrogram_image')
    with np.load(io.BytesIO(raw),allow_pickle=False) as data:
        config=json.loads(str(data['config']));operation=str(data['operation'])
        if operation=='enhance':
            audio,meta=enhance(data['audio'],int(data['sr']),config)
            value=dict(audio=audio,metadata=json.dumps(meta))
            stream=io.BytesIO();np.savez(stream,**value);sys.stdout.buffer.write(stream.getvalue());return
        gray=data['gray']
    if operation=='reconstruct':
        result=reconstruct(gray,config)
        value=dict(audio=result['audio'],sr=result['sr'],image=result['image'],metadata=json.dumps(result['metadata']))
    else:raise ValueError('invalid_spectrogram_config')
    stream=io.BytesIO();np.savez(stream,**value);sys.stdout.buffer.write(stream.getvalue())


if __name__=='__main__':main()
