"""Compatibility probe; no package installation or user environment mutation."""
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))


def check(component=None):
    try:
        import numpy as np
        import scipy.signal
        import torch
        import librosa
        import matplotlib
        import huggingface_hub
        from vocoder_child import devices,synthesize
        from m09_neural_models import VisibleMelNAFNet
    except ImportError:return dict(cuda=False,error='m09_python_dependencies')
    result=devices()
    if not result['cuda']:return dict(cuda=False,error='m09_gpu_unavailable')
    try:
        with torch.inference_mode():
            x=torch.full((1,80,8),-5.,device='cuda')
            y=VisibleMelNAFNet(80).eval().cuda()(x,torch.zeros(1,6,device='cuda'),torch.ones(1,80,device='cuda'))
            assert y.shape==x.shape and torch.isfinite(y).all()
        if component:
            import m09_pipeline_child as pipeline
            pipeline.ROOT=component
            t=np.arange(2400)/24000
            config=dict(freq_start=0,freq_end=8000,inference_device='cuda')
            if (component/'models/repair80.pt').is_file():audio,meta=pipeline.enhance(.1*np.sin(2*np.pi*150*t),24000,config)
            else:audio,meta=synthesize(.1*np.sin(2*np.pi*150*t),24000,component/'model',device='cuda')
            assert len(audio)==2400 and np.isfinite(audio).all() and meta['inference_device']=='cuda'
        result.update(torch_version=torch.__version__,python_version=sys.version.split()[0],model_checked=bool(component))
        return result
    except Exception:return dict(cuda=False,error='m09_vocoder_failed')


if __name__=='__main__':
    component=Path(sys.argv[2]) if len(sys.argv)==3 and sys.argv[1]=='--component-root' else None
    print(json.dumps(check(component)))
