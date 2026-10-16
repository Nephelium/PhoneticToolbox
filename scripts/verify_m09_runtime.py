"""Exercise native Python grant, real GPU/CPU synthesis and website prediction."""
import argparse
import io
import json
import os
from pathlib import Path
import time
from verification_artifacts import verification_run,write_json
from workbench_source import configure

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--python',type=Path,required=True);args=parser.parse_args()
    configure(ROOT)
    import cv2
    import numpy as np
    import soundfile as sf
    from ptb_desktop.m09_bridge import M09Bridge
    from ptb_worker.m09_parameters import predict
    from ptb_worker.spec2wav_child import prepare
    from ptb_worker.segmentation import unpack_bundle
    with verification_run('m09-parameters','devices','Native picker grant -> isolated GPU model compatibility probe -> same-input CPU/GPU reconstruction') as (out,scratch):
        previous=os.environ.get('PTB_M09_PYTHON_CONFIG')
        os.environ['PTB_M09_PYTHON_CONFIG']=str(scratch/'gpu-python.json')
        report=dict(success=False,cases=[])
        try:
            bridge=M09Bridge();grant=bridge.grant(args.python)
            start=time.perf_counter();report['compatibility']=bridge.check(grant['id']);report['check_seconds']=time.perf_counter()-start
            assert report['compatibility']['cuda'] and report['compatibility']['model_checked']
            image=cv2.imread(str(ROOT/'temporary-site/assets/spectrogram-examples/lu2025-modal.png'),0)
            for seconds,upper in [(.5102,5000),(1.,11025)]:
                predicted=predict(image,dict(time_start=0,time_end=seconds,freq_start=0,freq_end=upper))
                report['cases'].append(dict(kind='website-prediction',seconds=seconds,fmax=upper,**predicted))
                if upper==5000:np.testing.assert_allclose([predicted['predicted_window_ms'],predicted['dynamic_range_db']],[4.09480094909668,83.423828125],rtol=2e-5,atol=2e-5)
            gray=np.full((48,96),220,np.uint8);gray[8:12,::6]=20
            raw=cv2.imencode('.png',gray)[1].tobytes();waves={}
            for device in ('cuda','cpu'):
                start=time.perf_counter()
                config=dict(algorithm='neural_pipeline',auto_parameters=False,reconstruction_profile='calibrated',time_end=.12,freq_end=5000,win_length_ms=5,min_db=-60,max_db=0,n_iter=32,target_sr=0,phase_initialization='random',inference_device=device)
                bundle=unpack_bundle(prepare(raw,config),64_000_000);meta=json.loads(bundle.payloads[3])
                audio,sr=sf.read(io.BytesIO(bundle.payloads[2]));waves[device]=audio
                assert len(audio)==1200 and sr==10000 and np.isfinite(audio).all() and 0<np.max(np.abs(audio))<=.951
                assert meta['vocoder']['inference_device']==device and meta['phase_initialization']=='random'
                assert cv2.imdecode(np.frombuffer(bundle.payloads[1],np.uint8),0).shape==gray.shape
                report['cases'].append(dict(kind=device,seconds=time.perf_counter()-start,metadata=meta))
                print('PASS '+device,flush=True)
            report['cpu_gpu_max_abs_difference']=float(np.max(np.abs(waves['cuda']-waves['cpu'])))
            report['cpu_gpu_rms_difference']=float(np.sqrt(np.mean((waves['cuda']-waves['cpu'])**2)))
            report['success']=True
        finally:
            if previous is None:os.environ.pop('PTB_M09_PYTHON_CONFIG',None)
            else:os.environ['PTB_M09_PYTHON_CONFIG']=previous
            write_json(out/'report.json',report)
        print('REPORT='+str(out/'report.json'),flush=True)


if __name__=='__main__':main()
