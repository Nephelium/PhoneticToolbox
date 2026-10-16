"""Real relocated component + actual bounded M09 worker, isolated scratch only."""
import io
import json
import os
from pathlib import Path
import sys
import threading
from dataclasses import replace

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'backend/src'),str(ROOT/'packages/phonetic_core/src')]
from verification_artifacts import verification_run
from ptb_worker.mfa.components import ComponentManager
from ptb_worker.vocoder_components import REVISION
from ptb_worker import vocoder_catalog
from ptb_worker.segmentation import unpack_bundle,SEGMENT_LIMITS,digest
from ptb_worker.acoustic_executor import collect_scientific
from ptb_worker.io.scratch import Scratch
from ptb_worker.io.limits import Cancelled


def main():
    import argparse,subprocess
    import numpy as np
    import soundfile as sf
    ap=argparse.ArgumentParser();ap.add_argument('--build',type=Path,required=True);args=ap.parse_args()
    pinned=vocoder_catalog.RELEASE
    with verification_run('m09','v','final ZIP to isolated root; fixed CPU runner; three website cases; current bounded M09 child and cancellation') as (out,scratch):
        # This is the explicitly requested installed user component, retained as
        # a deliverable. It is separate from temporary QA inputs and scratch.
        from ptb_worker.vocoder_components import root
        target_root=root()
        if (target_root/'current.json').exists():
            raise ValueError('Existing component registration requires reuse verification')
        manager=ComponentManager(target_root,schema='ptb-vocoder-component/1',version_field='model_revision',version=REVISION,prefix='bigvgan-r1')
        from zipfile import ZipFile
        archive=args.build/pinned['filename']
        with ZipFile(archive) as z: raw=z.read('component-manifest.json')
        assert digest(raw)==pinned['manifest_sha256']
        trusted=json.loads(raw);trusted.update(sha256=pinned['sha256'],download_bytes=pinned['download_bytes'])
        trusted['files'].append(dict(path='component-manifest.json',bytes=len(raw),sha256=digest(raw)))
        trusted['installed_bytes']+=len(raw)
        def check(target):
            tested=subprocess.run([str(target/'env/python.exe'),'-I','-B',str(target/'runner.py'),'--check'],capture_output=True,timeout=180,creationflags=subprocess.CREATE_NO_WINDOW)
            (out/'relocated.stderr.txt').write_bytes(tested.stderr)
            assert tested.returncode==0,tested.stderr.decode('utf-8',errors='replace')
            return json.loads(tested.stdout)
        installed=manager.import_archive(archive,trusted,check)
        from ptb_worker.vocoder_components import status
        assert status()['ready']
        cases=json.loads((ROOT/'temporary-site/assets/spectrogram-examples/provenance.json').read_text('utf-8'))['cases']
        (scratch/'science').mkdir();(scratch/'cancel').mkdir()
        report=[]
        for case in cases:
            source=(ROOT/'temporary-site/assets/spectrogram-examples'/(case['id']+'-input.png')).read_bytes()
            outputs={}
            for use in ['none','bigvgan']:
                config=case['config']|dict(vocoder=use)
                header=dict(config=config,sha256=digest(source))
                with Scratch(scratch/'science',128_000_000) as work:
                    request=work.create(json.dumps(header).encode()+b'\n'+source,'.json')
                    bundle=unpack_bundle(collect_scientific('spec2wav',request,work,replace(SEGMENT_LIMITS,timeout_seconds=240,process_bytes=3_000_000_000),lambda:False),64_000_000)
                audio,sr=sf.read(io.BytesIO(bundle.payloads[2]));meta=json.loads(bundle.payloads[3])
                assert np.isfinite(audio).all() and np.std(audio)>1e-6 and sr==case['config']['target_sr']
                assert meta['config']['vocoder']==use
                assert meta['phase_method']==('griffin-lim' if use=='none' else 'griffin-lim+bigvgan')
                if use=='bigvgan':assert meta['vocoder']['model_revision']==REVISION and 'PKG-BIGVGAN-BASE' in meta['source_ids']
                outputs[use]=dict(samples=len(audio),sample_rate=sr,duration=len(audio)/sr,metadata=meta,audio_sha256=digest(bundle.payloads[2]))
            assert outputs['none']['samples']==outputs['bigvgan']['samples']
            report.append(dict(case=case['id'],outputs=outputs))
        # Cancel the actual parent job. Windows JOB_OBJECT kills its descendants.
        stop=threading.Event();timer=threading.Timer(.2,stop.set);timer.start()
        try:
            with Scratch(scratch/'cancel',128_000_000) as work:
                request=work.create(json.dumps(header).encode()+b'\n'+source,'.json')
                try:collect_scientific('spec2wav',request,work,replace(SEGMENT_LIMITS,process_bytes=3_000_000_000),stop.is_set)
                except Cancelled:pass
                else:raise AssertionError('Cancellation did not interrupt the real job')
        finally:timer.cancel();timer.join()
        (out/'report.json').write_text(json.dumps(dict(success=True,component=pinned,cases=report,cancellation='actual owned process interrupted'),ensure_ascii=False,indent=2)+'\n','utf-8')
        print('REPORT='+str(out/'report.json'),flush=True)


if __name__=='__main__':main()
