"""Install final neural ZIP into owned scratch, run actual bounded app workers."""
import argparse
import io
import json
import os
from pathlib import Path
import subprocess
import threading
from contextlib import contextmanager
from uuid import uuid4
from dataclasses import replace
from zipfile import ZipFile
import cv2
import numpy as np
import soundfile as sf
from verification_artifacts import no_links, write_json, finish
from ptb_worker import vocoder_catalog
from ptb_worker.vocoder_components import REVISION
from ptb_worker.mfa.components import ComponentManager
from ptb_worker.segmentation import digest, unpack_bundle, SEGMENT_LIMITS
from ptb_worker.io.scratch import Scratch
from ptb_worker.acoustic_executor import collect_scientific
from ptb_worker.io.limits import Cancelled


@contextmanager
def short_run():
    # Native Torch DLL paths need the same short root as the user installation.
    out=Path(__file__).resolve().parents[1]/'output/validation/m09'/('n-'+uuid4().hex[:8])
    no_links(out);out.mkdir(parents=True,exist_ok=False);scratch=out/'scratch';scratch.mkdir()
    marker=dict(schema='ptb-test-run/1',owner_pid=os.getpid(),state='active',
                recipe='final ZIP isolated import; real prediction, worker, cancellation')
    write_json(out/'run.json',marker);error=None
    try:yield out,scratch
    except BaseException as exc:error=exc;raise
    finally:finish(out,marker,error=error)


def main():
    p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);a=p.parse_args()
    pinned=vocoder_catalog.RELEASE
    with short_run() as (out,scratch):
        root=scratch;os.environ['PTB_VOCODER_ROOT']=str(root)
        manager=ComponentManager(root,schema='ptb-vocoder-component/1',version_field='model_revision',version=REVISION,prefix='n')
        archive=a.build/pinned['filename']
        with ZipFile(archive) as z:raw=z.read('component-manifest.json')
        assert digest(raw)==pinned['manifest_sha256']
        manifest=json.loads(raw);manifest.update(sha256=pinned['sha256'],download_bytes=pinned['download_bytes'])
        manifest['files'].append(dict(path='component-manifest.json',bytes=len(raw),sha256=digest(raw)))
        manifest['installed_bytes']+=len(raw)
        def check(target):
            r=subprocess.run([str(target/'env/python.exe'),'-I','-B',str(target/'pipeline.py'),'--check'],capture_output=True,timeout=180,creationflags=subprocess.CREATE_NO_WINDOW)
            (out/'self-test.stderr.txt').write_bytes(r.stderr)
            assert r.returncode==0,r.stderr.decode('utf-8','replace')
            return json.loads(r.stdout)
        manager.import_archive(archive,manifest,check)
        from ptb_worker.vocoder_components import status
        assert status()['pipeline_ready']
        from ptb_worker.spec2wav_preview import render
        # Publicly shipped website image; the test duration is explicit, not original truth.
        project=Path(__file__).resolve().parents[1]
        files=sorted((project/'temporary-site/assets/spectrogram-examples').glob('*-input.png'))
        assert files
        gray=cv2.imread(str(files[0]),0);assert gray is not None
        raw=cv2.imencode('.png',gray)[1].tobytes()
        config=dict(algorithm='neural_pipeline',auto_parameters=True,time_end=.5,freq_start=0,freq_end=8000,
                    frequency_axis='linear',n_iter=32,target_sr=24000,seed=0)
        preview=render(raw,config,timeout=90)
        assert preview['prediction']['win_length_ms']==int(preview['prediction']['win_length_ms'])
        records=[];(scratch/'science').mkdir()
        for changes in [{},dict(auto_parameters=False,win_length_ms=15,min_db=-60,max_db=0,
                               frequency_axis='log',freq_start=50,n_iter=2,time_end=.2)]:
            actual=config|changes
            header=dict(config=actual,sha256=digest(raw))
            with Scratch(scratch/'science',128_000_000) as work:
                request=work.create(json.dumps(header).encode()+b'\n'+raw,'.json')
                bundle=unpack_bundle(collect_scientific('spec2wav',request,work,
                    replace(SEGMENT_LIMITS,timeout_seconds=240,process_bytes=3_000_000_000),lambda:False),64_000_000)
            y,sr=sf.read(io.BytesIO(bundle.payloads[2]));meta=json.loads(bundle.payloads[3])
            assert len(y)==round(actual['time_end']*sr) and sr==24000
            assert np.isfinite(y).all() and 1e-6<np.sqrt(np.mean(y*y))<=.081 and np.max(np.abs(y))<=.951
            assert meta['phase_method']=='griffin-lim+repair80+bigvgan'
            assert meta['hop_length']*4==meta['win_length']
            if not changes:assert meta['prediction']==preview['prediction']
            else:assert meta['phase_initialization']=='random'
            records.append(dict(config=actual,metadata=meta,wav_sha256=digest(bundle.payloads[2])))
        stop=threading.Event();timer=threading.Timer(.2,stop.set);timer.start()
        try:
            with Scratch(scratch/'science',128_000_000) as work:
                request=work.create(json.dumps(header).encode()+b'\n'+raw,'.json')
                try:collect_scientific('spec2wav',request,work,replace(SEGMENT_LIMITS,process_bytes=3_000_000_000),stop.is_set)
                except Cancelled:pass
                else:raise AssertionError('Expected cancellation')
        finally:timer.cancel();timer.join()
        (out/'report.json').write_text(json.dumps(dict(success=True,release=pinned,preview=preview['prediction'],cases=records,
            cancellation=True,scope='Windows CPU final ZIP, isolated import and current bounded workers; no public server'),ensure_ascii=False,indent=2)+'\n','utf-8')
        print('REPORT='+str(out/'report.json'),flush=True)


if __name__=='__main__':main()
