"""M01-G read-only, previously authorized local recordings vs frozen P03 v2.

Records only case IDs, hashes and comparison counts. Inputs and scientific arrays
remain local and are never copied to committed fixtures or sent over HTTP.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import json
import sys
from uuid import uuid4
import numpy as np
from baseline_support import load_json,compare,sha
from phonetic_core.models.acoustic import AcousticConfig
from phonetic_core.models.associations import AcousticAssociations
from phonetic_core.ports.acoustic import AcousticBackends
from phonetic_core.services.acoustic import analyze_audio
from ptb_worker.io.audio import decode_wav
from ptb_worker.io.annotations import decode_textgrid
from ptb_worker.io.scratch import Scratch
from ptb_worker.native.reaper import Reaper

ROOT=Path(__file__).resolve().parents[1]


def pack(value):
    if isinstance(value,dict):return {k:pack(v) for k,v in value.items()}
    if isinstance(value,np.ndarray):
        row=dict(dtype=str(value.dtype),shape=list(value.shape),storage='values')
        if value.dtype.kind in 'fiu':
            row.update(values=np.where(np.isfinite(value),value,None).tolist(),
                nonfinite=np.where(np.isnan(value),1,np.where(np.isposinf(value),2,np.where(np.isneginf(value),3,0))).tolist())
        else:row['values']=value.tolist()
        return row
    return value


def main():
    out=ROOT/'output/validation/m01'/('natural-'+uuid4().hex);out.mkdir()
    baseline=ROOT/'output/validation/p03/20260909-165413'
    index={c['id']:c for c in load_json(baseline/'capture-summary.json')['cases']}
    rows=[]
    for case in ('LOCAL-01','LOCAL-06','LOCAL-09','LOCAL-12'):
        folder=baseline/case/'1';old=load_json(folder/'result.json.gz');request=load_json(folder/'request.json')
        wav=Path(request['input']);grid=Path(request['textgrid']) if request.get('textgrid') else None
        before=sha(wav);assert before==old['input_sha256']==index[case]['input_sha256'],case+' input changed since P03'
        tg_hash=sha(grid) if grid else None
        if grid:assert tg_hash in index[case]['sidecar_sha256'],case+' annotation changed since P03'
        expected=old['scientific'];assert expected['status']=='returned'
        audio=decode_wav(wav.read_bytes());associations=AcousticAssociations(tiers=decode_textgrid(grid.read_bytes()) if grid else ())
        config=AcousticConfig(**{k:v for k,v in expected['config'].items() if k!='reaper_bin_path'})
        target=out/case;target.mkdir()
        with Scratch(target,16_000_000) as scratch:
            native=Reaper(ROOT/'phonetic_toolbox/core/acoustic/reaper.exe',scratch)
            result=analyze_audio(audio,config,associations,AcousticBackends(reaper=native))
            assert scratch.used==0
        # M01-D01 intentionally corrects only sampling-rate metadata.
        expected_tracks={k:v for k,v in expected['result'].items() if k!='sampling_rate'}
        actual={k:pack(getattr(result,k)) for k in expected_tracks}
        differences=compare(expected_tracks,actual)
        entry=dict(case=case,sample_rate_hz=audio.sample_rate_hz,sample_count=len(audio.samples),
            frames=len(result.time_axis),columns=len(result.to_dataframe().columns),input_sha256=before,textgrid_sha256=tg_hash,
            v2_capture_sha256=sha(folder/'result.json.gz'),differences=differences,
            actual_backends=result.backend_events,metadata_rate_corrected=result.sampling_rate==audio.sample_rate_hz)
        rows.append(entry)
        (out/'report.json').write_text(json.dumps(dict(task='M01-G',cases=rows,privacy='local only; no recording or array copied'),ensure_ascii=False,indent=2),encoding='utf-8')
        assert not differences,case+': '+str(differences)
        assert list(result.to_dataframe().columns)==expected['dataframe_columns']
        assert result.sampling_rate==audio.sample_rate_hz
        assert sha(wav)==before and (not grid or sha(grid)==tg_hash)
        print(case+': exact time/mask and original-tolerance numeric comparison passed',flush=True)
    assert not any(k.startswith('phonetic_toolbox') for k in sys.modules)
    print(str((out/'report.json').relative_to(ROOT)))


if __name__=='__main__':main()
