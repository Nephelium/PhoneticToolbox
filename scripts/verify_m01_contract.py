"""M01-D actual native science -> JSON -> paired files; synthetic inputs only.

No HTTP, existing database, Codex browser, or persistent publication is used.
The expected scientific arrays are independent frozen M01-A captures.
"""
from pathlib import Path
import hashlib
import json
import pickle
import sys
from uuid import uuid4
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests/support'))
from baseline_support import RECIPES,create_fixture,load_json,compare
from m01_science import associations,pack
from verify_m01_native_io import short_grid
from phonetic_core.models.associations import AcousticAssociations
from phonetic_core.ports.acoustic import AcousticBackends
from phonetic_core.services.acoustic import analyze_audio
from ptb_worker.io.audio import decode_wav
from ptb_worker.io.annotations import decode_textgrid
from ptb_worker.io.lip import convert_local_legacy_lip,decode_lip
from ptb_worker.io.scratch import Scratch
from ptb_worker.io.parameter_exports import export_analysis
from ptb_worker.native.reaper import Reaper,REAPER_SHA256
from ptb_worker.acoustic_result import to_core_config,build_acoustic_result
from ptb_api.acoustic_models import AcousticRequest,AcousticResult,AcousticSettings,AcousticFileManifest
from ptb_api.acoustic_boundary import TrustedAcousticAsset,resolve_inputs,output_expiry


def uid(n):return f'00000000-0000-4000-8000-{n:012d}'
def sha(raw):return hashlib.sha256(raw).hexdigest()


def main():
    folder=ROOT/'output/validation/m01'/('contract-'+uuid4().hex);folder.mkdir(parents=True)
    rows=[]
    for case in ('ASSOCIATED','FORMULA-EXPORT','SERVICE-NONE'):
        target=folder/case;target.mkdir()
        old=load_json(ROOT/'tests/fixtures/m01'/f'{case}.json.gz')['scientific']
        wav=create_fixture(target,RECIPES[0]);blobs={'audio':wav.read_bytes()}
        assoc=None
        if case!='SERVICE-NONE':
            source=associations(case=='FORMULA-EXPORT')
            blobs['textgrid']=short_grid(source.tiers)
            blobs['lip']=convert_local_legacy_lip(pickle.dumps(source.lip))
            assoc=AcousticAssociations(lip=decode_lip(blobs['lip']),tiers=decode_textgrid(blobs['textgrid']))
        assets={uid(10+i):TrustedAcousticAsset(uid(10+i),uid(2),uid(1),sha(raw),role,role,'ready',200.)
                for i,(role,raw) in enumerate(blobs.items())}
        req=AcousticRequest(project_id=uid(1),idempotency_key='contract-'+case,
            inputs={a.role:dict(asset_id=a.asset_id,sha256=a.sha256) for a in assets.values()},
            config=dict(settings={k:old['config'][k] for k in AcousticSettings.model_fields},
                selection=dict(mode='legacy_service' if case=='SERVICE-NONE' else 'catalog',
                               keys=[] if case=='SERVICE-NONE' else old['config']['selected_parameter_keys']),
                backend_policy=dict(reaper='native_required')))
        snapshots=resolve_inputs(req,uid(2),assets.get,100.)
        audio=decode_wav(blobs['audio'])
        with Scratch(target,4_000_000) as scratch:
            native=Reaper(ROOT/'phonetic_toolbox/core/acoustic/reaper.exe',scratch)
            result=analyze_audio(audio,to_core_config(req.config),assoc,AcousticBackends(reaper=native))
            wire=build_acoustic_result(result,audio,req,snapshots,native_sha256=REAPER_SHA256)
            restored=AcousticResult.model_validate_json(wire.model_dump_json())
            assert restored==wire
            tracks={'Time_s':pack(np.array(restored.times_s))}
            for column in restored.numeric:
                values=np.array([v if mask==0 else (np.nan,np.inf,-np.inf)[mask-1]
                                 for v,mask in zip(column.values,column.nonfinite)])
                # Serialization is exact, including signed infinities and finite zero.
                np.testing.assert_array_equal(values,result.to_dataframe()[column.key].to_numpy())
                tracks[column.key]=pack(values)
            for column in restored.text:tracks[column.key]=pack(np.array(column.values,dtype=object))
            assert restored.column_order==old['columns']
            assert restored.times_s==old['time_axis']
            differences=compare(old['tracks'],tracks)
            assert not differences,differences
            assert restored.metadata.decoded.sample_rate_hz==44100
            assert restored.metadata.decoded.sample_count==len(audio.samples)
            assert any(b.actual=='native_reaper' and b.resource_sha256==REAPER_SHA256 for b in restored.metadata.backends)
            assert {c.key for c in restored.numeric if c.scope=='legacy_service_extension'}==(
                {'SOE_pF0','SOE_rF0'} if case=='SERVICE-NONE' else set())
            pair=export_analysis(result,scratch)  # Includes independent format readback.
            expiry=output_expiry(mode='server',operation='analysis',completed_at=150.,inputs=snapshots)
            files=[]
            for i,(fmt,raw) in enumerate((('xlsx',pair.xlsx),('sqlite',pair.sqlite))):
                (target/f'result.{fmt}').write_bytes(raw)
                files.append(dict(asset_id=uid(30+i),format=fmt,size_bytes=len(raw),sha256=sha(raw),expires_at=expiry))
            manifest=AcousticFileManifest(job_id=uid(20),metadata=restored.metadata,completed_at=150.,
                retention='server',expires_at=expiry,row_count=len(restored.times_s),files=files)
            assert AcousticFileManifest.model_validate_json(manifest.model_dump_json())==manifest
            assert scratch.used==0
        (target/'result.json').write_text(wire.model_dump_json(indent=2)+'\n',encoding='utf-8')
        (target/'manifest.json').write_text(manifest.model_dump_json(indent=2)+'\n',encoding='utf-8')
        assert sha(wav.read_bytes())==snapshots[0].sha256
        rows.append(dict(case=case,rows=len(restored.times_s),columns=len(restored.column_order),
            strict_golden_differences=differences,wire_sha256=sha(wire.model_dump_json().encode()),
            json_bytes=len(wire.model_dump_json().encode()),files=files,
            backend_events=[b.model_dump() for b in restored.metadata.backends]))
    assert not any(k.startswith('phonetic_toolbox') for k in sys.modules)
    report=dict(task='M01-D',platform='Windows',cases=rows,
        scope='Actual bounded native science, lossless wire serialization, prepared pairs; publication remains M01-F')
    (folder/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(report=str((folder/'report.json').relative_to(ROOT)),cases=rows),ensure_ascii=False))


if __name__=='__main__':main()
