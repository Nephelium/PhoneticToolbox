"""Read-only authorized corpus; verified slices go only to ignored validation output."""
import argparse
import hashlib
import io
import json
from pathlib import Path
from uuid import uuid4
import numpy as np
from scipy.io import wavfile
from ptb_worker.io.annotations import decode_textgrid
from ptb_worker.io.scratch import Scratch
from ptb_worker.segmentation import prepare_segments


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--corpus',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    out=root/'output/validation/m01-m02-r1'/('natural-'+uuid4().hex);out.mkdir(parents=True)
    report={'success':False,'files':[],'segments':0,'source_hashes_unchanged':False}
    sources={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.corpus.iterdir() if p.suffix.lower() in ('.wav','.textgrid','.json')}
    try:
        for index,path in enumerate(sorted(args.corpus.glob('*.wav'))):
            raw=path.read_bytes();grid=path.with_suffix('.TextGrid').read_bytes()
            rate,samples=wavfile.read(io.BytesIO(raw));tiers=decode_textgrid(grid)
            record={'name':path.name,'duration':len(samples)/rate,'tiers':[]}
            for tier in tiers:
                with Scratch(out,100_000_000) as scratch:
                    bundle=prepare_segments(raw,grid,tier.name,scratch,audio_name=path.name)
                for segment,blob in zip(bundle.manifest['segments'],bundle.payloads):
                    fs,values=wavfile.read(io.BytesIO(blob))
                    assert fs==rate and values.dtype==samples.dtype
                    np.testing.assert_array_equal(values,samples[int(segment['start_s']*rate):int(segment['end_s']*rate)])
                record['tiers'].append({'name':tier.name,'segments':len(bundle.payloads),'sample_exact':True})
                report['segments']+=len(bundle.payloads)
                if index==0:
                    target=out/'example'/tier.name;target.mkdir(parents=True,exist_ok=True)
                    for file,blob in zip(bundle.manifest['files'],bundle.payloads):(target/file['name']).write_bytes(blob)
                    parent=path.with_suffix('.ptb.json')
                    if parent.exists():
                        with Scratch(out,100_000_000) as scratch:
                            joined=prepare_segments(raw,grid,tier.name,scratch,audio_name=path.name,parent_result=parent.read_bytes())
                        assert all(s['parameter_status']=='included' for s in joined.manifest['segments'])
                        record['parameter_slice']=True
            report['files'].append(record)
            if (index+1)%12==0:print(f'Verified {index+1} WAV pairs',flush=True)
        report['source_hashes_unchanged']=all(hashlib.sha256(p.read_bytes()).hexdigest()==sha for p,sha in sources.items())
        assert report['source_hashes_unchanged'] and report['files']
        report['success']=True
    finally:
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf-8')
        print(str(out/'report.json'),flush=True)


if __name__=='__main__':main()
