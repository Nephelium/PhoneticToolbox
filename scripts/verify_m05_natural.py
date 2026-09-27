"""Authorized local camera recording regression against independent original V2 capture."""
import gzip,json,hashlib,sys
from pathlib import Path
import av,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'backend/src'),str(ROOT/'packages/phonetic_core/src')]
from ptb_worker.m05_video import analyze_video,JsonLinesSink
from ptb_worker.m05_results import export_tables,rows
from ptb_worker.m05_audio import extract_audio
from phonetic_core.lip.sequence import LipConfig

def main():
 p=Path(sys.argv[1]);oracle=json.load(gzip.open(p/'natural-v2.json.gz','rt',encoding='utf8'));inputs=json.loads((p/'natural-inputs.json').read_text('utf8'));report=[]
 for c in inputs['cases']:
  out=p/('offline-'+c['name']);out.mkdir();source=Path(c['video']);expected=next(x for x in oracle['scientific'] if x['case']==c['name'] and not x['filter'])
  hashes=[]
  with av.open(str(source)) as container:
   for f in container.decode(video=0):hashes.append(hashlib.sha256(f.to_ndarray(format='rgb24').tobytes()).hexdigest())
  with (out/'frames.jsonl').open('xb') as stream:meta=analyze_video(source,JsonLinesSink(stream,100_000_000),LipConfig(False))
  landmark_differences=0;metric_differences=0;masks=0;count=0
  for i,row in enumerate(rows(out/'frames.jsonl')):
   count+=1;original=expected['raw'][i];masks+=int((original is None)!=(row['raw_points'] is None))
   if original is not None and row['raw_points'] is not None:
    landmark_differences+=int(not np.array_equal(row['raw_points'],original))
    metric_differences+=int(any(v!=expected['metrics'][k][i] for k,v in row['metrics'].items()))
  meta['audio'],audio_names=extract_audio(source,out);export_tables(out/'frames.jsonl',out,meta)
  (out/'manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf8')
  report.append(dict(case=c['name'],frames=count,v2_frames=len(expected['raw']),rgb_equal=hashes==expected['rgb_hashes'],landmark_different_frames=landmark_differences,metric_different_frames=metric_differences,mask_differences=masks,timing=meta['timing'],audio=meta['audio'],validity=meta['validity']))
 (p/'natural-comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(report))
if __name__=='__main__':main()
