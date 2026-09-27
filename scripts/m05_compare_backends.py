"""Separate detector differences from same-input metric port differences; fixed gate."""
import gzip
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'packages/phonetic_core/src'))
from phonetic_core.lip.metrics import extract_lip_metrics
from phonetic_core.lip.sequence import LipSequence,LipConfig


def main():
    with gzip.open(ROOT/'tests/fixtures/m05/v2.json.gz','rt',encoding='utf-8') as stream: original=json.load(stream)
    browser=json.loads((ROOT/'output/validation/m05/chrome-feasibility.json').read_text('utf-8'))
    fixtures=json.loads((ROOT/'output/validation/m05/inputs/manifest.json').read_text('utf8'))
    spec=json.loads((ROOT/'resources/m05/metric-spec.json').read_text('utf8'))
    comparisons=[]
    for probe in browser['probes']:
        if not probe['success']:continue
        case=next(c for c in original['scientific'] if c['case']==probe['case'] and not c['filter'])
        expected_frames=next(c['frames'] for c in fixtures['cases'] if c['name']==probe['case'])
        timeline_matches=len(probe['results'])==len(expected_frames) and all(abs(r['time_ms']/1000-f['time_s'])<1e-12 for r,f in zip(probe['results'],expected_frames))
        distances=[];differences={key:[] for key in original['spec']['keys']};gate=timeline_matches
        mask_differences=sum((a['points'] is None)!=(b is None) for a,b in zip(probe['results'],case['raw']))
        for candidate,legacy in zip(probe['results'],case['raw']):
            if candidate['points'] is None or legacy is None:continue
            a,b=np.asarray(candidate['points'],np.float32),np.asarray(legacy,np.float32)
            distances.extend(np.linalg.norm(a-b,axis=1).tolist())
            gate=gate and bool(np.all(np.abs(a-b)<=np.maximum(1e-6,2e-6*np.abs(b))))
            # Use one Python formula for BOTH detectors to isolate model differences.
            ma,mb=extract_lip_metrics(a),extract_lip_metrics(b)
            for key in differences:
                if np.isfinite(ma[key]) and np.isfinite(mb[key]):
                    delta=abs(ma[key]-mb[key]);differences[key].append(delta)
                    gate=gate and delta<=max(1e-6,abs(mb[key])*2e-6)
                elif np.isfinite(ma[key])!=np.isfinite(mb[key]):gate=False
        def stats(values):
            return dict(mean=float(np.mean(values)),p95=float(np.percentile(values,95)),maximum=float(np.max(values))) if values else None
        neighbors=[np.asarray(n,np.int64) for n in spec['neighbors']]
        filtered=[LipSequence(LipConfig(True),neighbors) for _ in range(2)]
        movement={'web_post_filter':[],'legacy_post_filter':[]};metric_after={k:[] for k in differences}
        for candidate,legacy,frame in zip(probe['results'],case['raw'],expected_frames):
            a=filtered[0].process(candidate['points'],frame['time_s'])
            b=filtered[1].process(legacy,frame['time_s'])
            if a['points'] is not None:
                movement['web_post_filter'].extend(np.linalg.norm(np.asarray(a['points'])-np.asarray(candidate['points']),axis=1).tolist())
            if b['points'] is not None:
                movement['legacy_post_filter'].extend(np.linalg.norm(np.asarray(b['points'])-np.asarray(legacy),axis=1).tolist())
            if a['metrics'] and b['metrics']:
                for key in metric_after:
                    if a['metrics'][key] is not None and b['metrics'][key] is not None:metric_after[key].append(abs(a['metrics'][key]-b['metrics'][key]))
        comparisons.append(dict(case=probe['case'],delegate=probe['delegate'],mode=probe['mode'],
                                post_filter_diagnostic=dict(clock='same decoded PTS for both; separate from V2 nominal-fps oracle',displacement_px={k:stats(v) for k,v in movement.items()},metric_absolute_difference={k:stats(v) for k,v in metric_after.items()}),
                                frames=len(probe['results']),timeline_matches=timeline_matches,mask_differences=mask_differences,
                                landmarks_pixel_distance=stats(distances),metric_absolute_difference={k:stats(v) for k,v in differences.items()},
                                equivalence_gate=bool(gate and not mask_differences),
                                inference_ms=stats([r['inference_ms'] for r in probe['results']]),
                                heartbeat_max_ms=probe['heartbeat_max_ms']))
    report=dict(schema='m05-detector-comparison/1',comparisons=comparisons,
                candidate_only=not all(c['equivalence_gate'] for c in comparisons),
                thresholds=dict(absolute=1e-6,relative=2e-6,mask='exact',source='predeclared implementation plan'),
                limitations=['Public portrait transformations only','No natural speech, profile or device claim',
                             'CPU/GPU here means requested delegate, not proof of hardware acceleration',
                             'Model smoothing and old post-filter have separate identities'])
    (ROOT/'output/validation/m05/backend-comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(comparisons=len(comparisons),equivalent=sum(c['equivalence_gate'] for c in comparisons),candidate_only=report['candidate_only'])))


if __name__=='__main__':main()
