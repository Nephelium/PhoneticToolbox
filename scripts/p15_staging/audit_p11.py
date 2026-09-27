"""Verify downloaded P11 evidence and compare its source hashes, read-only."""
import argparse
import hashlib
import json
from pathlib import Path,PurePosixPath


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(evidence,repo):
    manifest=json.loads((evidence/'evidence-manifest.json').read_text('utf-8'))
    bad=[]
    for name,entry in manifest.items():
        relative=PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Invalid evidence path')
        target=evidence.joinpath(*relative.parts)
        if not target.is_file() or target.stat().st_size!=entry['bytes'] or sha(target)!=entry['sha256']:bad.append(name)
    profile=json.loads((evidence/'runtime2.json').read_text('utf-8'))
    receipt=json.loads((evidence/'receipt2.json').read_text('utf-8'))
    report=json.loads((evidence/'mixed-run1/report.json').read_text('utf-8'))
    digest=sha(evidence/'runtime2.json')
    valid=report.get('success') is True and report.get('thirty_minute_gate_passed') is True and report.get('mixed_elapsed_seconds',0)>=1800
    valid=valid and report.get('profile_sha256')==receipt.get('profile_sha256')==digest and not bad
    operations=[]
    for operation,folder in [('lpc_analysis','lpc-final'),('egg_analysis','egg-final'),('acoustic_analysis','acoustic-final')]:
        p=evidence/folder/'report.json';entry=receipt['operations'][operation]
        r=json.loads(p.read_text('utf-8'))
        bound=sha(p)==entry['sha256'] and r.get('profile_sha256')==digest and r.get('success') is True
        operations.append({'operation':operation,'bound_report':bound});valid=valid and bound
    changed=[];compared=0
    for name,digest_value in profile['hashes'].items():
        mapping=None
        for package,base in [('ptb_api','backend/src'),('ptb_worker','backend/src'),('phonetic_core','packages/phonetic_core/src')]:
            marker='/'+package+'/'
            if marker in name:
                mapping=base+'/'+package+'/'+name.split(marker,1)[1];break
        if mapping and mapping.endswith('.py'):
            compared+=1;current=repo/mapping
            if not current.is_file() or sha(current)!=digest_value:changed.append(mapping)
    return {'schema':'p15-p11-audit/1','historical_evidence_verified':bool(valid),
        'manifest_files':len(manifest),'manifest_mismatches':bad,'runtime_sha256':digest,
        'mixed_report_sha256':sha(evidence/'mixed-run1/report.json'),
        'mixed_elapsed_seconds':report.get('mixed_elapsed_seconds'),'receipt_operations':operations,
        'source_files_compared':compared,'current_source_changed':sorted(changed),
        'integrated_release_perf_verified':False,'public_rtt_seconds':report.get('public_rtt_seconds')}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence',type=Path,required=True);p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=audit(a.evidence,a.repo)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False))
    return 0 if result['historical_evidence_verified'] else 2


if __name__=='__main__':raise SystemExit(main())
