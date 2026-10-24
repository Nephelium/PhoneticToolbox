"""macOS port policy authorized on 2026-10-17: floating differences accepted.

Original strict parity tests stay intact. Domain units, structures and missing
values are checked separately. EGG quotient differences must follow the actual
one-sample event displacement, using the existing practical-equivalence checker.
"""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests/support'))
from practical_equivalence import derived,events

def verify(reference,actual,output):
    from macos_validation_identity import source_hashes,require_identity,sha256
    identity=source_hashes(ROOT);rows=[]
    for folder in (reference,actual):
        for phase in ('host','egg'):
            meta=json.loads((folder/(phase+'.json')).read_text('utf8'))
            require_identity(meta.get('source_sha256'),identity,'Numerical capture')
            if meta.get('capture_sha256')!=sha256(ROOT/'scripts/capture_macos_reference.py') or meta.get('npz_sha256')!=sha256(folder/(phase+'.npz')):
                raise RuntimeError('Numerical capture bytes or recipe changed')
    for phase in ('host','egg'):
        with np.load(reference/(phase+'.npz')) as old,np.load(actual/(phase+'.npz')) as new:
            assert set(old.files)==set(new.files),'Output keys changed'
            for key in old.files:
                a,b=new[key],old[key]
                assert a.shape==b.shape and a.dtype==b.dtype,(key,'structure')
                assert np.array_equal(np.isnan(a),np.isnan(b)) and np.array_equal(np.isinf(a),np.isinf(b)),(key,'missing mask')
                finite=np.isfinite(b);maximum=float(np.abs(a[finite]-b[finite]).max(initial=0))
                if a.dtype.kind in 'iub':np.testing.assert_array_equal(a,b);policy='integer exact'
                elif key.startswith('egg.') and key.rsplit('.',1)[-1] in ('gci_times','goi_times','peak_times'):
                    events(a,b,44100);policy='one sample at 44100 Hz'
                elif key.startswith('egg.') and '.cq' in key:
                    prefix=key.rsplit('.',1)[0]
                    pair=lambda obj:[obj[prefix+'.'+name] for name in ('gci_times','goi_times','peak_times')]
                    quotient=lambda obj:[obj[prefix+'.cq'+str(i)] for i in range(3)]
                    derived(quotient(new),quotient(old),pair(new),pair(old),44100);policy='quotients propagated from bounded events'
                else:
                    atol=1e-6
                    if key.startswith('m01.') and any(x in key.lower() for x in ('hnr','cpp')):atol=.001
                    if key.endswith('f0_hz'):atol=.01
                    if any(x in key for x in ('.times','.frequency')):atol=1e-10
                    np.testing.assert_allclose(a,b,atol=atol,rtol=1e-6,equal_nan=True,err_msg=key)
                    policy=f'absolute {atol}, relative 1e-6'
                rows.append(dict(key=key,shape=list(a.shape),max_abs=maximum,policy=policy))
    require_identity(source_hashes(ROOT),identity,'Numerical source changed during verification')
    report=dict(source_sha256=identity,success=True,schema='ptb-macos-numerical/1',scope='public fixed fixtures and current Windows source; floating difference acceptance authorized by user',comparisons=rows,
        reference_sha256={n:hashlib.sha256((reference/n).read_bytes()).hexdigest() for n in ('host.npz','egg.npz')})
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n');print('MAC NUMERICAL POLICY PASSED',len(rows))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('reference',type=Path);p.add_argument('actual',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    verify(a.reference,a.actual,a.output)
