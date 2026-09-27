"""Compare Windows captured originals with Linux originals, then Linux migrated exactly."""
import sys,json,itertools
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'tests/support'));sys.path.insert(0,str(root/'tests/parity'))
import m07_baseline as old
out=root/'output/validation/m07/linux-original';old.EVIDENCE=out
first,meta=old.capture('round1');second,other=old.capture('round2')
assert meta==other
for k in first:np.testing.assert_array_equal(first[k],second[k])
windows=np.load(root/'tests/fixtures/m07/v2.npz',allow_pickle=False);diff=[]
for k,a in first.items():
 b=windows[k]
 if not np.array_equal(a,b):diff.append(dict(field=k,max_abs=float(np.max(np.abs(a.astype(float)-b.astype(float)))) if a.shape==b.shape else None,shape=list(a.shape)))
import test_phonation_synthesis as tests
tests.DATA=first;tests.META=meta;analyses=tests.analyses.__wrapped__()
for i in range(4):tests.test_exact_analysis(analyses,i)
for reverse,kind,energy,normalize in itertools.product((False,True),(1,2,3),(False,True),(False,True)):tests.test_exact_generation(analyses,reverse,kind,energy,normalize)
for mode in ('normalize','onset'):tests.test_exact_controls_and_lpc_reuse(analyses,mode)
for n in (1,63,128,129,160,1000):tests.test_exact_tail(n)
report=dict(original_double_exact=True,migrated_same_platform_exact=True,cross_windows_differing_fields=diff,versions={k:meta[k] for k in ('numpy','scipy','parselmouth')},windows_gate=False)
(out/'report.json').write_text(json.dumps(report,indent=2),'utf8');print(json.dumps(report))
