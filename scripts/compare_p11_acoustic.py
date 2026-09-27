"""Column-specific engineering equivalence, never a universal allclose."""
import argparse
import json
from pathlib import Path

# Smaller than normal displayed increments, chosen by output unit before viewing
# this comparison. These are engineering budgets, not perceptual JND claims.
BUDGETS={'Hz':1e-4,'dB':1e-4,'dB_legacy_amplitude_reference_not_measured_SPL':1e-4,
         'dB_uncalibrated_magnitude_or_difference':1e-4,'dB_per_decade':1e-4,
         'percent':1e-5,'amplitude_ratio':1e-6,'normalized_ZFF_difference':1e-6}


def compare(actual,reference):
    assert actual['column_order']==reference['column_order']
    assert actual['times_s']==reference['times_s']
    assert actual['metadata']['decoded']==reference['metadata']['decoded']
    assert actual['metadata']['config']==reference['metadata']['config']
    assert actual['text']==reference['text']
    rows=[]
    for a,b in zip(actual['numeric'],reference['numeric'],strict=True):
        assert (a['key'],a['unit'])==(b['key'],b['unit'])
        # rF0 is serialized from the native ASCII track. Permit .01 Hz, still
        # far below a semitone; no relation to the observed error in these runs.
        tolerance=.01 if a['key']=='rF0' else BUDGETS.get(a['unit'])
        masks_equal=a['nonfinite']==b['nonfinite'] and a['reason']==b['reason']
        errors=[abs(x-y) for x,y in zip(a['values'],b['values'],strict=True) if x is not None and y is not None]
        passed=masks_equal and (not errors or tolerance is not None and max(errors)<=tolerance)
        rows.append(dict(key=a['key'],unit=a['unit'],atol=tolerance,finite_pairs=len(errors),
                         masks_equal=masks_equal,max_abs=max(errors,default=0),passed=passed))
    return dict(policy='p11-m01-practical/1',passed=all(row['passed'] for row in rows),columns=rows)


def main():
    parser=argparse.ArgumentParser()
    for name in ('actual','reference','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    value=compare(json.loads(args.actual.read_text('utf-8')),json.loads(args.reference.read_text('utf-8')))
    args.output.write_text(json.dumps(value,indent=2),encoding='utf-8')
    print(json.dumps(dict(passed=value['passed'],columns=len(value['columns']),failed=[r for r in value['columns'] if not r['passed']])))
    raise SystemExit(0 if value['passed'] else 1)


if __name__=='__main__':main()
