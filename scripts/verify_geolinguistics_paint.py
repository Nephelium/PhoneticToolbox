"""Owned synthetic Canvas/SVG/PNG layering regression of actual M19 components."""
import os
from pathlib import Path
import subprocess
from verification_artifacts import verification_run

ROOT=Path(__file__).resolve().parents[1]

if __name__=='__main__':
    with verification_run('geolinguistics-paint','browser',
        'Real Page/GeoMap/OpenLayers/exports; synthetic source/save port; overlapping opaque territory face, dispute and ten boundary symbols') as (out,scratch):
        print(out,flush=True)
        result=subprocess.run(['node',str(ROOT/'tests/e2e/geolinguistics-paint.cjs'),str(out),str(scratch)],cwd=ROOT,
            capture_output=True,text=True,encoding='utf-8',errors='replace',creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        (out/'browser.log').write_text(result.stdout+result.stderr,encoding='utf-8')
        if result.returncode:raise RuntimeError('Actual paint/export regression failed; see '+str(out/'report.json'))
