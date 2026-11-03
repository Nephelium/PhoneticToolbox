"""Actual complete candidate ZIPs, installed in an owned Chrome context.

This validates an already-open website, including closing/reopening its help
tab while fully offline. It does not claim offline navigation to the website.
"""
from pathlib import Path
import json
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from verification_artifacts import verification_run

def main():
    manifest=Path(sys.argv[1]).resolve()
    candidates=json.loads(manifest.read_text(encoding='utf8'))
    base=ROOT/'frontend/public/manual'
    project=json.loads((base/'project.json').read_text(encoding='utf8'))
    assert all(pack['sourceDigest']==project['sourceDigest'] for pack in candidates['packs'])
    with verification_run('manual-language-packs','complete-web-offline',
            'Real complete EN/JA local ZIP import, all browser networking disabled, 22 chapters plus all shared media, help unmount/reopen; not offline website navigation')as(out,scratch):
        spec={'out':str(out),'base':str(base),'packs':[{**pack,'path':str(manifest.parent/pack['file'])}for pack in candidates['packs']]}
        (scratch/'fixture.json').write_text(json.dumps(spec,ensure_ascii=False),encoding='utf8')
        subprocess.run(['node',str(ROOT/'scripts/manual/verify_complete_offline.cjs'),str(scratch/'fixture.json')],cwd=ROOT,check=True)

if __name__=='__main__':main()
