"""Test-only stdio transport for actual natural-input bounded Praat preview."""
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
SOURCES=[str(ROOT/p) for p in ('backend/src','packages/phonetic_core/src')]
sys.path[:0]=SOURCES;os.environ['PYTHONPATH']=os.pathsep.join(SOURCES)
from ptb_worker.spectrogram_session import SpectrogramSession

def main():
    manifest=json.loads((ROOT/'output/validation/p17/natural-inventory.json').read_text(encoding='utf8'))
    session=SpectrogramSession();raws={}
    try:
        for line in sys.stdin:
            request=json.loads(line)
            if request.get('op')=='shutdown':break
            try:
                kind=request['kind']
                if kind not in raws:raws[kind]=Path(manifest['selected'][kind]['path']).read_bytes()
                value=session.render(raws[kind],**request['view'])
                response=dict(id=request['id'],value=value)
            except Exception as exc:response=dict(id=request['id'],error=str(exc))
            print(json.dumps(response),flush=True)
    finally:session.close()

if __name__=='__main__':main()
