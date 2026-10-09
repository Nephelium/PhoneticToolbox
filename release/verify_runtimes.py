"""Exercise relocated MFA with its real public probe, without user data."""
import argparse
import json
import os
from pathlib import Path
import sys
from uuid import uuid4

ROOT=Path(__file__).resolve().parents[1]
for relative in ('backend/src','packages/phonetic_core/src'):sys.path.insert(0,str(ROOT/relative))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',type=Path);args=parser.parse_args()
    stage=args.stage.resolve()
    if not stage.is_relative_to(ROOT/'output/release-staging'):raise ValueError('Expected owned stage')
    output=ROOT/'output/validation/runtime-relocation'/uuid4().hex
    output.mkdir(parents=True)
    os.environ['PTB_M11_BUNDLED_COMPONENTS']=str(stage/'mfa')
    os.environ['PTB_M11_COMPONENT_ROOT']=str(output/'components')
    from ptb_worker.mfa.runtime import load_registry,select
    from ptb_worker.mfa.probe import register
    registry=load_registry()
    model=next(row for row in registry['models'] if row.get('dictionary_name')=='mandarin_pinyin_tab.dict')
    runtime,model=select(model['validated_runtime'],model['id'])
    result=register(runtime['path'],model['model'],model['dictionary'],root=output/'probe',publish=False)
    receipt=result['receipt']
    if receipt['runtime_fingerprint']!=runtime['fingerprint']:raise AssertionError('Relocation changed runtime identity')
    (output/'report.json').write_text(json.dumps(dict(success=True,scope='Relocated real MFA; public synthetic formant-a probe, not natural boundary accuracy',
        runtime_id=runtime['id'],model_id=model['id'],receipt=receipt),ensure_ascii=False,indent=2),'utf8')
    print(json.dumps(dict(success=True,report=str(output/'report.json'))),flush=True)

if __name__=='__main__':main()
