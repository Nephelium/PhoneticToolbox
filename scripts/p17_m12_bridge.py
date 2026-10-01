"""P17 owned file bridge; authorized recordings copied byte-for-byte, no audio generation."""
import base64
import hashlib
import json
from pathlib import Path
import shutil
import sys
from uuid import uuid4
from phonetic_core.annotation import parse_document
from ptb_desktop.file_provider import FileProvider
from ptb_desktop.task_bridge import TaskBridge

ROOT = Path(__file__).resolve().parents[1]

def main():
    inventory = json.loads((ROOT/'output/validation/p17/natural-inventory.json').read_text('utf8'))
    audio = Path(inventory['selected']['medium']['path'])
    grid = audio.with_suffix('.TextGrid')
    assert grid.is_file()
    out = ROOT/'output/validation/p17/M12'/uuid4().hex
    inputs = out/'inputs'; inputs.mkdir(parents=True)
    originals = []
    for source in [audio, grid]:
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        shutil.copyfile(source, inputs/source.name)
        originals.append({'path':str(source),'sha256':digest})
    raw=grid.read_bytes();doc=parse_document(raw.decode('utf-16' if raw[:2] in (b'\xff\xfe',b'\xfe\xff') else 'utf-8-sig'))
    (out/'source.json').write_text(json.dumps({'originals':originals,'audio':inventory['selected']['medium'],'document':doc},ensure_ascii=False,indent=2),'utf8')
    provider=FileProvider();grant=provider.choose('input',lambda:str(inputs));bridge=TaskBridge(provider,None)
    print(json.dumps({'ready':True,'out':str(out),'audio':audio.name,'document':doc},ensure_ascii=False),flush=True)
    for line in sys.stdin:
        req={}
        try:
            req=json.loads(line);op=req['op']
            if op=='choose':value=grant
            elif op=='read':
                data,digest=provider.read(req['id']);value={'base64':base64.b64encode(data).decode(),'sha256':digest}
            elif op=='inspect':
                value={'grids':{},'originals_unchanged':all(hashlib.sha256(Path(o['path']).read_bytes()).hexdigest()==o['sha256'] for o in originals)}
                for p in inputs.glob('*.TextGrid'):
                    data=p.read_bytes();value['grids'][p.name]=parse_document(data.decode('utf-16' if data[:2] in (b'\xff\xfe',b'\xfe\xff') else 'utf-8-sig'))
            else:value=bridge.invoke(req)
            print(json.dumps({'id':req['rpc_id'],'value':value},ensure_ascii=False),flush=True)
        except Exception as exc:print(json.dumps({'id':req.get('rpc_id'),'error':str(exc)},ensure_ascii=False),flush=True)
    provider.close()

if __name__=='__main__':main()
