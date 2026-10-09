"""Validate final EXE segments and full-size persistent-cache copy fallback."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'desktop/src'))
from PyInstaller.archive.readers import CArchiveReader
from ptb_desktop import startup_cache as cache

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--exe',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();out=args.output.resolve();exe=args.exe.resolve()
    if not out.is_relative_to(Path('D:/PTB-Compact-QA-20261006')):raise ValueError('Expected a new external owned QA directory')
    out.mkdir(parents=True,exist_ok=False)
    config=json.loads(CArchiveReader(str(exe)).extract('cache-payload.json'))
    payload=cache.Payload(exe,config);original=cache.os.link;start=time.monotonic()
    def unsupported(*args,**kwargs):raise OSError('Owned test volume does not support hard links')
    cache.os.link=unsupported
    try:app,live,cold=cache.prepare(payload,out/'cache');live.close()
    finally:cache.os.link=original
    assert cold['scienceAttachment']=={'linked':0,'copied':len(config['components']['science']['files'])}
    assert cold['hostAttachment']=={'linked':0,'copied':len(config['components']['host']['files'])}
    for name,source in config['sharedFiles'].items():
        assert not os.path.samefile(app.parent/'_internal'/name,app.parent/'_internal'/source)
    # Complete fresh content checks, including every copied science/host file.
    assert cache.verify_tree(app.parent,config['components']['apps']['files'],{})
    _,live,warm=cache.prepare(payload,out/'cache');live.close();assert warm['prepared']==[]
    target=app.parent/'PhoneticToolbox.exe';before=target.stat()
    with target.open('r+b') as file:file.seek(10);byte=file.read(1);file.seek(10);file.write(bytes([byte[0]^1]))
    os.utime(target,ns=(before.st_atime_ns,before.st_mtime_ns))
    _,live,repaired=cache.prepare(payload,out/'cache');live.close()
    assert repaired['prepared']==['apps'] and repaired['reused']==['science','host']
    assert cache.verify_tree(app.parent,config['components']['apps']['files'],{})
    report={'success':True,'scope':'Final EXE real embedded payload; all hard links forced unsupported, actual original SHA checks; same-size/mtime corruption repaired; not another physical filesystem',
            'cold':cold,'warm':warm,'repaired':repaired,'seconds':time.monotonic()-start,
            'files':{k:len(v['files']) for k,v in config['components'].items()}}
    (out/'report.json').write_text(json.dumps(report,indent=2),'utf8');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
