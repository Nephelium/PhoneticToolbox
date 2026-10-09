"""Exercise copy fallback against the complete real build archives."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'desktop/src'))
from ptb_desktop import compact_host


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--work',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    work,out=args.work.resolve(),args.output.resolve()
    if not work.is_relative_to(ROOT/'output') or not out.is_relative_to(ROOT/'output/validation'):
        raise ValueError('Expected owned build and validation directories')
    out.mkdir(parents=True,exist_ok=False)
    root=out/'payload';root.mkdir()
    shutil.copyfile(work/'snapshot/desktop-bundle.json',root/'desktop-bundle.json')
    for name in ('runtime-files.json','runtime-payload.tar.xz'):
        shutil.copyfile(work/'snapshot/runtime-archive'/name,root/name)
    for name in ('host-files.json','host-payload.tar.xz','host-archive.json'):
        shutil.copyfile(work/'host-archive'/name,root/name)
    original=compact_host.os.link
    def unsupported(*args,**kwargs):raise OSError('Owned test: volume does not support hard links')
    started=time.monotonic()
    compact_host.os.link=unsupported
    try:compact_host.expand(root)
    finally:compact_host.os.link=original
    value=json.loads((root/'host-files.json').read_text('utf8'))
    for name,row in value['files'].items():
        target=root/name
        assert target.stat().st_size==row['size'] and compact_host.digest(target)==row['sha256'],name
    for name,source in value['sharedFiles'].items():
        assert not os.path.samefile(root/name,root/source),name
    receipt=json.loads((root/'.ptb-host-ready.json').read_text('utf8'))
    assert receipt['restoration']['copiedFiles']==len(value['sharedFiles'])
    assert receipt['restoration']['linkedFiles']==0
    compact_host.expand(root)
    report=dict(success=True,hostFiles=len(value['files']),seconds=time.monotonic()-started,
                restoration=receipt['restoration'],scope='Real archives; forced hard-link unsupported error; all original host file hashes checked; no separate volume or machine')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
