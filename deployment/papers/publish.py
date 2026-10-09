"""Publish a staged full catalogue last, after verifying every referenced PDF.

Run as the owner of the /papers directory. Only content within that directory
is written. No service configuration, database or credentials are changed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import uuid


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',type=Path)
    parser.add_argument('--root',type=Path,default=Path('/var/www/phonetictoolbox-coming-soon/papers'))
    args=parser.parse_args();stage=args.stage.resolve();root=args.root.resolve()
    if root!=Path('/var/www/phonetictoolbox-coming-soon/papers'):raise ValueError('Unexpected publication root')
    root.mkdir(exist_ok=True)
    raw=(stage/'catalog.json').read_bytes()
    if len(raw)>4_000_000:raise ValueError('Catalogue too large')
    catalog=json.loads(raw);assert catalog['schema']=='ptb-papers/1'
    verified=[]
    for paper in catalog['papers']:
        for kind in ('original','translation'):
            asset=paper[kind];relative=asset['path']
            assert re.fullmatch(r'[a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+\.pdf',relative)
            source=stage/relative;target=root/relative
            if not source.is_file():source=target
            assert not source.is_symlink() and (source.resolve().is_relative_to(stage) or source.resolve().is_relative_to(root))
            assert source.stat().st_size==asset['size']<=50_000_000
            assert hashlib.sha256(source.read_bytes()).hexdigest()==asset['sha256']
            if source!=target:
                target.parent.mkdir(exist_ok=True)
                assert target.parent.resolve().is_relative_to(root)
                temp=target.with_name(target.name+'.'+uuid.uuid4().hex+'.tmp')
                shutil.copyfile(source,temp);temp.chmod(0o644);os.replace(temp,target)
            verified.append({'path':relative,'size':asset['size'],'sha256':asset['sha256']})
    temporary=root/('catalog.'+uuid.uuid4().hex+'.tmp')
    temporary.write_bytes(raw);temporary.chmod(0o644);os.replace(temporary,root/'catalog.json')
    print(json.dumps({'schema':'ptb-paper-publication/1','files':verified,'catalogSha256':hashlib.sha256(raw).hexdigest()}))


if __name__=='__main__':main()
