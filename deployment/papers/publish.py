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
    parser.add_argument('--expected-catalog-sha256',help='Refuse to publish if the complete catalog baseline changed.')
    args=parser.parse_args();stage=args.stage.resolve();root=args.root.resolve()
    if root!=Path('/var/www/phonetictoolbox-coming-soon/papers'):raise ValueError('Unexpected publication root')
    root.mkdir(exist_ok=True)
    # This publisher runs on the existing Linux content server. Hold one lock
    # through baseline verification, immutable asset writes and catalog swap.
    import fcntl
    with (root/'.publish.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        publish(stage,root,args.expected_catalog_sha256)


def publish(stage,root,expected):
    if expected:
        if not re.fullmatch(r'[a-f0-9]{64}',expected):raise ValueError('Invalid catalog baseline')
        if hashlib.sha256((root/'catalog.json').read_bytes()).hexdigest()!=expected:
            raise ValueError('Server catalog changed; refresh the baseline before publishing')
    raw=(stage/'catalog.json').read_bytes()
    if len(raw)>4_000_000:raise ValueError('Catalogue too large')
    catalog=json.loads(raw);assert catalog['schema']=='ptb-papers/1'
    verified=[]
    entries=[paper[kind] for paper in catalog['papers'] + catalog.get('journalPapers', [])
             for kind in (('original','translation','analysisPdf') if 'analysisPdf' in paper else ('original','translation'))]
    if 'notice' in catalog: entries.append(catalog['notice']['pdf'])
    for asset in entries:
        relative=asset['path']
        assert re.fullmatch(r'[a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+\.pdf',relative)
        source=stage/relative;target=root/relative
        if not source.is_file():source=target
        assert not source.is_symlink() and (source.resolve().is_relative_to(stage) or source.resolve().is_relative_to(root))
        assert source.stat().st_size==asset['size']<=50_000_000
        assert hashlib.sha256(source.read_bytes()).hexdigest()==asset['sha256']
        if target.exists():
            assert not target.is_symlink() and target.resolve().is_relative_to(root)
            assert target.stat().st_size==asset['size'] and hashlib.sha256(target.read_bytes()).hexdigest()==asset['sha256']
        elif source!=target:
            target.parent.mkdir(exist_ok=True)
            assert target.parent.resolve().is_relative_to(root)
            temp=target.with_name(target.name+'.'+uuid.uuid4().hex+'.tmp')
            shutil.copyfile(source,temp);temp.chmod(0o644);os.replace(temp,target)
        verified.append({'path':relative,'size':asset['size'],'sha256':asset['sha256']})
    temporary=root/('catalog.'+uuid.uuid4().hex+'.tmp')
    temporary.write_bytes(raw);temporary.chmod(0o644);os.replace(temporary,root/'catalog.json')
    print(json.dumps({'schema':'ptb-paper-publication/1','files':verified,'catalogSha256':hashlib.sha256(raw).hexdigest()}))


if __name__=='__main__':main()
