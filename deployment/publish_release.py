"""Publish authorized immutable downloads, then atomically replace small manifests.

The JSON plan maps local files to /releases/ paths. Credentials come only from
the supplied local SSH helper. Remote and local full hashes must match.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import shlex
import sys
import time

REMOTE = r'''
import hashlib,json,os,shutil,sys
from pathlib import Path
p=json.loads(sys.stdin.readline())
root=Path('/var/www/phonetictoolbox-coming-soon')
target=root/p['relative']
assert target.resolve().is_relative_to(root/'releases') and not target.is_symlink()
def same(file):
    if not file.is_file() or file.stat().st_size!=p['size']:return False
    with file.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()==p['sha256']
if p['action']=='check':
    print(json.dumps({'match':same(target),'exists':target.exists()}))
else:
    stage=Path('/home/admin')/('ptb-release-'+p['sha256']+'.part')
    assert stage.resolve()==stage and same(stage)
    assert p.get('mutable') or not target.exists() or same(target),'Immutable target already differs'
    target.parent.mkdir(parents=True,exist_ok=True)
    if not same(target):
        pending=target.with_name(target.name+'.'+p['sha256'][:16]+'.tmp')
        shutil.copyfile(stage,pending)
        assert same(pending)
        os.chmod(pending,0o644)
        os.replace(pending,target)
    stage.unlink()
    print(json.dumps({'success':same(target),'path':str(target),'sha256':p['sha256'],'size':p['size']}))
'''

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True)
    ap.add_argument('--ssh-helper',type=Path,required=True);ap.add_argument('--apply',action='store_true')
    args=ap.parse_args();plan=json.loads(args.plan.read_text('utf8'))
    rows=[]
    for item in plan['files']:
        file=Path(item['local']).resolve();relative=PurePosixPath(item['relative'])
        assert not relative.is_absolute() and relative.parts[0]=='releases' and '..' not in relative.parts
        with file.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
        mutable=bool(item.get('mutable'))
        assert not mutable or (relative.name=='latest.json' and file.stat().st_size<65536)
        rows.append(dict(local=str(file),relative=str(relative),size=file.stat().st_size,sha256=sha,mutable=mutable))
    receipt=args.plan.with_suffix('.receipt.json');results=[]
    def save():receipt.write_text(json.dumps({'applied':args.apply,'files':results},indent=2)+'\n','utf8')
    if not args.apply:results.extend(rows);save();return
    sys.path.insert(0,str(args.ssh_helper.parent))
    spec=importlib.util.spec_from_file_location('release_ssh',args.ssh_helper)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    with helper.connect() as client:
        def remote(row,action):
            stdin,stdout,stderr=client.exec_command('sudo -n python3 -c '+shlex.quote(REMOTE),timeout=600)
            stdin.write(json.dumps(dict(row,action=action))+'\n');stdin.flush();stdin.channel.shutdown_write()
            out=stdout.read().decode();err=stderr.read().decode();code=stdout.channel.recv_exit_status()
            if code:raise RuntimeError(err or out)
            return json.loads(out)
        for row in rows:
            state=remote(row,'check')
            if state['match']:
                results.append(dict(row,success=True,reused=True));save();continue
            if state['exists'] and not row['mutable']:raise RuntimeError('Immutable target differs: '+row['relative'])
            stage='/home/admin/ptb-release-'+row['sha256']+'.part'
            with client.open_sftp() as sftp:
                try:offset=sftp.stat(stage).st_size
                except FileNotFoundError:offset=0
                assert offset<=row['size']
                last=time.monotonic()
                with open(row['local'],'rb') as source,sftp.open(stage,'ab' if offset else 'wb') as target:
                    target.set_pipelined(True);source.seek(offset)
                    while chunk:=source.read(1024*1024):
                        target.write(chunk);offset+=len(chunk)
                        if time.monotonic()-last>30:
                            print(json.dumps({'file':Path(row['local']).name,'uploaded':offset,'total':row['size']}),flush=True);last=time.monotonic()
            result=remote(row,'publish');assert result['success'];results.append(dict(row,**result));save()
            print(json.dumps(result),flush=True)

if __name__=='__main__':main()
