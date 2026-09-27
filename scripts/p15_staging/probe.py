"""Read-only P15 inventory and content seal. No imports of recovery or migrations."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone

from host import release_files


def command(argv):
    try:
        p=subprocess.run(argv,stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=8)
        return {'exit_code':p.returncode,'stdout':p.stdout[:8192]}
    except (OSError,subprocess.TimeoutExpired): return {'available':False}


def inventory(root):
    result={'schema':'p15-inventory/1','at':datetime.now(timezone.utc).isoformat(),
        'platform':platform.platform(),'python':sys.version.split()[0],
        'python_executable':sys.executable,'deployed':False,'database':'not_inspected',
        'dns_tls':'not_verified','laboratory_linux':'not_tested'}
    if sys.platform=='linux':
        result['uid']=os.getuid()
        result['pid1']=Path('/proc/1/comm').read_text().strip()
        result['systemd_run']=shutil.which('systemd-run')
        result['systemctl']=shutil.which('systemctl')
        result['user_manager']=command(['systemctl','--user','show','--property=Version,ControlGroup'])
        cg=Path('/sys/fs/cgroup')
        result['cgroup_v2']=(cg/'cgroup.controllers').is_file()
        result['cgroup_root_writable']=os.access(cg,os.W_OK)
        result['current_cgroup']=Path('/proc/self/cgroup').read_text().strip()
        result['controllers']=(cg/'cgroup.controllers').read_text().strip() if result['cgroup_v2'] else ''
        result['hard_limits_verified']=False
        result['boundary_candidate']=bool(result['systemd_run'] and result['systemctl'] and
            result['user_manager'].get('exit_code')==0)
        result['disk_free_bytes']=shutil.disk_usage(Path.home()).free
    files=release_files(root)
    result['files']={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in sorted(files)}
    p11=root/'output/validation/p11-perf-20260927/delivery.json'
    if p11.is_file():
        d=json.loads(p11.read_text('utf-8'))
        changed=[]
        for name,digest in d.get('files',{}).items():
            p=root/name
            if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest: changed.append(name)
        result['p11_delivery']={'sha256':hashlib.sha256(p11.read_bytes()).hexdigest(),
            'status':d.get('status'),'server':d.get('server'), 'changed_files':changed,
            'integrated_release_inherits_perf':False}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seal-release',action='store_true')
    a=p.parse_args();root=a.root.resolve()
    result=inventory(root)
    if a.seal_release:
        result={'schema':'p15-release/1','files':result['files'],'verified':False,
            'note':'Content seal only. Not a runtime receipt, build proof, permission or deployment approval.'}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf-8',newline='\n') as f: json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps({'written':str(a.output),'deployed':False}))


if __name__=='__main__': main()
