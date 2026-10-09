"""Verify the delivered M13 package, its snapshot and its owned child processes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import psutil
from PyInstaller.archive.readers import CArchiveReader


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe',required=True,type=Path)
    parser.add_argument('--snapshot',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args()
    exe=args.exe.resolve(strict=True);snapshot=args.snapshot.resolve(strict=True);out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    archive=CArchiveReader(str(exe));names={name.replace('\\','/'):name for name in archive.toc}
    files={}
    for directory in ('frontend/dist','backend/src','desktop/src','packages/phonetic_core/src'):
        for source in sorted((snapshot/directory).rglob('*')):
            if not source.is_file():continue
            name=source.relative_to(snapshot).as_posix()
            expected=hashlib.sha256(source.read_bytes()).hexdigest()
            assert name in names,name
            assert hashlib.sha256(archive.extract(names[name])).hexdigest()==expected,name
            files[name]=expected
    css=[n for n in files if n.startswith('frontend/dist/assets/MandarinIpaPage-') and n.endswith('.css')]
    assert len(css)==1,css
    assert '.m13-settings-section>label:not(.font-family-select)' in archive.extract(names[css[0]]).decode('utf-8')
    config=json.loads(archive.extract(names['local-preview.json']))
    audit=dict(success=True,exe=str(exe),bytes=exe.stat().st_size,sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
               exact_snapshot_files=len(files),files=files,config=config)
    (out/'archive-report.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    env=os.environ.copy()
    for key in list(env):
        if key.startswith(('PYTHON','PTB_','QT_')):env.pop(key)
    windows=Path(env.get('SystemRoot','C:/Windows'))
    env['PATH']=os.pathsep.join(map(str,(windows/'System32',windows)))
    seen={}
    with (out/'stdout.log').open('wb') as stdout,(out/'stderr.log').open('wb') as stderr:
        process=subprocess.Popen([str(exe),'--verify-m13-preview',str(out/'results')],
            cwd=tempfile.gettempdir(),env=env,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
        parent=psutil.Process(process.pid);deadline=time.monotonic()+180
        while process.poll() is None:
            try:
                for child in parent.children(recursive=True):seen[(child.pid,child.create_time())]=child
            except psutil.NoSuchProcess:pass
            if time.monotonic()>deadline:
                for child in reversed(list(seen.values())):
                    try:child.kill()  # psutil checks the identity of this owned PID.
                    except psutil.NoSuchProcess:pass
                process.kill();process.wait();raise TimeoutError(str(out))
            time.sleep(.3)
    _,alive=psutil.wait_procs(list(seen.values()),timeout=5)
    report=dict(exit_code=process.returncode,remaining_owned_pids=[p.pid for p in alive],
                observed_children=len(seen),cwd=tempfile.gettempdir(),sanitized_environment=True)
    (out/'process-report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(out=str(out),**report)),flush=True)
    assert process.returncode==0 and not alive,str(out)
    native=json.loads((out/'results/report.json').read_text(encoding='utf-8'))
    assert native['success']
    assert len(native['layouts'])==16
    for name,digest in native['frontend'].items():assert files['frontend/dist/assets/'+name]==digest,name
    print('Verified actual frozen M13, exact source/assets snapshot, PNG and owned process cleanup.',flush=True)


if __name__=='__main__':main()
