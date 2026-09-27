"""Build and import a LOCAL candidate from an explicitly selected existing prefix.

No publication, deletion, downloads, dependency installation or global activation.
The separately generated manifest is a local audit, NOT publisher authenticity.
"""
import argparse
import json
import os
from pathlib import Path
import time
from uuid import uuid4
import zipfile
from ptb_worker.mfa.components import digest,atomic_json,ComponentManager,no_links
from ptb_worker.mfa.runtime import resolve_runtime
from ptb_worker.mfa.probe import register,publish_registration


def main():
    p=argparse.ArgumentParser();p.add_argument('--runtime',required=True);p.add_argument('--model',required=True);p.add_argument('--dictionary',required=True)
    a=p.parse_args();runtime,_=resolve_runtime(a.runtime)
    out=Path(__file__).resolve().parents[1]/'output/validation/m11'/('component-'+uuid4().hex)
    out.mkdir(parents=True);print(out,flush=True)
    os.environ['PTB_M11_COMPONENT_ROOT']=str(out/'installed')
    report=dict(success=False,scope='local Windows optional candidate; not a public release')
    try:
        started=time.monotonic();records=[];archive=out/'mfa-3.3.8-windows-x86_64-candidate.zip'
        paths=sorted(p for p in runtime.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
        with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
            for n,path in enumerate(paths):
                no_links(path);relative=path.relative_to(runtime).as_posix();sha=digest(path)
                records.append(dict(path=relative,bytes=path.stat().st_size,sha256=sha));z.write(path,relative)
                if n%5000==0:print(f'packed {n}/{len(paths)}',flush=True)
        dependencies=[]
        for path in (runtime/'conda-meta').glob('*.json'):
            d=json.loads(path.read_text(encoding='utf8'));dependencies.append({k:d.get(k) for k in ('name','version','build','url','sha256','license')})
        manifest=dict(schema='m11-component/1',id='mfa-338-windows-local-candidate',platform='windows',arch='x86_64',mfa_version='3.3.8',
            source='user-selected existing auto_alignment; locally measured candidate, no publisher authenticity claim',
            sha256=digest(archive),download_bytes=archive.stat().st_size,installed_bytes=sum(r['bytes'] for r in records),dependencies=dependencies,files=records)
        atomic_json(out/'manifest.json',manifest)
        atomic_json(out/'local-manifest-digest.json',dict(sha256=digest(out/'manifest.json'),trust='local separately audited source; not internet publisher proof'))
        report.update(download_bytes=manifest['download_bytes'],installed_bytes=manifest['installed_bytes'],file_count=len(records),build_seconds=time.monotonic()-started)
        prepared={};started=time.monotonic()
        def check(target):
            prepared.update(register(target,a.model,a.dictionary,publish=False))
            return dict(success=True,receipt=prepared['receipt'])
        target=ComponentManager(out/'installed').import_archive(archive,manifest,check)
        publish_registration(prepared,out/'installed')
        report.update(success=True,installed_path=str(target),import_seconds=time.monotonic()-started,receipt=prepared['receipt'])
    finally:
        atomic_json(out/'report.json',report)
        print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
