"""Select installed M03 runtime files for a local-only relocation probe; no installs."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import shutil
import sys
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT=Path(__file__).resolve().parents[1]
SEEDS=('phonetic-core','numpy','scipy','praat-parselmouth','matplotlib','pandas','pydantic')
SKIP={'__pycache__','tests','test','testing','benchmarks','idlelib','turtledemo','ensurepip'}
API=('__init__','models','protocol_version','acoustic_models','acoustic_batch_models','font_models','egg_models')
WORKER=('__init__','egg_child','egg_exports','egg_preview','egg_export_names','egg_runtime','fonts','acoustic_errors')

def runtime_distributions():
    todo=list(SEEDS);found={}
    while todo:
        name=canonicalize_name(todo.pop())
        if name in found:continue
        d=metadata.distribution(name);found[name]=d
        for text in d.requires or []:
            req=Requirement(text)
            if req.marker is None or req.marker.evaluate({'extra':''}):todo.append(req.name)
    return found

def acceptable(path):
    return not (set(part.lower() for part in path.parts)&SKIP) and path.suffix.lower() not in {'.pyc','.pyo','.pdb'} and path.name!='direct_url.json'

def stage(output):
    prefix=Path(sys.prefix).resolve()
    if prefix!=ROOT/'.venv/m03-compatible':raise ValueError('Use the locked M03 compatible interpreter')
    output=output.resolve()
    if not output.is_relative_to(ROOT/'output/validation/m03-runtime') or output.exists():raise ValueError('Use a new owned probe directory')
    output.mkdir(parents=True);payload=output/'payload';payload.mkdir();items={};packages=[]
    def copy(source,target,owner):
        source=source.resolve();target=Path(target)
        if not (source.is_relative_to(prefix) or source.is_relative_to(ROOT)) or target.is_absolute() or '..' in target.parts:raise ValueError('Unsafe inventory path')
        if not source.is_file():raise ValueError('Missing installed file: '+str(source))
        key=target.as_posix()
        if key in items:return
        dest=payload/target;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
        items[key]=dict(path=key,size=dest.stat().st_size,sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),owner=owner)
    for source in prefix.iterdir():
        if source.is_file() and (source.suffix.lower()=='.dll' or source.name in {'python.exe','LICENSE_PYTHON.txt'}):copy(source,Path('runtime')/source.relative_to(prefix),'conda-runtime')
    for folder in ('Lib','DLLs'):
        for source in (prefix/folder).rglob('*'):
            rel=source.relative_to(prefix)
            if source.is_file() and 'site-packages' not in rel.parts and acceptable(rel):copy(source,Path('runtime')/rel,'python-stdlib')
    for source in (prefix/'Library/bin').glob('*.dll'):copy(source,Path('runtime')/source.relative_to(prefix),'conda-native')
    for name,d in sorted(runtime_distributions().items()):
        packages.append(dict(name=name,version=d.version,license=d.metadata.get('License-Expression') or d.metadata.get('License') or 'unknown'))
        for entry in d.files or []:
            source=Path(d.locate_file(entry)).resolve()
            if not source.is_relative_to(prefix/'Lib/site-packages'):continue
            rel=source.relative_to(prefix)
            if not acceptable(rel):continue
            if name=='phonetic-core' and 'phonetic_core' in rel.parts and not ('egg' in rel.parts or rel.name=='__init__.py' and rel.parent.name=='phonetic_core'):continue
            copy(source,Path('runtime')/rel,name)
    conda=[]
    for record in sorted((prefix/'conda-meta').glob('*.json')):
        value=json.loads(record.read_text('utf-8'))
        if not any(('runtime/'+f.replace('\\','/')) in items for f in value.get('files',[])):continue
        copy(record,Path('runtime/conda-meta')/record.name,'conda-record')
        conda.append({k:value.get(k) for k in ('name','version','build','license','url','sha256')})
        for f in value.get('files',[]):
            if 'license' in f.lower() or 'copyright' in f.lower():
                source=prefix/f
                if source.is_file():copy(source,Path('runtime')/f,value['name']+'-license')
    for package,names in [('ptb_api',API),('ptb_worker',WORKER)]:
        for name in names:copy(ROOT/'backend/src'/package/(name+'.py'),Path('backend')/package/(name+'.py'),'application-adapter')
    for name in ('__init__.py','limits.py'):copy(ROOT/'backend/src/ptb_worker/io'/name,Path('backend/ptb_worker/io')/name,'application-adapter')
    copy(ROOT/'backend/src/ptb_worker/assets/DoulosSIL-Regular.ttf','backend/ptb_worker/assets/DoulosSIL-Regular.ttf','ASSET-DOULOS')
    copy(ROOT/'tests/fixtures/m03/EGG-SYN-PCM16.npz','fixture.npz','M03-public-synthetic')
    copy(ROOT/'desktop/experiments/m03_probe_child.py','probe_child.py','application-probe')
    copy(ROOT/'third_party/source-registry.json','sources/source-registry.json','application-source-registry')
    for source in (ROOT/'third_party/licenses').rglob('*'):
        if source.is_file():copy(source,Path('sources/licenses')/source.relative_to(ROOT/'third_party/licenses'),'existing-license-evidence')
    manifest=dict(schema_version='m03-probe/1',local_only=True,redistribution_approved=False,packages=packages,conda=conda,files=list(items.values()))
    (payload/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(output=str(output),files=len(items),bytes=sum(i['size'] for i in items.values()),packages=len(packages))))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);stage(parser.parse_args().output)
