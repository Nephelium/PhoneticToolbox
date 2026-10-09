"""Relocate existing locked runtimes into a new owned release stage.

No install, environment rewrite or source-data cleanup occurs. M03 uses the
reviewed stage_m03_runtime inventory. M05 is converted from a venv redirector
to a self-contained CPython prefix with its original standard library and wheels.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def allowed(folder,names):
    ignored=[]
    for name in names:
        path=Path(folder)/name
        if path.is_symlink() or getattr(path,'is_junction',lambda:False)():
            raise ValueError('Runtime links require explicit review')
        if name=='__pycache__' or name.endswith(('.pyc','.pyo','.egg-link')) or name.startswith('__editable.') or name=='direct_url.json':ignored.append(name)
    return ignored

def copy(source,target):
    source=Path(source)
    if not source.is_file() or source.is_symlink():raise ValueError('Invalid runtime input')
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)

def egg_build_receipt(target):
    # The small reviewed conda build receipt is required by both EGG and LPC.
    # Preserve the strict scientific fingerprint instead of relaxing it.
    original=ROOT/'.venv/m03-compatible'
    name='scipy-1.16.3-py311hf127856_1.json'
    receipt=original/'conda-meta'/name
    value=json.loads(receipt.read_text('utf8'))
    if value.get('version')!='1.16.3' or value.get('build')!='py311hf127856_1':
        raise ValueError('Unexpected reviewed SciPy build')
    for relative in ('Lib/site-packages/scipy/__init__.py','Lib/site-packages/scipy/__config__.py'):
        if sha(original/relative)!=sha(target/relative):
            raise ValueError('Staged SciPy differs from its build receipt')
    destination=target/'conda-meta'/name
    if destination.exists() and sha(destination)!=sha(receipt):
        raise ValueError('Existing SciPy build receipt differs')
    if not destination.exists():copy(receipt,destination)
    # Scientific public imports also reach helper modules in testing trees.
    # Keep complete matching package files, avoiding brittle name pruning.
    for name in ('numpy','scipy','matplotlib','pandas'):
        source=original/'Lib/site-packages'/name
        destination=target/'Lib/site-packages'/name
        for folder,dirs,files in os.walk(source):
            ignored=set(allowed(folder,dirs+files))
            dirs[:]=[name for name in dirs if name not in ignored]
            for name in files:
                if name in ignored:continue
                path=Path(folder)/name;output=destination/path.relative_to(source)
                if output.exists():
                    if sha(path)!=sha(output):raise ValueError('Scientific runtime file differs from its reviewed source')
                else:copy(path,output)

def m05(target):
    config={line.split('=',1)[0].strip():line.split('=',1)[1].strip()
            for line in (ROOT/'.venv/m05/pyvenv.cfg').read_text('utf8').splitlines() if '=' in line}
    base=Path(config['home']).resolve();site=ROOT/'.venv/m05/Lib/site-packages'
    if not base.is_absolute() or not (base/'python.exe').is_file():raise ValueError('Original M05 CPython base unavailable')
    target.mkdir()
    for source in base.iterdir():
        if source.is_file() and (source.suffix.lower()=='.dll' or source.name in ('python.exe','pythonw.exe','LICENSE_PYTHON.txt')):
            copy(source,target/source.name)
    def std_ignore(folder,names):
        return list(set(allowed(folder,names))|({'site-packages'} if Path(folder)==base/'Lib' else set()))
    shutil.copytree(base/'Lib',target/'Lib',ignore=std_ignore)
    shutil.copytree(base/'DLLs',target/'DLLs',ignore=allowed)
    shutil.copytree(site,target/'Lib/site-packages',ignore=allowed)
    # Follow native standard-library dependencies from the installed base.
    # Wheel-specific native directories remain exactly where their loaders expect.
    import pefile
    candidates={p.name.lower():p for p in (base/'Library/bin').glob('*.dll')}
    queue=[p for p in target.rglob('*') if p.is_file() and p.suffix.lower() in ('.pyd','.dll','.exe')]
    seen=set();copied=set()
    while queue:
        source=queue.pop()
        if source in seen:continue
        seen.add(source)
        try:
            with pefile.PE(str(source),fast_load=True) as pe:
                pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_IMPORT']])
                imports=[entry.dll.decode('ascii').lower() for entry in getattr(pe,'DIRECTORY_ENTRY_IMPORT',[])]
        except pefile.PEFormatError:continue
        for name in imports:
            if name in candidates and name not in copied:
                destination=target/'DLLs'/candidates[name].name
                if not destination.exists():copy(candidates[name],destination)
                copied.add(name);queue.append(destination)
    return dict(base_version=config.get('version'),native_dependencies=sorted(copied))

def mfa(target):
    registry=json.loads((ROOT/'output/m11c-028b881d/registry.json').read_text('utf8'))
    runtime=registry['runtimes'][0]
    source=Path(runtime['path'])
    # Preserve all fingerprinted runtime files and their license notices.
    shutil.copytree(source,target/'runtime',ignore=shutil.ignore_patterns('__pycache__'))
    receipt=Path(runtime['receipt']);copy(receipt,target/'receipt.json')
    bundled_runtime={**runtime,'path':'runtime','receipt':'receipt.json','origin':'builtin'}
    models=[]
    for model in registry['models']:
        row={**model,'origin':'builtin'}
        for role in ('model','dictionary'):
            path=Path(model[role]);relative=f'resources/{role}/{model[role+"_sha256"]}{path.suffix}'
            copy(path,target/relative)
            if sha(target/relative)!=model[role+'_sha256']:raise ValueError('MFA model identity changed')
            row[role]=relative
        row['source']='bundled model and dictionary; see component notices'
        models.append(row)
    (target/'registry-bundled.json').write_text(json.dumps(dict(schema='m11-registry/1',runtimes=[bundled_runtime],models=models),ensure_ascii=False,indent=2),'utf8')
    return dict(runtime_id=runtime['id'],fingerprint=runtime['fingerprint'],models=len(models))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True,type=Path);parser.add_argument('--m03-stage',required=True,type=Path);parser.add_argument('--verify-only',action='store_true')
    options=parser.parse_args();output=options.output.resolve()
    if not output.is_relative_to(ROOT/'output/release-staging') or (output.exists() and not options.verify_only):raise ValueError('Use a new release-staging directory')
    source=options.m03_stage.resolve()
    if not source.is_relative_to(ROOT/'output/validation/m03-runtime') or not (source/'payload/manifest.json').is_file():raise ValueError('Expected inventoried M03 stage')
    if not options.verify_only:
        output.mkdir(parents=True);print('staging M03',flush=True)
        shutil.copytree(source/'payload/runtime',output/'egg',ignore=allowed)
        print('staging M05',flush=True);info_m05=m05(output/'m05')
        print('staging MFA',flush=True);info_mfa=mfa(output/'mfa')
    else:
        # Recheck a retained failed stage without copying or deleting it.
        if not all((output/name/'python.exe').is_file() for name in ('egg','m05')):raise ValueError('Incomplete stage')
        registry=json.loads((output/'mfa/registry-bundled.json').read_text('utf8'))
        info_m05=dict(verification_only=True)
        info_mfa=dict(runtime_id=registry['runtimes'][0]['id'],fingerprint=registry['runtimes'][0]['fingerprint'],models=len(registry['models']))
    egg_build_receipt(output/'egg')
    report=dict(schema='ptb-runtime-stage/1',m05=info_m05,mfa=info_mfa,runtimes={})
    for name in ('egg','m05'):
        executable=output/name/'python.exe'
        report['runtimes'][name]=dict(path=name+'/python.exe',sha256=sha(executable))
    environment={key:value for key,value in os.environ.items() if key not in ('PYTHONHOME','PYTHONPATH')}
    environment['PATH']=os.pathsep.join([os.environ['SystemRoot']+'/System32',os.environ['SystemRoot']])
    for name,expression in [('egg','import numpy,scipy,parselmouth,matplotlib,pandas; from scipy.io import wavfile; import numpy.testing; print(numpy.__version__,scipy.__version__)'),
                            ('m05','import cv2,mediapipe,av,numpy,scipy,matplotlib; print(cv2.__version__,mediapipe.__version__,av.__version__)')]:
        checked=subprocess.run([str(output/name/'python.exe'),'-I','-B','-c',expression],cwd=output,env=environment,capture_output=True,text=True,timeout=90)
        report['runtimes'][name].update(import_returncode=checked.returncode,import_output=checked.stdout.strip(),diagnostic=checked.stderr[-3000:])
        if checked.returncode:
            (output/'stage-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf8')
            raise RuntimeError('Relocated runtime import failed: '+name+'\n'+checked.stderr[-3000:])
    (output/'stage-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf8')
    print(json.dumps(dict(output=str(output),imports='passed')),flush=True)

if __name__=='__main__':main()
