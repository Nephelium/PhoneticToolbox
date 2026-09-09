"""M01-A independent capture and explicit non-overwriting public freeze."""
import argparse
import datetime
import importlib.metadata
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

from baseline_support import RECIPES, compare, create_fixture, load_json, sha, write_json
from m01_baseline_support import cases

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'output/validation/m01'


def freeze(folder):
    folder=folder.resolve()
    if not folder.is_relative_to(OUTPUT): raise ValueError('Capture must belong to M01 output')
    report=load_json(folder/'capture-summary.json')
    if len(report['cases']) != len(cases()) or {c['id'] for c in report['cases']} != {c['id'] for c in cases()}:
        raise ValueError('Only a complete reviewed capture can be frozen')
    target=ROOT/'tests/fixtures/m01'
    if target.exists(): raise ValueError('Refusing to replace an existing M01 baseline')
    # Read/validate everything before creating any frozen artifact.
    staged=[]
    for row in report['cases']:
        if row['repeat_count']!=2 or row['repeat_differences']: raise ValueError('Repeat not verified')
        source=(ROOT/row['capture_path']).resolve() if row.get('capture_path') else folder/row['id']/'1/result.json.gz'
        if not source.is_relative_to(OUTPUT): raise ValueError('Reference capture outside M01 output')
        if sha(source)!=row['sha256']: raise ValueError('Capture changed since verification')
        captured=load_json(source)
        if captured['producer']['root_kind']!='original_v2' or captured['case_id']!=row['id']:
            raise ValueError('Unexpected producer or case ID')
        raw=source.read_bytes()
        staged.append((row,raw))
    target.mkdir(parents=True)
    rows=[]
    for row,raw in staged:
        destination=target/(row['id']+'.json.gz')
        with destination.open('xb') as f: f.write(raw)
        rows.append({k:v for k,v in dict(row,baseline_path=destination.relative_to(ROOT).as_posix()).items() if k!='capture_path'})
    write_json(target/'manifest.json',{'schema_version':1,'task':'M01-A',
        'producer':'original-v2-conda-process','v3_algorithm_parity':'not_implemented',
        'privacy':'public synthetic only','cases':rows})


def audit_environment(reference_python, source, folder):
    old=load_json(ROOT/'tests/fixtures/golden/SYN-VOWEL-44100.json.gz')['producer']['installed_dependencies']
    code="""import importlib.metadata,json
from packaging.requirements import Requirement
names=['numpy','scipy','pandas','praat-parselmouth','openpyxl','xlsxwriter']
out={}
for name in names:
 try:
  d=importlib.metadata.distribution(name)
  active=[Requirement(x).name for x in d.requires or [] if not Requirement(x).marker or Requirement(x).marker.evaluate({'extra':''})]
  out[name]={'version':d.version,'requires':d.requires or [],'active_requires':active,'license':d.metadata.get('License'),'license_expression':d.metadata.get('License-Expression'),'home_page':d.metadata.get('Home-page'),'project_urls':d.metadata.get_all('Project-URL') or []}
  names.extend(n.lower().replace('_','-') for n in active if n.lower().replace('_','-') not in names)
 except importlib.metadata.PackageNotFoundError: out[name]=None
print(json.dumps(out))
"""
    run=subprocess.run([str(reference_python),'-B','-X','utf8','-c',code],capture_output=True,text=True,encoding='utf-8',check=True)
    current=json.loads(run.stdout)
    installed={d.metadata['Name'].lower().replace('_','-'):d.version for d in importlib.metadata.distributions()}
    normalized_old={k.lower().replace('_','-'):v for k,v in old.items()}
    for name,data in current.items():
        if data: data['p03_version']=normalized_old.get(name); data['matches_p03']=data['version']==normalized_old.get(name)
    cache_roots=[ROOT/'.venv', Path(os.environ.get('LOCALAPPDATA',''))/'pip/Cache/wheels',
                 Path(os.environ.get('LOCALAPPDATA',''))/'uv/cache/wheels-v5']
    wheels=[]
    # Only known runtime/package cache locations, never a user-directory recursive scan.
    for index,cache in enumerate(cache_roots):
        if cache.exists():
            for p in cache.rglob('*.whl'):
                if any(p.name.lower().replace('_','-').startswith(n+'-') for n in current):
                    wheels.append({'cache_scope':index,'filename':p.name,'size_bytes':p.stat().st_size,
                                   'valid_wheel_zip':zipfile.is_zipfile(p),'sha256':sha(p)})
    report={'platform':sys.platform,'python_version':sys.version.split()[0],
        'reference_packages':current,'v3_versions':{n:installed.get(n) for n in current},
        'wheel_candidates':wheels,'wheel_search_scope':['project .venv','pip wheels cache','uv wheels-v5 cache'],
        'not_installed_or_locked_this_turn':True,'no_cross_platform_claim':True}
    write_json(folder/'environment-audit.json',report)
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--case',action='append')
    parser.add_argument('--reuse-from',type=Path,help='Reuse individually verified unchanged cases, rerun --case selections')
    parser.add_argument('--freeze-from',type=Path,help='Explicitly freeze a reviewed complete capture; never overwrite')
    args=parser.parse_args()
    if args.freeze_from:
        freeze(args.freeze_from); print('M01 synthetic baseline frozen without overwriting P03'); return
    selected=[c for c in cases() if not args.case or c['id'] in args.case]
    if not selected or (args.case and set(args.case)!={c['id'] for c in selected}): raise ValueError('Unknown case')
    evidence=load_json(ROOT/'docs/baseline/local-evidence.json')
    source=Path(evidence['baseline']['root']).resolve()
    context=load_json(OUTPUT/'context-before-plan.json')
    reference_python=Path(context['v2_environment'])/'python.exe'
    inventory=load_json(ROOT/'docs/modules/evidence/M01-parameter-settings.json')
    for row in inventory['sources']:
        assert sha(source/row['path'])==row['sha256'], 'v2 source changed; review baseline before capture'
    run_id=datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    output=OUTPUT/run_id; output.mkdir(parents=True,exist_ok=False)
    input_dir=output/'inputs'; input_dir.mkdir()
    harmonic=create_fixture(input_dir,RECIPES[0]); empty=create_fixture(input_dir,next(r for r in RECIPES if r['id']=='SYN-EMPTY'))
    audit_environment(reference_python,source,output)
    print('Capture output: '+output.relative_to(ROOT).as_posix(),flush=True)
    rows=[]
    if args.reuse_from:
        previous=args.reuse_from.resolve()
        if not previous.is_relative_to(OUTPUT) or not args.case: raise ValueError('Reuse requires M01 capture and explicit changed cases')
        specifications={c['id']:c for c in cases()}
        for row in load_json(previous/'capture-summary.json')['cases']:
            if row['id'] in args.case: continue
            first=(ROOT/row['capture_path']).resolve() if row.get('capture_path') else previous/row['id']/'1/result.json.gz'
            if not first.is_relative_to(OUTPUT): raise ValueError('Reused case outside M01 output')
            old_request=load_json(first.parent/'request.json')
            spec={k:v for k,v in old_request.items() if k not in {'source_root','output_dir','input'}}
            if spec!=specifications.get(row['id']): raise ValueError('Reused recipe changed')
            second=first.parent.parent/'2/result.json.gz'
            if sha(first)!=row['sha256'] or compare(load_json(first)['scientific'],load_json(second)['scientific']):
                raise ValueError('Reused capture changed or is not repeatable')
            rows.append(dict(row,capture_path=first.relative_to(ROOT).as_posix()))
    for case in selected:
        captured=[]
        for repeat in [1,2]:
            folder=output/case['id']/str(repeat); folder.mkdir(parents=True)
            temporary=folder/'tmp'; temporary.mkdir()
            request=dict(case,source_root=str(source),output_dir=str(folder),input=str(empty if case.get('empty') else harmonic))
            request_file=folder/'request.json'; write_json(request_file,request)
            env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONUTF8':'1','TEMP':str(temporary),'TMP':str(temporary),
                 'MPLCONFIGDIR':str(temporary/'matplotlib'),'QT_QPA_PLATFORM':'offscreen'}
            prefix=reference_python.parent
            env['PATH']=os.pathsep.join([str(prefix),str(prefix/'Library/bin'),str(prefix/'Scripts'),env.get('PATH','')])
            command=[str(reference_python),'-B','-X','utf8','-X','faulthandler',str(ROOT/'scripts/m01_baseline_worker.py'),str(request_file)]
            with (folder/'worker.log').open('w',encoding='utf-8') as log:
                result=subprocess.run(command,cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=240)
            if result.returncode: raise RuntimeError(f'{case["id"]} repeat {repeat}: exit {result.returncode}; inspect its worker.log')
            captured.append(load_json(folder/'result.json.gz'))
            print(f'{case["id"]} repeat {repeat}: captured',flush=True)
        differences=compare(captured[0]['scientific'],captured[1]['scientific'])
        if differences: raise RuntimeError(f'{case["id"]}: repeat differences {differences}')
        rows.append({'id':case['id'],'sha256':sha(output/case['id']/'1/result.json.gz'),
            'repeat_count':2,'repeat_differences':differences,
            'capture_path':(output/case['id']/'1/result.json.gz').relative_to(ROOT).as_posix()})
        write_json(output/'progress.json',{'run_id':run_id,'cases':rows})
    for row in inventory['sources']: assert sha(source/row['path'])==row['sha256']
    report={'run_id':run_id,'cases':rows,'source_files_unchanged':len(inventory['sources']),
            'status':'repeat_verified','v3_algorithm_parity':'not_implemented',
            'new_case_count':len(selected),'reused_case_count':len(rows)-len(selected)}
    write_json(output/'capture-summary.json',report)
    write_json(OUTPUT/'latest-m01-capture.json',report)
    print(json.dumps({'run_id':run_id,'cases':len(rows),'repeat_verified':True}),flush=True)


if __name__=='__main__': main()
