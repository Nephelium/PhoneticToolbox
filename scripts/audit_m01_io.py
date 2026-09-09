"""Record C additions without regenerating the historical B dependency snapshot."""
from pathlib import Path
import hashlib
import importlib.metadata as metadata
import json
import sqlite3

ROOT=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    path=ROOT/'third_party/source-registry.json';registry=json.loads(path.read_text('utf-8'))
    sources={s['id']:s for s in registry['sources']};packages=[]
    used=['backend/src/ptb_worker/io/parameter_exports.py','requirements-m01-io.lock']
    for name in ('openpyxl','et-xmlfile'):
        d=metadata.distribution(name);sid='M01-PY-'+name.upper()
        code_files=[f for f in d.files or [] if str(f).endswith('.py') and d.locate_file(f).is_file()]
        inventory={'name':name,'version':d.version,'license':d.metadata.get('License'),
            'requires':d.requires or [],'source_id':sid,
            'installed_python_files':[{'path':str(f).replace('\\','/'),'sha256':sha(d.locate_file(f))} for f in code_files]}
        packages.append(inventory)
        sources[sid]={'id':sid,'title':name,'kind':'dependency','modules':['M01'],
            'authors':d.metadata.get('Author') or 'Not recorded in installed metadata',
            'actual_included_version':d.version,'license':d.metadata.get('License') or 'unknown',
            'urls':{'release':f'https://pypi.org/project/{name}/{d.version}/'},
            'verification_date':'2026-09-09','distribution_status':'local-development-locked; redistribution review pending',
            'acknowledgement_group':'software','used_at':used,
            'relationship':'Unmodified dependency; project-owned bounded writer adapter uses pinned internal writer API',
            'evidence':'third_party/m01-io-inventory.json'}
    references=[
        ('M01-WIN32','Windows Job Objects and named pipes','Microsoft',
         {'process':'https://learn.microsoft.com/en-us/windows/win32/procthread/process-creation-flags',
          'job':'https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects',
          'membership':'https://learn.microsoft.com/en-us/windows/win32/api/jobapi/nf-jobapi-isprocessinjob',
          'pipe':'https://learn.microsoft.com/en-us/windows/win32/ipc/named-pipe-type-read-and-wait-modes'},
         ['backend/src/ptb_worker/native/windows.py']),
        ('M01-WAVE','RIFF WAVE / WAVEFORMATEXTENSIBLE','Microsoft',
         {'documentation':'https://learn.microsoft.com/en-us/windows/win32/api/mmreg/ns-mmreg-waveformatextensible'},
         ['backend/src/ptb_worker/io/audio.py']),
        ('M01-PICKLE','Python 3.11 restricted pickle loading','Python Software Foundation',
         {'documentation':'https://docs.python.org/3.11/library/pickle.html'},['backend/src/ptb_worker/io/lip.py'])]
    for sid,title,authors,urls,paths in references:
        sources[sid]={'id':sid,'title':title,'kind':'documentation-reference','modules':['M01'],'authors':authors,
            'actual_included_version':'Official documentation consulted 2026-09-09; Windows / CPython 3.11.14 APIs tested locally',
            'license':'Documentation referenced, no third-party document/code redistributed',
            'urls':urls,'verification_date':'2026-09-09','distribution_status':'reference only',
            'acknowledgement_group':'software','used_at':paths,'relationship':'API/format reference; project-owned adapter implementation'}
    additions={
        'SRC-REAPER':['backend/src/ptb_worker/native/reaper.py','resources/manifests/acoustic.json'],
        'SRC-PRAAT':['packages/phonetic_core/src/phonetic_core/acoustic/textgrid.py'],
        'M01-PY-SCIPY':['backend/src/ptb_worker/io/audio.py'],
        'P06-SQLITE':['backend/src/ptb_worker/io/parameter_exports.py']}
    for sid,paths in additions.items():
        source=sources[sid];source['used_at']=list(dict.fromkeys(source.get('used_at',[])+paths))
        source['modules']=list(dict.fromkeys(source['modules']+['M01']))
        source['m01_io_evidence']='docs/testing/m01-io-report.md'
    environment=[]
    for d in sorted(metadata.distributions(),key=lambda d:d.metadata['Name'].lower()):
        name=d.metadata['Name'].lower().replace('_','-')
        if name in ('phonetic-core','ptb-api'):continue
        matches=[s for s in sources.values() if s.get('title','').lower().replace('_','-')==name and d.version in str(s.get('actual_included_version',''))]
        if not matches:raise ValueError('Unregistered installed dependency: '+name)
        match=next((s for s in matches if s['id'].startswith('M01-PY-')),matches[0])
        match['used_at']=list(dict.fromkeys(match.get('used_at',[])+['requirements-m01-io.lock']))
        environment.append({'name':name,'version':d.version,'source_id':match['id']})
    registry['sources']=list(sources.values());registry['total_records']=len(sources);dump(path,registry)
    dump(ROOT/'third_party/m01-io-inventory.json',{'task':'M01-C','python':'3.11.14','sqlite':sqlite3.sqlite_version,
        'new_runtime_packages':packages,'all_locked_packages':environment,'science_runtime':'requirements-m01-science.lock',
        'combined_lock_sha256':sha(ROOT/'requirements-m01-io.lock'),
        'note':'B inventory remains historical. Installed wheel metadata has MIT labels but no standalone license file; redistribution audit remains open.'})
    print(json.dumps({'sources':len(sources),'new_runtime_packages':2,'sqlite':sqlite3.sqlite_version}))


if __name__=='__main__':main()
