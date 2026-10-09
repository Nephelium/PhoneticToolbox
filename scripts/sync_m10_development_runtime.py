"""Synchronize only this repo's M10 Python code into the existing project runtime.

No dependency install, global environment, user profile or other module changes.
Keep the previous bytes and package record in the task's validation directory.
"""
import base64
import csv
import hashlib
import io
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
site=ROOT/'.venv/m09-ui/Lib/site-packages'
source=ROOT/'packages/phonetic_core/src/phonetic_core/vocal_tract'
target=site/'phonetic_core/vocal_tract'
record=site/'phonetic_core-3.0.0a1.dist-info/RECORD'
assert target.is_dir() and record.is_file(),'Existing project runtime required'
backup=ROOT/'output/validation/m10-r11/runtime-before';backup.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def non_voice():return {p.relative_to(site).as_posix():sha(p) for p in (site/'phonetic_core').rglob('*')
                       if p.is_file() and '__pycache__' not in p.parts and 'vocal_tract' not in p.parts}
before=non_voice();changed={}
if not (backup/'RECORD').exists():(backup/'RECORD').write_bytes(record.read_bytes())
for p in source.glob('*.py'):
    dest=target/p.name
    if dest.is_file() and dest.read_bytes()==p.read_bytes():continue
    old=backup/p.name
    if dest.is_file() and not old.exists():old.write_bytes(dest.read_bytes())
    dest.write_bytes(p.read_bytes())
    key=dest.relative_to(site).as_posix()
    changed[key]=('sha256='+base64.urlsafe_b64encode(hashlib.sha256(dest.read_bytes()).digest()).decode().rstrip('='),str(dest.stat().st_size))
rows=list(csv.reader(io.StringIO(record.read_text('utf-8'))));seen=set()
for row in rows:
    key=row[0].replace('\\','/');seen.add(key)
    if key in changed:row[1:]=changed[key]
rows.extend([key,*value] for key,value in changed.items() if key not in seen)
if changed:
    stream=io.StringIO(newline='');csv.writer(stream,lineterminator='\n').writerows(rows);record.write_text(stream.getvalue(),'utf-8')
assert before==non_voice(),'Unexpected change outside M10'
report_path=backup.parent/'runtime-sync.json'
previous=json.loads(report_path.read_text('utf-8')) if report_path.exists() else {}
report={'runtime':str(site.parent.parent),'changed':sorted(set(previous.get('changed',[]))|set(changed)),'changed_this_sync':list(changed),'unchanged_other_module_files':len(before),
        'backup':str(backup),'installed_m10':{p.name:sha(p) for p in target.glob('*.py')}}
report_path.write_text(json.dumps(report,indent=2),'utf-8')
print(json.dumps(report))
