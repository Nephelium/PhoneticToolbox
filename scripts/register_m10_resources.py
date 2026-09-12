"""Refresh only M10 source attribution and current resource hashes."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def read(path):return json.loads((ROOT/path).read_text('utf-8'))
def write(path,data):(ROOT/path).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
registry=read('third_party/source-registry.json')
sources={s['id']:s for s in registry['sources']}
for sid,paths in {
 'SRC-VTL':['resources/vocal_tract','packages/phonetic_core/src/phonetic_core/vocal_tract','scripts/build_m10_geometry.py'],
 'SRC-THREE':['frontend/public/vocal-tract/vendor'],
 'ASSET-HEAD':['frontend/public/vocal-tract/assets/head.json'],
 'ASSET-NASAL':['frontend/public/vocal-tract/assets/nasal.json'],
 'PKG-SOUNDDEVICE':['desktop/src/ptb_desktop/vocal_tract/audio_output.py'],
}.items():
 s=sources[sid];s['used_at']=list(dict.fromkeys(s.get('used_at',[])+paths));s['m10_migration_evidence']='docs/modules/evidence/M10-migration.json'
 sources[sid]['m10_verification_date']='2026-09-11'
sources['SRC-VTL']['m10_modifications']='API 2.4 binary unchanged. Bridge m10/2: manual tongue root, physical anterior lateral relief, narrow-gap readout and shared tube recomputation. R4 constrains posterior targets against speaker pharynx/body dimensions. Whisper audition +24 dB, native raw synthesis unchanged. TS3 l=-0.6; internal s=0.32 baseline retained but its UI example removed. GPL-3.0-or-later.'
sources['PKG-SOUNDDEVICE']['m10_runtime_version']='0.5.3 in requirements-m10-ui.lock'
sources['ASSET-NASAL']['m10_modifications']='nasal/3 display registration: y += 1.14 - 0.28*x cm, for all vertices and landmarks. Original triangles retained. Both +/-5 mm slice contours checked within head; true outlet closures replace spurious SVG chords. No change to acoustic nasal tube.'
if 'PROJECT-M10' not in sources:
 registry['sources'].append({'id':'PROJECT-M10','title':'PhoneticToolbox 声道界面与平台适配','kind':'project-integration','modules':['M10'],
 'authors':'井韶子 / PhoneticToolbox contributors','urls':{},'license':'Project integration; native-derived adapter GPL-3.0-or-later; third-party components separately listed',
 'actual_included_version':'v3 M10 Windows recording R5 / m10/2','verification_date':'2026-09-11','distribution_status':'local-validation-only',
 'acknowledgement_group':'software','evidence':'M10-source-map.md; migrated source hashes and separately recorded behavior corrections'})
project=next(s for s in registry['sources'] if s['id']=='PROJECT-M10')
project.update(actual_included_version='v3 M10 Windows recording R5 / m10/2',verification_date='2026-09-11',
    m10_modifications='Portable sequence files, preset library, replay cache, shared display geometry, deterministic video export; docs/testing/m10-recording-features-report.md')
for sid,title,authors,url,used_at in [
    ('REF-WEBCODECS','WebCodecs specification','W3C Web Media Working Group','https://www.w3.org/TR/webcodecs/',['frontend/public/vocal-tract/video.js']),
    ('REF-WEBM','WebM Container Guidelines','The WebM Project','https://www.webmproject.org/docs/container/',['desktop/src/ptb_desktop/vocal_tract/video.py']),
]:
    item={'id':sid,'title':title,'authors':authors,'kind':'specification-reference','modules':['M10'],
        'urls':{'specification':url},'license':'Specification reference only; no specification text or external muxer source code redistributed',
        'actual_included_version':'Specification consulted 2026-09-11; runtime encoder from existing Qt WebEngine 6.11.2',
        'observed_upstream_commit':None,'verification_date':'2026-09-11','distribution_status':'reference-only',
        'acknowledgement_group':'software','used_at':used_at,'evidence':'docs/testing/m10-recording-features-report.md'}
    existing=next((s for s in registry['sources'] if s['id']==sid),None)
    if existing is None:registry['sources'].append(item)
    else:existing.update(item)
registry['total_records']=len(registry['sources']);write('third_party/source-registry.json',registry)
manifest=read('contracts/resource-manifest.json')
manifest['resources']=[x for x in manifest['resources'] if not x['path'].startswith(('frontend/public/vocal-tract/','resources/vocal_tract/'))]
for folder in ['frontend/public/vocal-tract','resources/vocal_tract']:
 for p in sorted((ROOT/folder).rglob('*')):
  if not p.is_file():continue
  relative=p.relative_to(ROOT).as_posix()
  sid='SRC-VTL' if folder.startswith('resources/') else 'SRC-THREE' if '/vendor/' in relative else 'ASSET-HEAD' if p.name=='head.json' else 'ASSET-NASAL' if p.name=='nasal.json' else 'PROJECT-M10'
  manifest['resources'].append({'path':relative,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'source_id':sid})
manifest['scope']='P04 shared assets and M10 Windows local native/geometry/UI resources'
# Hash the bundle first, then include its current bytes in the master manifest.
bundle=read('resources/vocal_tract/bundle-manifest.json');bundle['version']='3.0.0-m10.5'
known={item['path'].replace('gui/resources/vocal_tract/','frontend/public/vocal-tract/') for item in bundle['files']}
for folder in ['frontend/public/vocal-tract','resources/vocal_tract']:
 for p in sorted((ROOT/folder).rglob('*')):
  relative=p.relative_to(ROOT).as_posix()
  if p.is_file() and p.name!='bundle-manifest.json' and relative not in known:
   bundle['files'].append({'path':relative});known.add(relative)
for item in bundle['files']:
 item['path']=item['path'].replace('gui/resources/vocal_tract/','frontend/public/vocal-tract/')
 p=ROOT/item['path'];item.update(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
write('resources/vocal_tract/bundle-manifest.json',bundle)
for x in manifest['resources']:
 if x['path'].endswith('bundle-manifest.json'):x['sha256']=hashlib.sha256((ROOT/x['path']).read_bytes()).hexdigest()
write('contracts/resource-manifest.json',manifest)
evidence=read('docs/modules/evidence/M10-migration.json')
for item in evidence['files']:
 assert hashlib.sha256((ROOT/item['source']).read_bytes()).hexdigest()==item['source_sha256'],item['source']
 item['current_sha256']=hashlib.sha256((ROOT/item['target']).read_bytes()).hexdigest()
evidence['behavior_revision']='R5 / native m10/2; built without verification by user request; docs/plans/2026-09-11-m10-onset.md'
write('docs/modules/evidence/M10-migration.json',evidence)
print('Registered M10 resources and source records')
