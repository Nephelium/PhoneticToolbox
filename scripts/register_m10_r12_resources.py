"""Refresh only M10 source provenance and resource hashes (R12/R13/R14)."""
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
revision=sys.argv[1] if len(sys.argv)>1 else 'r12'
assert revision in ('r12','r13','r14'),revision
def read(p):return json.loads((ROOT/p).read_text('utf-8'))
def write(p,obj):(ROOT/p).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n','utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

registry=read('third_party/source-registry.json')
sources={s['id']:s for s in registry['sources']}
changes={
    'SRC-VTL':'M10-R12: context-checked native blade depression, height-aware tongue-tip retraction, posterior/upper circle tangent selection without angle wrap, and anterior ventral return for curled poses. Hard-tissue/floor constraints remain; native geometry and tube are recomputed together. Original API DLLs and speakers preserved; GPL-3.0-or-later.',
    'PROJECT-M10':'M10-R12: larynx handle always available in 2-D/3-D, rigid native dental solids exclude reconstructed tongue underside and display caps, exposed velum outline omits internal tissue closure edges, and curled underside avoids self intersection. No soft-tissue mechanics or perceptual validation claim.',
}
if revision=='r13':
    changes={
        'SRC-VTL':'M10-R13: stable auxiliary tip height, tangent-continuous depressed blade, curvature-weighted native ribs for a rounded raised tip, denser circular target and tangent-aligned ventral return. Native surfaces and tube recalculated together; topology and file formats unchanged. Original API DLLs/speakers preserved; GPL-3.0-or-later.',
        'PROJECT-M10':'M10-R13: dorsal velum attachment reaches the anterior velopharyngeal port through two tangent-matched segments, removing the false pocket above the uvula. Same 3-D tissue drives the sagittal mask. Original nasal mesh and nasal acoustic parameters preserved.',
    }
if revision=='r14':
    changes={
        'SRC-VTL':'M10-R14: continuous raised/curled tip radius compensation, posterior arc attachment for finite tip thickness, native material blade rib export, and 257 longitudinal sections (96 transverse samples). Same native geometry feeds tube recomputation. Reference 129-section build retained for measured timing/area comparisons. Original API DLLs/speakers preserved; GPL-3.0-or-later.',
        'PROJECT-M10':'M10-R14: blade handle attaches to the actual native meridian in 2-D/3-D, with midpoint drag gain; exact anterior oral tissue cuts supplement the sampled air reference to remove false grey slivers. Dynamic section counts in controls, charts and runtime. Finite parametric geometry, not a muscle mechanics simulation.',
    }
for sid,note in changes.items():
    sources[sid][f'm10_{revision}_modifications']=note
    sources[sid][f'm10_{revision}_verification_date']='2026-10-07'
    sources[sid][f'm10_{revision}_evidence']=f'docs/testing/2026-10-07-m10-{revision}-report.md'
sources['SRC-VTL']['used_at']=list(dict.fromkeys(sources['SRC-VTL'].get('used_at',[])+[f'resources/vocal_tract/sources/m10_{revision}_patch.py']))
sources['PROJECT-M10']['used_at']=list(dict.fromkeys(sources['PROJECT-M10'].get('used_at',[])+['frontend/public/vocal-tract/rigid-contact.mjs']))
sources['PROJECT-M10']['actual_included_version']=f'v3 M10-{revision.upper()} / geometry layout m10/3'
sources['PROJECT-M10']['verification_date']='2026-10-07'
write('third_party/source-registry.json',registry)
files=[p for folder in ['frontend/public/vocal-tract','resources/vocal_tract'] for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.pyo')]
bundle=read('resources/vocal_tract/bundle-manifest.json');bundle['version']=f'3.0.0-m10.{revision}'
items={i['path']:i for i in bundle['files']}
for p in sorted(files):
    if p.name=='bundle-manifest.json':continue
    rel=p.relative_to(ROOT).as_posix()
    if rel not in items:items[rel]={'path':rel};bundle['files'].append(items[rel])
    items[rel].update(bytes=p.stat().st_size,sha256=sha(p))
write('resources/vocal_tract/bundle-manifest.json',bundle)
manifest=read('contracts/resource-manifest.json');items={i['path']:i for i in manifest['resources']}
for p in sorted(files):
    rel=p.relative_to(ROOT).as_posix()
    if rel not in items:
        items[rel]={'path':rel,'source_id':'SRC-VTL' if rel.startswith('resources/') else 'PROJECT-M10'};manifest['resources'].append(items[rel])
    items[rel]['sha256']=sha(p)
write('contracts/resource-manifest.json',manifest)
print(json.dumps({'resources':len(files),'bundle':len(bundle['files']),'sources':list(changes)}))
