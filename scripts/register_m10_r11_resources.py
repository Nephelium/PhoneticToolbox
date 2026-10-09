"""Refresh M10-R11 evidence/hashes without rewriting other source records."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def read(path):return json.loads((ROOT/path).read_text('utf-8'))
def write(path,value):(ROOT/path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n','utf-8')
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

registry=read('third_party/source-registry.json')
sources={s['id']:s for s in registry['sources']}
changes={
 'SRC-VTL':'M10-R11 / m10/3: original API DLL and speaker preserved; context-checked native tongue-tip/blade constraints, independent inferior larynx offset with fixed hyoid/epiglottis attachments, rigid uvula-to-tongue contact with flexible palatal attachment, refreshed contour/intersection caches, and zero area for fully sealed cross-sections after empirical corrections. GPL-3.0-or-later; scripts/build_m10_geometry.py and resources/vocal_tract/sources/m10_r11_patch.py.',
 'PROJECT-M10':'M10-R11: continuous nasal branch registration; fixed-size native uvula body and native dorsal tongue triangles; background-colored soft palate in sagittal view; half/full 3-D model; geometric and acoustic area readouts; independent larynx through history, presets, keyframes, synthesis and version-2 interchange. docs/testing/2026-10-06-m10-r11-report.md.',
 'ASSET-NASAL':'M10-R11: original nasal/3 affine registration and mesh unchanged. The display connector uses continuous native branch distance, welded smooth lofts with consistent anterior/posterior correspondence and a unified oral-to-nasal +5 mm reference section. No individual MRI segmentation or recalibration of the native acoustic nasal tube.',
}
for sid,change in changes.items():
    sources[sid]['m10_r11_modifications']=change
    sources[sid]['m10_r11_verification_date']='2026-10-06'
    sources[sid]['m10_r11_evidence']='docs/testing/2026-10-06-m10-r11-report.md'
sources['PROJECT-M10']['actual_included_version']='v3 M10-R11 / geometry m10/3'
sources['PROJECT-M10']['verification_date']='2026-10-06'
sources['SRC-VTL']['used_at']=list(dict.fromkeys(sources['SRC-VTL'].get('used_at',[])+['resources/vocal_tract/sources/m10_r11_patch.py']))
write('third_party/source-registry.json',registry)

files=[]
for folder in ['frontend/public/vocal-tract','resources/vocal_tract']:
    files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.pyo'))
bundle=read('resources/vocal_tract/bundle-manifest.json');bundle['version']='3.0.0-m10.r11'
by_path={item['path']:item for item in bundle['files']}
for p in sorted(files):
    relative=p.relative_to(ROOT).as_posix()
    if p.name=='bundle-manifest.json':continue
    if relative not in by_path:
        item={'path':relative};bundle['files'].append(item);by_path[relative]=item
    by_path[relative].update(bytes=p.stat().st_size,sha256=digest(p))
write('resources/vocal_tract/bundle-manifest.json',bundle)
manifest=read('contracts/resource-manifest.json');by_path={item['path']:item for item in manifest['resources']}
for p in sorted(files):
    relative=p.relative_to(ROOT).as_posix()
    if relative not in by_path:
        sid='SRC-VTL' if relative.startswith('resources/') else 'PROJECT-M10'
        item={'path':relative,'source_id':sid};manifest['resources'].append(item);by_path[relative]=item
    by_path[relative]['sha256']=digest(p)
write('contracts/resource-manifest.json',manifest)
print(f'M10-R11: {len(files)} resource hashes and {len(changes)} source records updated')
