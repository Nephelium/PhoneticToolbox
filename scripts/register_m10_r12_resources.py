"""Refresh only M10 source provenance and resource hashes (R12–R18)."""
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
revision=sys.argv[1] if len(sys.argv)>1 else 'r12'
assert revision in ('r12','r13','r14','r15','r16','r17','r18'),revision
date='2026-10-12' if revision in ('r17','r18') else '2026-10-10' if revision in ('r15','r16') else '2026-10-07'
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
if revision=='r15':
    changes={
        'SRC-VTL':'M10-R15: fixed-capacity native arrays with runtime 65/129/257 cross-section counts and matching centerline exponent. Default 129; 257 option compared against the unmodified R14 257 capture. Real geometry, tube integrals and transfer function follow the selected resolution. Original API DLLs and speakers preserved; GPL-3.0-or-later.',
        'PROJECT-M10':'M10-R15: persisted sampling preference, single voiced/silent keyframe playback/F0/video from 50 ms, continuous silent-frame motion with unchanged ordinary interpolation, and atomic MP4 H.264/AAC export using the existing isolated PyAV runtime. Qt WebCodecs VP8/Opus packets are intermediate only. No physical-device or final-EXE claim.',
    }
    pyav=next((s for s in registry['sources'] if s['id']=='PKG-PYAV'),None)
    if pyav is None:
        pyav={'id':'PKG-PYAV','title':'PyAV','kind':'software-library','modules':['M05','M10'],
               'authors':'PyAV contributors','urls':{'homepage':'https://pyav.basswood-io.com','source':'https://github.com/PyAV-Org/PyAV'},
               'license':'BSD-3-Clause (binding); bundled FFmpeg codecs have separate build-dependent terms',
               'actual_included_version':'16.1.0 in existing M05 isolated runtime',
               'observed_upstream_commit':None,'verification_date':date,'distribution_status':'review-required',
               'acknowledgement_group':'software','evidence':'requirements/requirements-m05.lock; installed av-16.1.0.dist-info/METADATA; docs/testing/2026-10-10-m10-r15-report.md',
               'used_at':['backend/src/ptb_worker/m05_media_export.py','desktop/src/ptb_desktop/vocal_tract/mp4.py']}
        registry['sources'].append(pyav)
    sources['SRC-FFMPEG']['modules']=list(dict.fromkeys(sources['SRC-FFMPEG']['modules']+['M10']))
    sources['SRC-FFMPEG']['used_at']=list(dict.fromkeys(sources['SRC-FFMPEG'].get('used_at',[])+['desktop/src/ptb_desktop/vocal_tract/mp4.py']))
    sources['SRC-FFMPEG']['m10_r15_use']='Existing PyAV FFmpeg libraries encode H.264/AAC offline; system ffmpeg/ffprobe executables are independent QA tools only. Existing exact-binary/license review boundary remains.'
    registry['total_records']=len(registry['sources'])
if revision=='r16':
    changes={
        'SRC-VTL':'M10-R16: anterior side bracing and central groove continue to the tongue tip; positive bracing contacts the native hull, negative relief follows native material stations. Surface, sections and equivalent tube recomputed together. Layout m10/3, geometry method m10/r16. Original API DLLs and speakers preserved; GPL-3.0-or-later.',
        'PROJECT-M10':'M10-R16: open-domain upper/lower air sheets preserve separated lateral channels and are clipped against rigid dental solids even when teeth are hidden. Display tongue underside returns to jaw-relative floor rib 12; this is a geometric attachment reference, not measured frenulum anatomy or tissue mechanics. Version 3 exchange documents record geometry method; legacy imports disclose recomputation. WASAPI shared conversion preserves 48 kHz synthesis and export.',
    }
if revision=='r17':
    changes={
        'SRC-VTL':'M10-R17: bound inward blade depression relative to moving body/tip support (3 mm geometric regularizer), smooth blend, geometric narrow-gap profiles shared by acoustic integration and display and cached per pose; empirical area corrections and genuine closure retained. Method m10/r17, layout m10/3. Original synthesis/analysis DLLs and speakers preserved; GPL-3.0-or-later.',
        'PROJECT-M10':'M10-R17: jaw-relative floor rib 10 and ventral return constrained to the actual inner jaw and lower teeth, with triangle-interior contact and whole displayed tongue subtraction against both dental arches; exact reduced nasal reference section, lower-allocation contact/air sheets, native-response side dragging; viewport PCM waveform detail, automatic/fixed amplitude axes, shared time-axis Ctrl-wheel zoom (0.001-12 seconds), double-click reset. Version 4 documents identify method; v1-v3 imports disclose recomputation. Native frication and pressure dependence verified without changing noise generation or applying output normalization.',
    }
if revision=='r18':
    changes={
        'SRC-VTL':'M10-R18: compact arc-length local dorsal offsets before native hard-tissue contact; material handle station exported; zero offsets preserve R17 geometry digests in six reference poses. Method m10/r18, layout m10/3. Original synthesis DLLs and speakers retained; GPL-3.0-or-later.',
        'PROJECT-M10':'M10-R18: relative surface/underside handles, constrained ventral display and native dorsal acoustics, version 5 exchange with legacy 1-4 imports; favorite builtin/custom shapes; pose-free aspiration intervals (10 ms minimum) on continuous posture motion with exact native excitation boundaries. Underside is a tissue display envelope, not an extra sublingual acoustic branch. No final EXE, physical-device or perceptual claim.',
    }
for sid,note in changes.items():
    sources[sid][f'm10_{revision}_modifications']=note
    sources[sid][f'm10_{revision}_verification_date']=date
    sources[sid][f'm10_{revision}_evidence']=f'docs/testing/{date}-m10-{revision}-report.md'
sources['SRC-VTL']['used_at']=list(dict.fromkeys(sources['SRC-VTL'].get('used_at',[])+[f'resources/vocal_tract/sources/m10_{revision}_patch.py']))
sources['PROJECT-M10']['used_at']=list(dict.fromkeys(sources['PROJECT-M10'].get('used_at',[])+['frontend/public/vocal-tract/rigid-contact.mjs']+(['frontend/public/vocal-tract/airway-mesh.mjs'] if revision=='r16' else [])))
sources['PROJECT-M10']['actual_included_version']=f'v3 M10-{revision.upper()} / geometry layout m10/3'
sources['PROJECT-M10']['verification_date']=date
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
