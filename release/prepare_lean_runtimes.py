"""Prepare new, tested runtime subsets without modifying source environments.

This is a packaging preparation tool, not an installer or dependency downloader.
The EGG/LPC MKL dispatcher and numerical binaries remain byte-identical. M05's
unused JAX/SciPy and development files are omitted only in the new destination.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
M05_PACKAGES = {
    'av', 'av.libs', 'numpy', 'numpy.libs', 'cv2', 'mediapipe', 'absl',
    'attr', 'attrs', 'google', 'flatbuffers', 'matplotlib', 'mpl_toolkits',
    'PIL', 'pillow.libs', 'fontTools', 'contourpy', 'cycler', 'kiwisolver',
    'dateutil', 'packaging', 'pyparsing', 'six.py', 'pylab.py', 'sounddevice.py',
    '_sounddevice_data', 'cffi', 'pycparser',
}
M05_DISTRIBUTIONS = {
    'av', 'numpy', 'opencv-contrib-python', 'mediapipe', 'absl-py', 'attrs',
    'protobuf', 'flatbuffers', 'matplotlib', 'pillow', 'fonttools', 'contourpy',
    'cycler', 'kiwisolver', 'python-dateutil', 'packaging', 'pyparsing', 'six',
    'sounddevice', 'cffi', 'pycparser',
}
DEVELOPMENT_SUFFIXES = {'.pdb', '.lib', '.a', '.obj', '.o', '.h', '.hpp', '.pxd', '.pyx'}
# numpy.testing is an imported public API, including in SciPy's array API
# compatibility layer. Its nested tests can be omitted, but the API cannot.
IGNORED_DIRECTORIES = {'__pycache__', 'tests', 'test', 'benchmarks'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', 'utf8')


def is_notice(path):
    lowered = [p.lower() for p in path.parts]
    return any(p.startswith(('license', 'licence', 'copying', 'copyright', 'notice'))
               for p in lowered) or any(p in ('licenses', 'licences') for p in lowered)


def select(runtime, relative, selected_metadata):
    parts = relative.parts
    # Text of installed notices survives even when stored under a test folder.
    if is_notice(relative):
        return True, 'license-or-notice'
    if '__pycache__' in parts or relative.suffix.lower() in {'.pyc', '.pyo'}:
        return False, 'bytecode-cache'
    if relative.suffix.lower() in DEVELOPMENT_SUFFIXES:
        return False, 'development-header-or-linker-debug-file'
    if parts[:3] == ('Lib', 'site-packages', 'numpy'):
        # NumPy 2.2.6 imports _core.tests._natype from its public testing API,
        # which SciPy's runtime array compatibility import requests. Keep its
        # complete support package, including the referenced test resources.
        return True, 'numpy-runtime-including-testing-support'
    if any(p in IGNORED_DIRECTORIES for p in parts):
        return False, 'upstream-test-or-benchmark'
    if relative.suffix.lower() in {'.whl', '.conda'}:
        return False, 'embedded-installer-archive'
    if parts[:2] == ('Lib', 'site-packages'):
        package = parts[2]
        if runtime == 'egg':
            return True, 'original-egg-package'
        if package in M05_PACKAGES or package in selected_metadata or package.startswith('_cffi_backend.'):
            # The classic face graph references only face detector/landmark
            # payloads. Other Solutions remain importable but their unused model
            # payloads are not advertised as supported application features.
            if package == 'mediapipe' and relative.suffix.lower() in {'.tflite', '.binarypb'}:
                tail = parts[3:]
                if len(tail) >= 2 and tail[0] == 'modules' and not tail[1].startswith('face_'):
                    return False, 'non-face-mediapipe-model'
            return True, 'm05-facemesh-media-import-closure'
        return False, 'outside-m05-application-import-closure'
    if parts[:2] == ('Lib', 'ensurepip'):
        return False, 'dependency-installation-tools'
    if parts[:2] in {('Library', 'include'), ('Library', 'lib')}:
        return False, 'development-sdk'
    if parts[0] == 'Scripts':
        return False, 'unused-console-entrypoints'
    # All original MKL CPU dispatch and thread-layer DLLs are retained. The
    # subset must not silently select a different BLAS or CPU-specific variant.
    return True, 'original-interpreter-native-and-data'


PROBE = r'''
import hashlib, importlib.metadata as md, importlib.util, io, json, os
from pathlib import Path
import struct, sys
from fractions import Fraction

kind, snapshot, output = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
output.mkdir(parents=True, exist_ok=False)
prefix=Path(sys.prefix)
os.environ['PATH']=os.pathsep.join([str(prefix), str(prefix/'Library/bin'),
    str(Path(os.environ['SystemRoot'])/'System32'), os.environ['SystemRoot']])
os.environ['MPLCONFIGDIR']=str(output/'matplotlib-cache')
os.environ['MPLBACKEND']='Agg'
handles=[]
if (prefix/'Library/bin').is_dir():handles.append(os.add_dll_directory(str(prefix/'Library/bin')))
sys.path[:0]=[str(snapshot/'backend/src'),str(snapshot/'packages/phonetic_core/src')]
def digest(raw):return hashlib.sha256(raw).hexdigest()
def file_sha(path):return digest(Path(path).read_bytes())
def decode_bundle(payload):
    n=struct.unpack('<Q',payload[:8])[0]; header=json.loads(payload[8:8+n]); position=8+n; files={}
    assert not header.get('error'),header
    for item in header['files']:
        raw=payload[position:position+item['size_bytes']];position+=len(raw)
        assert digest(raw)==item['sha256'];files[item['name']]=raw
    assert position==len(payload)
    return files
report={'success':False,'runtime':kind,'prefix':str(prefix),'tests':[],'signatures':{}}
try:
    import numpy as np
    if kind=='egg':
        from scipy.io import wavfile
        from ptb_worker.egg_runtime import fingerprint
        from ptb_worker.egg_child import prepare as egg_prepare
        from ptb_worker.lpc_child import prepare as lpc_prepare
        from ptb_api.font_models import FigureFontSnapshot
        from ptb_worker.fonts import check_fonts
        report['fingerprint']=fingerprint()
        fonts=check_fonts(FigureFontSnapshot(fallback_policy='portable'))
        assert fonts['available'];report['fonts']=fonts
        fixture=np.load(snapshot/'preview-fixtures/EGG-SYN-PCM16.npz')
        wave=io.BytesIO();wavfile.write(wave,44100,np.column_stack((fixture['load.audio_signal'],fixture['load.egg_signal_raw'])))
        egg_raw=wave.getvalue()
        cases=[('single',dict(mode='single',roi_start=0,roi_end=.5)),
               ('preview',dict(mode='preview',roi_start=0,roi_end=.5,micro_center=.2)),
               ('batch-images',dict(mode='batch',generate_images=True)),
               ('inverse',dict(mode='inverse',roi_start=.1,roi_end=.15))]
        for name, config in cases:
            files=decode_bundle(egg_prepare(egg_raw,config,'synthetic-egg.wav'))
            report['signatures'][name]={n:digest(b) for n,b in files.items()}
            report['tests'].append(name)
        # Invoke the unchanged REAPER native adapter in its own bounded scratch.
        from ptb_worker.native.reaper import Reaper
        from ptb_worker.io.limits import Limits
        from ptb_worker.managed_scratch import ReservedNativeScratch
        scratch=output/'native.wav';scratch.touch(exist_ok=False)
        native=Reaper(snapshot/'resources/research/reaper.exe',ReservedNativeScratch(scratch,4000000),
            Limits(input_bytes=4000000,samples=5760000,output_bytes=2000000,process_bytes=1000000000,timeout_seconds=25))
        files=decode_bundle(egg_prepare(egg_raw,dict(mode='single',roi_start=0,roi_end=.5,
            keep_reaper_f0=True,f0_policy='audio-f0/2'),'synthetic-egg.wav',reaper=native))
        report['signatures']['reaper']={n:digest(b) for n,b in files.items()};report['tests'].append('native-reaper')
        time=np.arange(16000)/16000
        wave=io.BytesIO();wavfile.write(wave,16000,(.25*np.sin(2*np.pi*150*time)).astype(np.float32))
        files=decode_bundle(lpc_prepare(wave.getvalue(),dict(roi_start=.1,roi_end=.15,order=20,
            font={'fallback_policy':'portable'}),'synthetic-tone.wav'))
        report['signatures']['lpc']={n:digest(b) for n,b in files.items()};report['tests'].append('lpc-png-wav-spectrum')
        report['tests'].append('font-preflight')
    else:
        import av, cv2, mediapipe as mp
        assert {n:md.version(n) for n in ('mediapipe','numpy','opencv-contrib-python','av')}=={
            'mediapipe':'0.10.14','numpy':'2.2.6','opencv-contrib-python':'4.13.0.92','av':'16.1.0'}
        report['fingerprint']={n:md.version(n) for n in ('mediapipe','numpy','opencv-contrib-python','av')}
        report['unavailable_candidates']={n:importlib.util.find_spec(n) is None for n in ('jax','jaxlib','scipy')}
        for refine in (False,True):
            with mp.solutions.face_mesh.FaceMesh(static_image_mode=True,max_num_faces=1,refine_landmarks=refine) as mesh:
                result=mesh.process(np.zeros((128,128,3),dtype=np.uint8))
                assert result.multi_face_landmarks is None
            report['tests'].append('facemesh-'+('478' if refine else '468')+'-graph-and-black-frame')
        # Fresh public synthetic media, variable video PTS, with decoded audio.
        media=output/'public-synthetic.mp4'
        pts=[0,3000,7500,10500,15000,18000,22500,25500,30000,33000,37500,40500]
        with av.open(str(media),'w',format='mp4') as container:
            vs=container.add_stream('libx264',rate=30);vs.width=128;vs.height=128;vs.pix_fmt='yuv420p'
            vs.time_base=vs.codec_context.time_base=Fraction(1,90000);vs.options={'bf':'0','crf':'18','preset':'veryfast'}
            aus=container.add_stream('aac',rate=16000);aus.layout='mono'
            for i,t in enumerate(pts):
                image=np.zeros((128,128,3),np.uint8);image[:,:,0]=i*12
                frame=av.VideoFrame.from_ndarray(image,format='rgb24');frame.pts=t;frame.time_base=Fraction(1,90000)
                for packet in vs.encode(frame):container.mux(packet)
            for packet in vs.encode():container.mux(packet)
            samples=(.1*np.sin(2*np.pi*150*np.arange(8000)/16000)).astype(np.float32)[None,:]
            for start in range(0,8000,1024):
                frame=av.AudioFrame.from_ndarray(np.ascontiguousarray(samples[:,start:start+1024]),format='fltp',layout='mono')
                frame.sample_rate=16000;frame.pts=start;frame.time_base=Fraction(1,16000)
                for packet in aus.encode(frame):container.mux(packet)
            for packet in aus.encode():container.mux(packet)
        from ptb_worker.m05_video import analyze_video,JsonLinesSink
        from phonetic_core.lip.sequence import LipConfig
        frame_path=output/'frames.jsonl'
        with frame_path.open('xb') as f:metadata=analyze_video(media,JsonLinesSink(f,10000000),LipConfig())
        assert metadata['timing']['decoded_frames']==12
        assert metadata['complete'] and metadata['validity']['missing']==12
        decoded=[json.loads(s) for s in frame_path.read_text('utf8').splitlines()]
        report['signatures']['video-rows']=file_sha(frame_path)
        report['signatures']['video-pts']=[[r['pts'],r['time_base'],r['time_s']] for r in decoded]
        report['signatures']['face-models']=metadata['model_hashes']
        report['tests'].append('full-offline-m05-variable-pts-pipeline')
        from ptb_worker.m05_results import export_tables
        tables=output/'tables';tables.mkdir();names=export_tables(frame_path,tables,metadata)
        report['signatures']['table-values']={n:file_sha(tables/n) for n in names if n!='preview.json'}
        from ptb_worker.m05_media_export import recording_bundle,inspect_recording
        saved=output/'saved';saved.mkdir(); receipt=recording_bundle(media,saved)
        assert receipt['pts_preserved'] and receipt['video_frames']==12 and receipt['audio']['present']
        with av.open(str(saved/'raw_recording.mp4')) as container:
            times=[float(f.pts*f.time_base) for f in container.decode(video=0)]
        report['signatures']['saved-pts']=times
        report['signatures']['saved-wav']=file_sha(saved/'audio_recording.wav')
        inspected=output/'inspected';inspected.mkdir(); inspection=inspect_recording(media,inspected)
        assert inspection['audio']['present'] and inspection['waveform']['times']
        report['signatures']['inspection']=inspection
        report['tests'].extend(['csv-lip-exports','recording-mp4-aac-wav-save-and-pts-readback','recording-inspection'])
        # Exercise the actual animation encoder with explicitly synthetic points.
        from ptb_worker.m05_animation import export_animation
        animation_metadata=dict(coordinates=dict(resolutions=[[128,128]]),timing=dict(anchor_s=0,last_video_pts_s=.45))
        for format in ('mp4','gif'):
            target=output/('animation.'+format)
            result=export_animation(frame_path,target,animation_metadata,quality='small',format=format,full_mesh=True)
            assert target.stat().st_size>0
            with av.open(str(target)) as container:count=sum(1 for _ in container.decode(video=0))
            assert count>0
            report['signatures']['animation-'+format]=dict(frames=count,size=target.stat().st_size)
            report['tests'].append('animation-'+format+'-encode-and-decode')
        assert not any(n=='jax' or n.startswith('jax.') or n=='scipy' or n.startswith('scipy.') for n in sys.modules)
        report['tests'].append('no-jax-or-scipy-import-for-current-m05-entrypoints')
    # These are the original science modules in this selected prefix. Snapshot
    # application sources are separately hashed and not an ambient PYTHONPATH.
    watched=('numpy','scipy','pandas','parselmouth','matplotlib') if kind=='egg' else ('numpy','cv2','av','mediapipe')
    report['module_files']={n:str(Path(sys.modules[n].__file__).resolve()) for n in watched if n in sys.modules}
    assert all(Path(p).is_relative_to(prefix) for p in report['module_files'].values())
    report['success']=True
finally:
    (output/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n','utf8')
print(json.dumps({'success':report['success'],'runtime':kind,'tests':report['tests']}),flush=True)
'''


def metadata_names(source):
    included = set()
    for directory in (source / 'Lib/site-packages').glob('*.dist-info'):
        metadata = directory / 'METADATA'
        if not metadata.is_file():
            continue
        name = next((line[6:].strip().lower().replace('_', '-') for line in
                     metadata.read_text('utf8', errors='replace').splitlines()
                     if line.startswith('Name: ')), '')
        if name in M05_DISTRIBUTIONS:
            included.add(directory.name)
    return included


def conda_notices(source, destination):
    """Preserve exact local package-cache notices and binding evidence."""
    rows=[]
    for metadata in sorted((source/'conda-meta').glob('*.json')):
        record=json.loads(metadata.read_text('utf8'))
        cache=Path(record.get('extracted_package_dir',''))
        archive=Path(record.get('package_tarball_full_path',''))
        expected=record.get('sha256')
        found=[]
        if (cache/'info/licenses').is_dir():
            found=[p for p in (cache/'info/licenses').rglob('*') if p.is_file()]
        if not found:
            rows.append(dict(package=metadata.stem,files=[],status='no-local-info-licenses',
                             recorded_license=record.get('license')))
            continue
        if not expected or not archive.is_file() or sha(archive)!=expected:
            raise RuntimeError('The exact local Conda notice archive could not be bound: '+metadata.stem)
        files=[]
        for notice in sorted(found):
            relative=notice.relative_to(cache/'info/licenses')
            dest=destination/'notices/conda'/metadata.stem/relative
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(notice,dest)
            checksum=sha(notice)
            assert sha(dest)==checksum
            files.append(dict(path=dest.relative_to(destination).as_posix(),size=dest.stat().st_size,
                              sha256=checksum,source_path=str(notice)))
        rows.append(dict(package=metadata.stem,archive=str(archive),archive_sha256=expected,
                         files=files,status='exact-archive-matched-local-extracted-cache',
                         evidence='Notice bytes are copied from the matching local extracted package cache; this receipt records their hashes separately from the archive hash.'))
    write_json(destination/'notices/conda-cache-evidence.json',dict(schema='ptb-conda-cache-notices/1',packages=rows))
    return rows


def prepare(options):
    bundle = options.source_bundle.resolve()
    stage = options.stage.resolve()
    if not stage.is_relative_to(ROOT / 'output/release-staging'):
        raise ValueError('The new destination must be in output/release-staging')
    if stage.exists():
        raise ValueError('Choose a new destination; existing materials are retained')
    runtime_root = bundle / '_internal/runtimes'
    if not all((runtime_root / name / 'python.exe').is_file() for name in ('egg', 'm05')):
        raise ValueError('An existing verified flattened runtime bundle is required')
    stage.mkdir(parents=True)
    selections = []
    for runtime in ('egg', 'm05'):
        source, target = runtime_root / runtime, stage / runtime
        allowed_metadata = metadata_names(source)
        selected = []
        omitted = []
        for path in sorted(source.rglob('*')):
            if not path.is_file():
                continue
            relative = path.relative_to(source)
            chosen, reason = select(runtime, relative, allowed_metadata)
            item = dict(path=relative.as_posix(), size=path.stat().st_size, reason=reason)
            if chosen:
                destination_relative = relative
                if (runtime == 'm05' and is_notice(relative) and relative.parts[:2] == ('Lib','site-packages')
                    and relative.parts[2] not in M05_PACKAGES and relative.parts[2] not in allowed_metadata
                    and not relative.parts[2].startswith('_cffi_backend.')):
                    # Preserve omitted packages' source/notice evidence without
                    # creating importable empty scipy/jax namespace packages.
                    destination_relative = Path('notices/omitted-distributions') / relative
                    item['source_path'] = item['path']
                    item['path'] = destination_relative.as_posix()
                    item['reason'] = 'omitted-distribution-notice-evidence'
                dest = target / destination_relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, dest)
                item['sha256'] = sha(path)
                if sha(dest) != item['sha256']:
                    raise RuntimeError('Copy mismatch: ' + relative.as_posix())
                selected.append(item)
            else:
                omitted.append(item)
        selections.append(dict(runtime=runtime, source=str(source), selected=selected, omitted=omitted,
            source_bytes=sum(row['size'] for row in selected+omitted),
            selected_bytes=sum(row['size'] for row in selected), omitted_bytes=sum(row['size'] for row in omitted)))
        print(json.dumps(dict(runtime=runtime,files=len(selected),bytes=sum(r['size'] for r in selected),
                              omitted_bytes=sum(r['size'] for r in omitted))),flush=True)
    notice_rows=conda_notices(runtime_root/'egg',stage/'egg')
    notice_files=[item for row in notice_rows for item in row['files']]
    notice_manifest=stage/'egg/notices/conda-cache-evidence.json'
    notice_files.append(dict(path=notice_manifest.relative_to(stage/'egg').as_posix(),size=notice_manifest.stat().st_size,
                             sha256=sha(notice_manifest),reason='generated-conda-cache-notice-evidence'))
    selections[0]['supplemental_notices']=notice_files
    selections[0]['supplemental_notice_bytes']=sum(item['size'] for item in notice_files)
    selections[0]['selected_bytes']+=selections[0]['supplemental_notice_bytes']
    write_json(stage/'selection-manifest.json',dict(schema='ptb-lean-runtime-selection/1',
        source_bundle=str(bundle),created_at=datetime.now(timezone.utc).isoformat(),runtimes=selections,
        policy='No MFA. EGG original MKL dispatcher retained. M05 current classic FaceMesh/media entrypoints only; no GenAI/JAX. Original environments untouched.'))
    # The exact frozen scientific/source snapshot is used for comparisons.
    snapshot=stage/'validation/snapshot'
    for relative in ('backend/src','packages/phonetic_core/src'):
        shutil.copytree(bundle/'_internal'/relative,snapshot/relative,
                        ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for relative in ('resources/m05/legacy-runtime.json','resources/research/reaper.exe',
                     'preview-fixtures/EGG-SYN-PCM16.npz'):
        destination=snapshot/relative;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(bundle/'_internal'/relative,destination)
    write_json(stage/'validation/snapshot-sha256.json',{p.relative_to(snapshot).as_posix():sha(p)
        for p in sorted(snapshot.rglob('*')) if p.is_file()})
    probe=stage/'validation/probe.py';probe.write_text(PROBE,'utf8')
    environment={k:v for k,v in os.environ.items() if k not in
        {'PYTHONHOME','PYTHONPATH','PTB_EGG_PYTHON','PTB_M05_PYTHON','PTB_M11_COMPONENT_ROOT'}}
    environment['PATH']=os.pathsep.join([os.environ['SystemRoot']+'/System32',os.environ['SystemRoot']])
    environment['PYTHONDONTWRITEBYTECODE']='1'
    def run(runtime, original):
        prefix=runtime_root/runtime if original else stage/runtime
        output=stage/'validation'/((runtime+'-reference') if original else (runtime+'-subset'))
        result=subprocess.run([str(prefix/'python.exe'),'-I','-B',str(probe),runtime,str(snapshot),str(output)],
            cwd=stage/'validation',env=environment,capture_output=True,text=True,encoding='utf8',errors='replace',
            timeout=240,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        (output.parent/(output.name+'.stdout.log')).write_text(result.stdout,'utf8')
        (output.parent/(output.name+'.stderr.log')).write_text(result.stderr,'utf8')
        report=json.loads((output/'result.json').read_text('utf8')) if (output/'result.json').is_file() else {}
        return runtime,original,result.returncode,report
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(run,runtime,original) for runtime in ('egg','m05') for original in (True,False)]
        results=[f.result() for f in futures]
    report=dict(schema='ptb-runtime-stage/1',profile='non-mfa-minimum-tested/1',success=False,
                runtimes={},comparisons={},scope='Isolated interpreter and actual synthetic scientific/media entrypoints, not physical devices or final EXE validation.')
    for runtime,original,code,result in results:
        print(json.dumps(dict(runtime=runtime,reference=original,returncode=code,success=result.get('success'))),flush=True)
        report['comparisons'].setdefault(runtime,{})['reference' if original else 'subset']=dict(returncode=code,result=result)
    for runtime in ('egg','m05'):
        pair=report['comparisons'][runtime]
        reference,subset=pair['reference'],pair['subset']
        same=(reference['returncode']==subset['returncode']==0 and reference['result'].get('success') and
              subset['result'].get('success') and reference['result']['signatures']==subset['result']['signatures'] and
              reference['result']['fingerprint']==subset['result']['fingerprint'])
        pair['signatures_and_fingerprint_identical']=bool(same)
        report['runtimes'][runtime]=dict(path=runtime+'/python.exe',sha256=sha(stage/runtime/'python.exe'),
            import_returncode=subset['returncode'],functional_comparison_passed=bool(same),
            bytes=next(row['selected_bytes'] for row in selections if row['runtime']==runtime))
    report['success']=all(r['functional_comparison_passed'] for r in report['runtimes'].values())
    write_json(stage/'stage-report.json',report)
    print(json.dumps(dict(success=report['success'],stage=str(stage),runtime_bytes=sum(r['bytes'] for r in report['runtimes'].values()),
                          manifest_sha256=sha(stage/'selection-manifest.json'))),flush=True)
    return 0 if report['success'] else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-bundle',type=Path,required=True)
    parser.add_argument('--stage',type=Path,required=True)
    return prepare(parser.parse_args())


if __name__=='__main__':
    raise SystemExit(main())
