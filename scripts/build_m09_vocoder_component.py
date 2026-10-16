"""Build the separately downloaded Windows CPU BigVGAN component.

Copies only the installed dependency closure and a relocatable project-owned
Python base; no global installation, training checkpoint, private data, or app
EXE. Generates the immutable catalog and payload manifest from actual bytes.
"""
import argparse
import ast
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
import os
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def copy(source, target):
    source = source.absolute()
    for p in (source, *source.parents):
        if p.is_symlink() or getattr(p.lstat(), 'st_file_attributes', 0) & 0x400:
            raise ValueError('Linked build source rejected')
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--repair-checkpoint',type=Path)
    args = ap.parse_args()
    pipeline=bool(args.repair_checkpoint)
    args.output.mkdir(parents=True, exist_ok=False)
    lifecycle=args.output / 'artifact-lifecycle.json'
    lifecycle.write_text(json.dumps(dict(schema='ptb-build-artifacts/1', owner_pid=os.getpid(), state='active',
        started_at=datetime.now(timezone.utc).isoformat(), purpose='optional BigVGAN CPU component'),indent=2)+'\n','utf-8')
    stage = args.output / 'component'
    stage.mkdir()
    for src in Path(sys.base_prefix).iterdir():
        if src.is_file() and src.suffix in ('.exe','.dll','.txt'):
            copy(src, stage / 'env' / src.name)
    for folder in ['Lib', 'DLLs']:
        for src in (Path(sys.base_prefix) / folder).rglob('*'):
            rel = src.relative_to(sys.base_prefix)
            if src.is_file() and not any(p in ('site-packages','__pycache__','test','tests','idlelib','ensurepip') for p in rel.parts):
                copy(src, stage / 'env' / rel)
    # Dependency closure respects markers; it does not carry the evaluation-only
    # Vocos, EnCodec, TorchAudio, pip or setuptools environments into this model.
    installed = {canonicalize_name(d.metadata['Name']):d for d in metadata.distributions()}
    todo = ['torch', 'librosa', 'matplotlib', 'huggingface-hub']
    names = set()
    while todo:
        name = canonicalize_name(todo.pop())
        if name in names:
            continue
        names.add(name)
        for raw in installed[name].requires or []:
            requirement = Requirement(raw)
            if not requirement.marker or requirement.marker.evaluate({'extra':''}):
                todo.append(requirement.name)
    site = (Path(sys.prefix) / 'Lib/site-packages').resolve()
    dependencies = []
    for name in sorted(names):
        dist = installed[name]
        dependencies.append(dict(name=dist.metadata['Name'], version=dist.version,
                                 license=dist.metadata.get('License'), license_expression=dist.metadata.get('License-Expression')))
        for relative in dist.files or []:
            source = Path(dist.locate_file(relative)).resolve()
            if not source.is_relative_to(site) or not source.is_file():
                continue
            parts = source.relative_to(site).parts
            if '__pycache__' in parts or source.suffix in ('.pyc','.pdb','.lib','.h','.hpp','.a'):
                continue
            copy(source, stage / 'env/Lib/site-packages' / source.relative_to(site))
    # Config, decoder code, and generator weights come from one pinned upstream.
    for source in args.model_root.rglob('*'):
        if source.is_file() and source.name != 'download-manifest.json':
            copy(source, stage / 'model' / source.relative_to(args.model_root))
    copy(ROOT / 'backend/src/ptb_worker/vocoder_child.py', stage / 'runner.py')
    model_sizes={}
    if pipeline:
        import torch
        import numpy as np
        for name,checkpoint in [('repair80',args.repair_checkpoint)]:
            # A single read freezes an atomically published training checkpoint.
            import io
            raw=checkpoint.read_bytes(); saved=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
            weights=saved['model']
            if not all(torch.isfinite(v).all() for v in weights.values()):raise ValueError('Non-finite weights')
            target=stage/'models'/f'{name}.pt';target.parent.mkdir(exist_ok=True)
            torch.save(dict(model=weights),target)
            model_sizes[name]=dict(checkpoint_sha256=hashlib.sha256(raw).hexdigest(),
                checkpoint_bytes=len(raw),inference_bytes=target.stat().st_size,step=saved.get('step'),
                epoch=saved.get('epoch'),config=saved.get('config',{}))
        for name in ['m09_neural_models','vocoder_child']:
            copy(ROOT/f'backend/src/ptb_worker/{name}.py',stage/f'{name}.py')
        copy(ROOT/'backend/src/ptb_worker/m09_pipeline_child.py',stage/'pipeline.py')
        for name in ['__init__.py','reconstruction.py','adaptive.py']:
            copy(ROOT/f'packages/phonetic_core/src/phonetic_core/spec2wav/{name}',stage/'phonetic_core/spec2wav'/name)
        (stage/'phonetic_core/__init__.py').write_text('',encoding='utf-8')
        # Publication metadata excludes local paths and training configs.
        (args.output/'model-freeze.json').write_text(json.dumps(model_sizes,ensure_ascii=False,indent=2)+'\n','utf-8')
        manifest_models={name:{k:v for k,v in record.items() if k!='config'} for name,record in model_sizes.items()}
        (stage/'models/provenance.json').write_text(json.dumps(manifest_models,indent=2)+'\n','utf-8')
    json_text = lambda v: json.dumps(v, ensure_ascii=False, indent=2) + '\n'
    (stage / 'runtime-dependencies.json').write_text(json_text(dependencies), 'utf-8')
    (stage / 'NOTICE.txt').write_text('BigVGAN base optional component. Model/source revision 760f0c694aba68b8c200dd52ffd3b4ecafa7028c.\n'
        'NVIDIA BigVGAN and HiFi-GAN adaptations: MIT, see model/LICENSE and model source attribution.\n'
        'Alias-free Torch activation: Apache-2.0, see model/alias-free-torch-LICENSE.\n'
        'Python and dependency license/NOTICE files remain at their original package paths.\n'
        'PhoneticToolbox runner is a fixed CPU inference adapter; no training/discriminator or private input bundled.\n', 'utf-8')
    # Run the actual new Python tree, before computing its final identity.
    tested = subprocess.run([str(stage / 'env/python.exe'), '-I', '-B', str(stage / ('pipeline.py' if pipeline else 'runner.py')), '--check'],
        capture_output=True, timeout=180)
    (args.output / 'self-test.stderr.txt').write_bytes(tested.stderr)
    if tested.returncode:
        raise RuntimeError('Component Python self-test failed; inspect self-test.stderr.txt')
    receipt = json.loads(tested.stdout)
    if receipt.get('success') is not True:
        raise ValueError('Invalid component self-test receipt')
    (args.output / 'self-test.json').write_text(json_text(receipt), 'utf-8')
    files = [dict(path=p.relative_to(stage).as_posix(), bytes=p.stat().st_size, sha256=digest(p))
             for p in sorted(stage.rglob('*')) if p.is_file()]
    component_id='spec2wav-neural-windows-x64-r2' if pipeline else 'bigvgan-base-windows-x64-r1'
    payload = dict(schema='ptb-vocoder-component/1', id=component_id,
        platform='windows', arch='x86_64', model_revision='760f0c694aba68b8c200dd52ffd3b4ecafa7028c',
        source='NVIDIA BigVGAN base 22 kHz; project CPU component r1', dependencies=dependencies,
        installed_bytes=sum(f['bytes'] for f in files), files=files)
    manifest = stage / 'component-manifest.json'
    manifest.write_text(json_text(payload), 'utf-8')
    filename = 'PhoneticToolbox-Spec2Wav-neural-r2-windows-x64.zip' if pipeline else 'PhoneticToolbox-BigVGAN-base-r1-windows-x64.zip'
    archive = args.output / filename
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(stage.rglob('*')):
            if p.is_file():
                z.write(p, p.relative_to(stage).as_posix())
    release = dict(id=payload['id'], sha256=digest(archive), manifest_sha256=digest(manifest),
        download_bytes=archive.stat().st_size, installed_bytes=payload['installed_bytes'] + manifest.stat().st_size,
        filename=filename, url='https://www.phonetictoolbox.com/releases/components/vocoder/bigvgan-base-r1/' + filename,
        trusted_host='www.phonetictoolbox.com')
    if pipeline:
        release.update(pipeline=True,published=False,
            url='https://www.phonetictoolbox.com/releases/components/spec2wav/neural-r2/'+filename)
    (args.output / 'release.json').write_text(json_text(release), 'utf-8')
    catalog = ROOT / 'backend/src/ptb_worker/vocoder_catalog.py'
    prior={}
    exec(catalog.read_text('utf-8'),prior)
    previous=prior.get('PREVIOUS_RELEASE') or prior.get('RELEASE')
    catalog.write_text('"""Fixed identity generated by scripts/build_m09_vocoder_component.py."""\nRELEASE = ' + repr(release) + '\nPREVIOUS_RELEASE = '+repr(previous)+'\n', 'utf-8')
    state=json.loads(lifecycle.read_text('utf-8'));state.update(state='completed',finished_at=datetime.now(timezone.utc).isoformat(),archive_sha256=release['sha256'])
    lifecycle.write_text(json_text(state),'utf-8')
    print(json_text(release), flush=True)


if __name__ == '__main__':
    main()
