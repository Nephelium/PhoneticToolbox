"""Optional BigVGAN installation through the shared pinned archive manager."""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
from threading import Lock
import time
import zipfile

from .mfa.components import ComponentManager, atomic_json, digest, no_links
from . import vocoder_catalog

REVISION = '760f0c694aba68b8c200dd52ffd3b4ecafa7028c'
_lock = Lock()
_state = dict(stage='idle', completed=0, total=0)


def root():
    return Path(os.environ.get('PTB_VOCODER_ROOT', str(Path(os.environ.get('LOCALAPPDATA', str(Path.home()))) / 'PhoneticToolbox/v3/components/vocoder'))).absolute()


def installed():
    base = root()
    no_links(base)
    current = base / 'current.json'
    no_links(current)
    if not current.is_file():
        raise ValueError('m09_vocoder_missing')
    value = json.loads(current.read_text('utf-8'))
    pinned=next((r for r in (vocoder_catalog.RELEASE,getattr(vocoder_catalog,'PREVIOUS_RELEASE',{})) if r and r.get('id')==value.get('id')),None)
    if not pinned:
        raise ValueError('m09_vocoder_missing')
    path = Path(value['path']).absolute()
    if not path.is_relative_to(base / 'versions'):
        raise ValueError('m09_vocoder_invalid')
    no_links(path)
    manifest_path=path / 'component-manifest.json'
    no_links(manifest_path)
    if not manifest_path.is_file() or digest(manifest_path)!=pinned.get('manifest_sha256'):
        raise ValueError('m09_vocoder_invalid')
    for name in ['env/python.exe', 'runner.py', 'model/config.json', 'model/bigvgan_generator.pt']:
        p = path / name
        no_links(p)
        if not p.is_file():
            raise ValueError('m09_vocoder_invalid')
    return path


def status():
    pipeline_ready=False
    try:
        path=installed()
        ready = True
        pipeline_ready=bool(vocoder_catalog.RELEASE.get('pipeline')) and json.loads((path/'component-manifest.json').read_text('utf-8'))['id']==vocoder_catalog.RELEASE['id']
    except (ValueError, OSError, KeyError):
        ready = False
    pinned = vocoder_catalog.RELEASE
    downloadable=pinned if pinned.get('published',True) else getattr(vocoder_catalog,'PREVIOUS_RELEASE',{})
    from .m09_runtime import status as runtime_status
    return dict(**runtime_status(),ready=ready, pipeline_ready=pipeline_ready, available=bool(downloadable),
                pipeline_available=bool(pinned.get('pipeline')) and pinned.get('published',False),
                pipeline_download_bytes=pinned.get('download_bytes',0),model='BigVGAN base',
                download_bytes=downloadable.get('download_bytes', 0), installed_bytes=(pinned if pipeline_ready else downloadable).get('installed_bytes', 0),
                progress=dict(_state))


def install(local_archive=None):
    global _state
    if not _lock.acquire(False):
        raise ValueError('m09_vocoder_busy')
    try:
        pinned = vocoder_catalog.RELEASE
        if local_archive is None and not pinned.get('published',True):pinned=getattr(vocoder_catalog,'PREVIOUS_RELEASE',{})
        current=status()
        if current['ready'] and (not pinned.get('pipeline') or current.get('pipeline_ready')):
            return status()
        if not pinned or (local_archive is None and not pinned.get('published',True)):
            raise ValueError('m09_vocoder_not_published')
        manager = ComponentManager(root(), schema='ptb-vocoder-component/1', version_field='model_revision', version=REVISION, prefix='bigvgan-r1')
        def progress(stage, done=0, total=0):
            global _state
            _state = dict(stage=stage, completed=done, total=total)
        archive = Path(local_archive) if local_archive is not None else manager.download(pinned, progress=lambda d,t: progress('downloading', d,t))
        no_links(archive)
        if archive.stat().st_size!=pinned['download_bytes'] or digest(archive)!=pinned['sha256']:
            raise ValueError('m09_vocoder_invalid')
        progress('verifying')
        with zipfile.ZipFile(archive) as z:
            info = z.getinfo('component-manifest.json')
            if info.file_size > 32_000_000:
                raise ValueError('m09_vocoder_invalid')
            raw = z.read(info)
        if hashlib.sha256(raw).hexdigest() != pinned['manifest_sha256']:
            raise ValueError('m09_vocoder_invalid')
        trusted = json.loads(raw)
        if trusted.get('id') != pinned['id']:
            raise ValueError('m09_vocoder_invalid')
        trusted.update(sha256=pinned['sha256'], download_bytes=pinned['download_bytes'])
        trusted['files'].append(dict(path='component-manifest.json', bytes=len(raw), sha256=pinned['manifest_sha256']))
        trusted['installed_bytes'] += len(raw)
        def check(target):
            progress('checking')
            result = subprocess.run([str(target / 'env/python.exe'), '-I', '-B', str(target / ('pipeline.py' if pinned.get('pipeline') else 'runner.py')), '--check'],
                capture_output=True, timeout=120, creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            if result.returncode:
                raise ValueError('m09_vocoder_self_test_failed')
            receipt = json.loads(result.stdout)
            if receipt.get('success') is not True:
                raise ValueError('m09_vocoder_self_test_failed')
            return receipt
        manager.import_archive(archive, trusted, check, progress=lambda d,t: progress('extracting', d,t))
        progress('complete')
        return status()
    except Exception:
        _state = dict(stage='failed', completed=0, total=0)
        raise
    finally:
        _lock.release()


def validate_assets(path):
    manifest=json.loads((path/'component-manifest.json').read_text('utf-8'))
    for row in manifest['files']:
        if row['path'].startswith(('model/','models/')) or row['path']=='env/python.exe':
            p=path/row['path'];no_links(p)
            if p.stat().st_size!=row['bytes'] or digest(p)!=row['sha256']:raise ValueError('m09_vocoder_invalid')
    return manifest


def apply(result,device='auto'):
    """No disk scratch: bounded IPC; nested process belongs to parent's job."""
    import numpy as np
    from phonetic_core.spec2wav.reconstruction import stft
    path = installed()
    # The pinned manifest protects the installed runner, model and interpreter.
    validate_assets(path)
    from .m09_runtime import interpreter
    python,device=interpreter(path,device)
    stream = io.BytesIO()
    np.savez(stream, audio=result['audio'], sr=result['sr'],device=device)
    completed = subprocess.run([str(python), '-I', '-B', str(Path(__file__).with_name('vocoder_child.py')),'--model-root',str(path/'model')],
        input=stream.getvalue(), capture_output=True, timeout=180,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    if completed.returncode or len(completed.stdout) > 32_000_000:
        raise ValueError('m09_vocoder_failed')
    with np.load(io.BytesIO(completed.stdout), allow_pickle=False) as data:
        audio = data['audio']; metadata = json.loads(str(data['metadata']))
    if audio.shape != result['audio'].shape or not np.isfinite(audio).all():
        raise ValueError('m09_vocoder_failed')
    result['audio'] = audio
    meta = result['metadata']
    refresh_image(result)
    meta.update(vocoder=metadata, phase_method='griffin-lim+bigvgan',
                pcm16_clipped_samples=int(np.count_nonzero((audio>=1)|(audio < -1))))
    return result


def refresh_image(result):
    import numpy as np
    from phonetic_core.spec2wav.reconstruction import stft
    audio=result['audio'];meta=result['metadata']
    native = audio if result['sr']==meta['native_sr'] else np.interp(
        np.arange(round(len(audio) / result['sr'] * meta['native_sr'])) * result['sr'] / meta['native_sr'], np.arange(len(audio)), audio)
    comparison = np.abs(stft(native, meta['n_fft'], meta['hop_length'], np.hanning(meta['win_length'])))
    db = 10 * np.log10(np.maximum(1e-10, comparison))
    db -= db.max(); db = np.maximum(db, -80)
    result['image'] = np.flipud(np.uint8(np.clip(-db / max(1e-10, -float(db.min())) * 255, 0, 255)))


def pipeline(gray,config):
    import numpy as np
    from .m09_parameters import predict
    prediction=predict(gray,config) if config.get('auto_parameters',True) else None
    applied=dict(config)
    if prediction:
        for key in ('win_length_ms','min_db','max_db'):applied[key]=prediction[key]
    # Also compatible with installed r1 components: they must not run their
    # former parameter model. Both preview and rendering use the built-in model.
    applied['auto_parameters']=False
    from phonetic_core.spec2wav.adaptive import reconstruct_calibrated
    keys=('time_start','time_end','freq_start','freq_end','min_db','max_db','win_length_ms','n_iter','target_sr','seed','frequency_axis','phase_initialization')
    result=reconstruct_calibrated(gray,**{k:applied[k] for k in keys if k in applied})
    path=installed()
    if not status()['pipeline_ready']:raise ValueError('m09_vocoder_missing')
    manifest=validate_assets(path)
    from .m09_runtime import interpreter
    python,device=interpreter(path,config.get('inference_device','auto'))
    applied['inference_device']=device
    stream=io.BytesIO();np.savez(stream,audio=result['audio'],sr=result['sr'],config=json.dumps(applied),operation='enhance')
    completed=subprocess.run([str(python),'-I','-B',str(Path(__file__).with_name('m09_pipeline_child.py')),'--component-root',str(path)],
        input=stream.getvalue(),capture_output=True,timeout=210,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    if completed.returncode or len(completed.stdout)>32_000_000:raise ValueError('m09_vocoder_failed')
    with np.load(io.BytesIO(completed.stdout),allow_pickle=False) as data:
        audio=data['audio'];vocoder=json.loads(str(data['metadata']))
    if audio.shape!=result['audio'].shape or not np.isfinite(audio).all():
        raise ValueError('m09_vocoder_failed')
    result['audio']=audio;meta=result['metadata']
    meta.update(component_id=manifest['id'],vocoder=vocoder,phase_method='griffin-lim+repair80+bigvgan')
    meta.update(prediction=prediction,applied_config=applied,
                parameter_backend='builtin-numpy-float32' if prediction else 'manual')
    return result
