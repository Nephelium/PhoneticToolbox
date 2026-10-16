"""User-selected local Python, verified in an isolated inference child."""
import json
import os
from pathlib import Path
import subprocess
from .mfa.components import atomic_json,digest,no_links


def registry_path():
    from .vocoder_components import root
    return Path(os.environ.get('PTB_M09_PYTHON_CONFIG',str(root()/'gpu-python.json'))).absolute()


def fingerprint(path):
    path=Path(path).absolute();no_links(path)
    if not path.is_file() or path.name.lower() not in ('python.exe','python3','python'):
        raise ValueError('m09_python_invalid')
    stat=path.stat()
    return dict(path=str(path),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)


def registered():
    path=registry_path();no_links(path)
    try:
        saved=json.loads(path.read_text('utf-8'))
        if saved.get('schema')!='ptb-m09-python/1' or not saved.get('cuda'):return None
        if fingerprint(saved['path'])!={k:saved[k] for k in ('path','bytes','mtime_ns')}:return None
        return saved
    except (OSError,ValueError,KeyError):return None


def status():
    saved=registered()
    return dict(gpu_ready=bool(saved),gpu_name=saved.get('gpu_name') if saved else None,
                python_name=Path(saved['path']).parent.name if saved else None)


def check_and_register(python):
    identity=fingerprint(python)
    before=digest(Path(identity['path']))
    from .vocoder_components import installed,validate_assets
    command=[identity['path'],'-I','-B',str(Path(__file__).with_name('m09_runtime_child.py'))]
    try:component=installed()
    except ValueError:component=None
    if component:
        validate_assets(component)
        command += ['--component-root',str(component)]
    try:
        process=subprocess.run(command,capture_output=True,timeout=120,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        if process.returncode or len(process.stdout)>16384:raise ValueError('m09_python_check_failed')
        result=json.loads(process.stdout)
    except (OSError,subprocess.TimeoutExpired,json.JSONDecodeError):raise ValueError('m09_python_check_failed') from None
    if not result.get('cuda'):
        raise ValueError(result.get('error') if result.get('error') in ('m09_gpu_unavailable','m09_python_dependencies','m09_vocoder_failed') else 'm09_python_check_failed')
    if fingerprint(python)!=identity or digest(Path(identity['path']))!=before:raise ValueError('m09_python_changed')
    target=registry_path();no_links(target);target.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(target,dict(schema='ptb-m09-python/1',**identity,python_sha256=before,**result))
    return result


def interpreter(component,device):
    if device not in ('auto','cpu','cuda'):raise ValueError('invalid_spectrogram_config')
    saved=registered() if device!='cpu' else None
    if saved:
        path=Path(saved['path'])
        if digest(path)!=saved['python_sha256']:raise ValueError('m09_python_changed')
        return path,'cuda'
    if device=='cuda':raise ValueError('m09_gpu_unavailable')
    return component/'env/python.exe','cpu'
