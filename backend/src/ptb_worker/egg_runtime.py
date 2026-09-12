"""Host-selected compatibility runtime. Never replaces DLLs in the API/Qt host."""
import os
from pathlib import Path


def command(request, pipe):
    from .acoustic_errors import AcousticFailure
    configured = os.environ.get('PTB_EGG_PYTHON','')
    path = Path(configured)
    if not configured or not path.is_absolute() or not path.is_file() or path.name.lower() != 'python.exe':
        raise AcousticFailure('egg_runtime_unavailable')
    if not (path.parent/'conda-meta/scipy-1.16.3-py311hf127856_1.json').is_file():
        raise AcousticFailure('egg_runtime_mismatch')
    # -I excludes ambient PYTHONPATH and user site. The fixed bootstrap inserts
    # only this installed backend package and initializes this child's DLL path.
    return [str(path),'-I','-B',str(Path(__file__).with_name('egg_bootstrap.py')),str(request),pipe]


def fingerprint():
    import importlib.metadata as metadata
    import json
    import sys
    import scipy
    import numpy as np
    from .acoustic_errors import AcousticFailure
    lock = Path(sys.prefix)/'conda-meta/scipy-1.16.3-py311hf127856_1.json'
    if not lock.is_file(): raise AcousticFailure('egg_runtime_mismatch')
    record = json.loads(lock.read_text('utf-8'))
    config = scipy.__config__.CONFIG
    blas = config['Build Dependencies']['blas']
    versions = {k:metadata.version(k) for k in ('phonetic-core','numpy','scipy','praat-parselmouth','matplotlib','pandas')}
    expected = dict(zip(versions,['3.0.0a1','2.2.6','1.16.3','0.4.7','3.10.8','2.3.3']))
    required = ['libblas-3.11.0-4_hf2e6a31_mkl','liblapack-3.11.0-4_hf9ab0e9_mkl','mkl-2025.3.0-hac47afa_454']
    if versions != expected or record.get('build') != 'py311hf127856_1' or blas['name'] != 'blas' or not all((lock.parent/(name+'.json')).is_file() for name in required):
        raise AcousticFailure('egg_runtime_mismatch')
    return dict(versions=versions,scipy_build=record['build'],blas='MKL 2025.3.0',conda_builds=required,
                numpy_blas=np.__config__.CONFIG['Build Dependencies']['blas']['name'])
