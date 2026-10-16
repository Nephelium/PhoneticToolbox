import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from ptb_worker import m09_runtime as runtime
from ptb_worker import vocoder_components


def test_gpu_requires_checked_environment_and_cpu_remains_available(tmp_path,monkeypatch):
    monkeypatch.setenv('PTB_M09_PYTHON_CONFIG',str(tmp_path/'gpu.json'))
    assert not runtime.status()['gpu_ready']
    assert runtime.interpreter(tmp_path,'auto')==(tmp_path/'env/python.exe','cpu')
    with pytest.raises(ValueError,match='m09_gpu_unavailable'):runtime.interpreter(tmp_path,'cuda')


def test_probe_registers_only_success_and_changed_python_requires_recheck(tmp_path,monkeypatch):
    monkeypatch.setenv('PTB_M09_PYTHON_CONFIG',str(tmp_path/'gpu.json'))
    python=tmp_path/'python.exe';python.write_bytes(b'owned fake executable')
    def absent():raise ValueError('m09_vocoder_missing')
    monkeypatch.setattr(vocoder_components,'installed',absent)
    result=dict(cuda=True,gpu_name='fixture GPU',model_checked=False)
    monkeypatch.setattr(runtime.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0,stdout=json.dumps(result).encode()))
    assert runtime.check_and_register(python)['cuda']
    assert runtime.interpreter(tmp_path,'cuda')==(python,'cuda')
    assert runtime.interpreter(tmp_path,'cpu')==(tmp_path/'env/python.exe','cpu')
    python.write_bytes(b'changed')
    assert not runtime.status()['gpu_ready']
    with pytest.raises(ValueError):runtime.interpreter(tmp_path,'cuda')


def test_failed_probe_does_not_register_or_install(tmp_path,monkeypatch):
    monkeypatch.setenv('PTB_M09_PYTHON_CONFIG',str(tmp_path/'gpu.json'))
    python=tmp_path/'python.exe';python.write_bytes(b'owned fake executable')
    def absent():raise ValueError('m09_vocoder_missing')
    monkeypatch.setattr(vocoder_components,'installed',absent)
    monkeypatch.setattr(runtime.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0,stdout=b'{"cuda":false,"error":"m09_python_dependencies"}'))
    with pytest.raises(ValueError,match='m09_python_dependencies'):runtime.check_and_register(python)
    assert not (tmp_path/'gpu.json').exists()
