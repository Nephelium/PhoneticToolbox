import hashlib
import json
import pytest
from ptb_desktop.bundle_manifest import runtime_bindings


def fixture(root):
    runtime=root/'runtimes/egg/python.exe';runtime.parent.mkdir(parents=True);runtime.write_bytes(b'fixture')
    return dict(schema='desktop-bundle/1',portable=True,platform='win32',architecture='x86_64',
        runtimes={'PTB_EGG_PYTHON':{'path':'runtimes/egg/python.exe','sha256':hashlib.sha256(b'fixture').hexdigest()}})


def test_runtime_binding_survives_moving_the_bundle(tmp_path):
    root=tmp_path/'staged';root.mkdir();value=fixture(root)
    (root/'desktop-bundle.json').write_text(json.dumps(value))
    moved=tmp_path/'moved';root.rename(moved)
    assert runtime_bindings(moved,host_platform='win32',host_arch='AMD64')['PTB_EGG_PYTHON']==str(moved/'runtimes/egg/python.exe')


def test_clean_windows_environment_uses_interpreter_abi(tmp_path,monkeypatch):
    from ptb_desktop import bundle_manifest as module
    monkeypatch.setattr(module.platform,'machine',lambda:'')
    monkeypatch.setattr(module.sys,'platform','win32')
    monkeypatch.setattr(module.sysconfig,'get_platform',lambda:'win-amd64')
    (tmp_path/'desktop-bundle.json').write_text(json.dumps(fixture(tmp_path)))
    assert runtime_bindings(tmp_path)['PTB_EGG_PYTHON'].endswith('python.exe')
    monkeypatch.setattr(module.sysconfig,'get_platform',lambda:'win-arm64')
    with pytest.raises(ValueError,match='architecture'):
        runtime_bindings(tmp_path)

def test_bundled_mfa_registry_pinned_and_rebased(tmp_path):
    value=fixture(tmp_path)
    path=tmp_path/'runtimes/mfa/registry-bundled.json';path.parent.mkdir();path.write_bytes(b'{"schema":"m11-registry/1"}')
    value['mfa']={'registry':'runtimes/mfa/registry-bundled.json','sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    (tmp_path/'desktop-bundle.json').write_text(json.dumps(value))
    assert runtime_bindings(tmp_path,host_platform='win32',host_arch='AMD64')['PTB_M11_BUNDLED_COMPONENTS']==str(path.parent)
    path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='hash mismatch'):runtime_bindings(tmp_path,host_platform='win32',host_arch='AMD64')


@pytest.mark.parametrize('change',['absolute','escape','hash','platform','portable'])
def test_invalid_bundle_binding_is_rejected(tmp_path,change):
    value=fixture(tmp_path);item=value['runtimes']['PTB_EGG_PYTHON']
    if change=='absolute':item['path']='D:/developer/python.exe'
    elif change=='escape':item['path']='../python.exe'
    elif change=='hash':item['sha256']='0'*64
    elif change=='platform':value['platform']='darwin'
    else:value['portable']=False
    (tmp_path/'desktop-bundle.json').write_text(json.dumps(value))
    with pytest.raises(ValueError):runtime_bindings(tmp_path,host_platform='win32',host_arch='AMD64')
