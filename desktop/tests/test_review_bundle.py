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
