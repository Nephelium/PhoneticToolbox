import hashlib
from io import BytesIO
import json
from pathlib import Path
import stat
import subprocess
import sys
import zipfile
import pytest

from ptb_desktop import update_apply as apply
from ptb_desktop.updates import UpdateError, _atomic_json

VERSION = '3.0.0-preview.2'


def onefile_manifest(body):
    return dict(schema='ptb-desktop-release/1', version=VERSION, entry='PhoneticToolbox.exe', layout='onefile/1',
                executable=dict(size=len(body), sha256=hashlib.sha256(body).hexdigest()))


def test_onefile_update_without_external_runtime_keeps_original(tmp_path):
    body = b'MZonefile-fixture'
    package = tmp_path / 'onefile.zip'
    with zipfile.ZipFile(package, 'w') as archive:
        archive.writestr('PhoneticToolbox/application.json', json.dumps(onefile_manifest(body)))
        archive.writestr('PhoneticToolbox/PhoneticToolbox.exe', body)
    exe = apply.extract_portable(package, tmp_path / 'new', VERSION)
    assert exe.read_bytes() == body
    assert not (exe.parent / '_internal').exists()
    exe.write_bytes(b'MZcorrupt')
    with pytest.raises(UpdateError):
        apply.release_layout(exe.parent, VERSION)


def test_onefile_helper_is_self_contained(tmp_path):
    app = tmp_path / 'program'; app.mkdir()
    origin = app / 'PhoneticToolbox.exe'; origin.write_bytes(b'MZonefile-fixture')
    (app / 'application.json').write_text(json.dumps(onefile_manifest(origin.read_bytes())), 'utf8')
    root = tmp_path / 'updates'; root.mkdir()
    helper = apply.stage_helper(root, origin)
    assert helper.read_bytes() == origin.read_bytes()
    assert not (helper.parent / '_internal').exists()


def test_helper_preserves_frozen_startup_data_without_copying_science(tmp_path):
    app = tmp_path / 'program'; app.mkdir()
    origin = app / 'PhoneticToolbox.exe'; origin.write_bytes(b'MZowned-fixture')
    internal = app / '_internal'
    text = internal / 'setuptools/_vendor/jaraco/text/Lorem ipsum.txt'
    text.parent.mkdir(parents=True); text.write_text('startup resource', encoding='utf8')
    qt = internal / 'PyQt6'; qt.mkdir(); (qt / 'QtCore.pyd').write_bytes(b'owned qt extension')
    science = internal / 'runtimes/mfa/runtime'; science.mkdir(parents=True)
    (science / 'python.exe').write_bytes(b'MZnot-a-helper-dependency')
    root = tmp_path / 'updates'; root.mkdir()
    helper = apply.stage_helper(root, origin)
    payload = helper.parent / '_internal'
    assert (payload / 'setuptools/_vendor/jaraco/text/Lorem ipsum.txt').read_text('utf8') == 'startup resource'
    assert (payload / 'PyQt6/QtCore.pyd').read_bytes() == b'owned qt extension'
    assert (payload / 'PyQt6/Qt6/plugins').is_dir()
    assert not (payload / 'runtimes').exists()


def bundle():
    return {'schema':'desktop-bundle/1','portable':True,'platform':'win32','architecture':'x86_64','runtimes':{}}


def zip_body(extra=(), version=VERSION):
    stream = BytesIO()
    with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('PhoneticToolbox/application.json',json.dumps({'schema':'ptb-desktop-release/1','version':version,'entry':'PhoneticToolbox.exe'}))
        archive.writestr('PhoneticToolbox/PhoneticToolbox.exe',b'MZrelease-fixture')
        archive.writestr('PhoneticToolbox/_internal/desktop-bundle.json',json.dumps(bundle()))
        for name, body in extra:
            if isinstance(name,str):
                raw=zipfile.ZipInfo(name);raw.filename=name;raw.orig_filename=name;raw.compress_type=zipfile.ZIP_DEFLATED;name=raw
            archive.writestr(name,body)
    return stream.getvalue()


@pytest.mark.parametrize('name',['../outside','/absolute','C:/drive','PhoneticToolbox/../escape','PhoneticToolbox/a:b',
 'PhoneticToolbox/a\\b','PhoneticToolbox/CON.txt','PhoneticToolbox/Lpt1','PhoneticToolbox/trailing.',
 'PhoneticToolbox/trailing ','PhoneticToolbox//empty','PhoneticToolbox/a/./b','OtherRoot/file',
 'PhoneticToolbox/PHONETICTOOLBOX.EXE','PhoneticToolbox/control\x01.txt','PhoneticToolbox/PhoneticToolbox.exe/sub'])
def test_windows_path_and_case_aliases_rejected(name):
    with zipfile.ZipFile(BytesIO(zip_body([(name,b'bad')]))) as archive, pytest.raises(UpdateError):
        apply.archive_members(archive)


def test_link_member_rejected():
    info=zipfile.ZipInfo('PhoneticToolbox/link')
    info.create_system=3
    info.external_attr=(stat.S_IFLNK|0o777)<<16
    with zipfile.ZipFile(BytesIO(zip_body([(info,b'elsewhere')]))) as archive, pytest.raises(UpdateError):
        apply.archive_members(archive)


def test_bomb_and_counts_rejected(monkeypatch):
    with zipfile.ZipFile(BytesIO(zip_body([('PhoneticToolbox/bomb',b'\0'*1000000)]))) as archive, pytest.raises(UpdateError):
        apply.archive_members(archive)
    monkeypatch.setattr(apply,'MAX_FILES',2)
    with zipfile.ZipFile(BytesIO(zip_body())) as archive, pytest.raises(UpdateError):
        apply.archive_members(archive)


def test_extract_new_sibling_keeps_old_and_refuses_existing(tmp_path):
    old=tmp_path/'old';old.mkdir();(old/'keep.txt').write_text('old')
    package=tmp_path/'update.zip';package.write_bytes(zip_body())
    new=tmp_path/'new'
    assert apply.extract_portable(package,new,VERSION)==new/'PhoneticToolbox/PhoneticToolbox.exe'
    assert (old/'keep.txt').read_text()=='old'
    with pytest.raises(UpdateError):apply.extract_portable(package,new,VERSION)
    assert (new/'PhoneticToolbox/PhoneticToolbox.exe').read_bytes()==b'MZrelease-fixture'


def test_layout_version_and_missing_runtime_fail(tmp_path):
    package=tmp_path/'update.zip';package.write_bytes(zip_body(version='3.0.0-preview.99'))
    with pytest.raises(UpdateError):apply.extract_portable(package,tmp_path/'new',VERSION)
    assert package.exists()


def plan_fixture(tmp_path, *, kind='portable'):
    root=tmp_path/'updates';folder=root/'apply'/('a'*32);folder.mkdir(parents=True)
    package=root/'downloads'/('b'*32)/('package.zip' if kind=='portable' else 'setup.exe');package.parent.mkdir(parents=True)
    package.write_bytes(zip_body() if kind=='portable' else b'MZinstaller-fixture')
    helper=root/'helpers'/('c'*32)/'PhoneticToolbox.exe';helper.parent.mkdir(parents=True);helper.write_bytes(b'MZhelper-fixture')
    _atomic_json(helper.parent/'helper-files.json',{'schema':'ptb-update-helper/1','files':[]})
    origin=tmp_path/'old'/'PhoneticToolbox.exe';origin.parent.mkdir();origin.write_bytes(b'MZold-fixture')
    if kind=='installer':_atomic_json(origin.parent/'.ptb-installed.json',{'schema':'ptb-install/1','kind':'installer'})
    request=folder/'request.json';token='d'*32
    plan={'schema':'ptb-apply-update/1','token':token,'kind':kind,'size':package.stat().st_size,
          'sha256':hashlib.sha256(package.read_bytes()).hexdigest(),'package':str(package),'version':VERSION,
          'currentVersion':'3.0.0-preview.1','helperExe':str(helper),'helperSize':helper.stat().st_size,
          'helperSha256':hashlib.sha256(helper.read_bytes()).hexdigest(),
          'helperFilesSha256':hashlib.sha256((helper.parent/'helper-files.json').read_bytes()).hexdigest(),
          'originExe':str(origin),'userData':str(apply.user_data_root()),'parentPid':12345}
    _atomic_json(request,plan)
    _atomic_json(folder/'closed.json',{'schema':'ptb-update-closed/1','token':token})
    return root,request,token,helper,package,origin


def test_handoff_waits_then_extracts_and_launches(tmp_path):
    root,request,token,helper,package,origin=plan_fixture(tmp_path)
    events=[]
    def wait(handle,pid):
        assert not list(tmp_path.glob('PhoneticToolbox-*'))
        events.append(('wait',handle,pid))
    def launch(args,**options):
        events.append(('launch',args));return type('Child',(),{'pid':999})()
    assert apply.run_handoff(request,token,42,root=root,executable=helper,wait=wait,launch=launch)==0
    assert events[0]==('wait',42,12345) and events[1][0]=='launch'
    assert origin.read_bytes()==b'MZold-fixture' and package.exists()
    assert json.loads((request.parent/'status.json').read_text('utf-8'))['state']=='started'


@pytest.mark.parametrize('change',['token','closed','package','helper','manifest','parent-wait','launch'])
def test_handoff_failure_keeps_original_and_never_claims_complete(tmp_path,change):
    root,request,token,helper,package,origin=plan_fixture(tmp_path)
    if change=='token':token='wrong'
    if change=='closed':(request.parent/'closed.json').write_text('{}')
    if change=='package':package.write_bytes(b'changed')
    if change=='helper':helper.write_bytes(b'changed')
    if change=='manifest':(helper.parent/'helper-files.json').write_text('{}')
    def wait(*args):
        if change=='parent-wait':raise UpdateError('APPLY_PROCESS','parent alive')
    calls=[]
    def launch(*args,**kwargs):
        calls.append(args)
        if change=='launch':raise OSError('fail')
        return type('Child',(),{'pid':999})()
    assert apply.run_handoff(request,token,42,root=root,executable=helper,wait=wait,launch=launch)==1
    assert origin.read_bytes()==b'MZold-fixture' and package.exists()
    assert json.loads((request.parent/'status.json').read_text('utf-8'))['state']=='failed'
    if change!='launch':assert not calls


def test_installer_failed_exit_does_not_launch_app(tmp_path):
    root,request,token,helper,package,origin=plan_fixture(tmp_path,kind='installer')
    calls=[]
    def launch(args,**kwargs):
        calls.append(args);return type('Child',(),{'wait':lambda self:5})()
    assert apply.run_handoff(request,token,42,root=root,executable=helper,wait=lambda *args:None,launch=launch)==1
    assert len(calls)==1 and '/CURRENTUSER' in calls[0] and '/NORESTART' in calls[0]
    assert '/DIR='+str(origin.parent) in calls[0]
    assert origin.exists() and package.exists()


def test_installer_success_requires_new_manifest_then_launch(tmp_path):
    root,request,token,helper,package,origin=plan_fixture(tmp_path,kind='installer')
    calls=[]
    def launch(args,**kwargs):
        calls.append(args)
        if len(calls)==1:
            _atomic_json(origin.parent/'application.json',{'schema':'ptb-desktop-release/1','version':VERSION,'entry':'PhoneticToolbox.exe'})
            _atomic_json(origin.parent/'_internal/desktop-bundle.json',bundle())
        return type('Child',(),{'wait':lambda self:0,'pid':777})()
    assert apply.run_handoff(request,token,42,root=root,executable=helper,wait=lambda *args:None,launch=launch)==0
    assert calls[1]==[str(origin)]


@pytest.mark.skipif(sys.platform!='win32',reason='Windows process identity handle')
def test_real_owned_handle_survives_exit_and_rejects_pid_mismatch():
    proc=subprocess.Popen([sys.executable,'-c','import time;time.sleep(0.2)'],creationflags=subprocess.CREATE_NO_WINDOW)
    api=apply.kernel();handle=api.OpenProcess(0x100000|0x1000,False,proc.pid)
    assert handle
    try:
        with pytest.raises(UpdateError):apply.wait_parent(handle,proc.pid+1,10)
        apply.wait_parent(handle,proc.pid,3000)
        assert proc.wait(timeout=3)==0
        apply.wait_parent(handle,proc.pid,10)
    finally:
        api.CloseHandle(handle);proc.wait(timeout=3)
