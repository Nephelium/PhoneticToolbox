"""Source Qt seeds owned data; a real frozen helper installs and relaunches.

The older version number is a QA metadata fixture. The package and helper bytes,
Windows parent handle, installer/ZIP and next process are real, not dry adapters.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from uuid import uuid4

ROOT=Path(__file__).resolve().parents[1]
for folder in ('desktop/src','backend/src','packages/phonetic_core/src'):
    sys.path.insert(0,str(ROOT/folder))
os.environ['PYTHONPATH']=os.pathsep.join(str(ROOT/p) for p in ('desktop/src','backend/src','packages/phonetic_core/src'))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--package',required=True,type=Path);parser.add_argument('--origin',required=True,type=Path);parser.add_argument('--output',required=True,type=Path);parser.add_argument('--kind',choices=('portable','installer'),required=True);parser.add_argument('--port',required=True,type=int);parser.add_argument('--profile',type=Path);args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    if not out.is_relative_to(ROOT/'output/validation'):raise ValueError('Expected owned output')
    profile=args.profile.resolve() if args.profile else out/'profile'
    if args.profile:
        temp=Path(os.environ['TEMP']).resolve()
        if not profile.is_relative_to(temp) or not profile.name.startswith('PTB-Preview1-'):
            raise ValueError('Expected an owned Preview QA profile under the current user Temp directory')
    profile.mkdir();os.environ['LOCALAPPDATA']=str(profile)
    os.environ['QT_QPA_PLATFORM']='offscreen';os.environ['QTWEBENGINE_CHROMIUM_FLAGS']='--mute-audio --disable-gpu --autoplay-policy=no-user-gesture-required'
    from ptb_desktop.platform_paths import user_data_root,windows_workbench_storage
    from ptb_worker.local_workspace import prepare_workspace
    from ptb_desktop.host import Workbench,register_scheme
    from PyQt6.QtCore import QEventLoop,QTimer
    from PyQt6.QtWidgets import QApplication
    db,cache=prepare_workspace(user_data_root()/'local-preview-20260927',ROOT/'backend/migrations')
    register_scheme();app=QApplication(['PhoneticToolbox']);app.setApplicationName('PhoneticToolbox-v3')
    w=Workbench(ROOT/'frontend/dist',jobs_path=db,local_files_root=cache,vocal_profile=out/'vocal');w.showMaximized()
    def pause(ms=100):
        loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
    def js(code):
        loop=QEventLoop();box=[];w.page.runJavaScript(code,lambda x:(box.append(x),loop.quit()));QTimer.singleShot(6000,loop.quit);loop.exec();assert box;return box[0]
    def until(code):
        end=time.monotonic()+50
        while time.monotonic()<end:
            if js(code):return
            pause()
        raise AssertionError(code)
    def click(text):
        code='(()=>{const wanted='+json.dumps(text)+';const e=[...document.querySelectorAll("button")].find(e=>e.offsetParent&&[wanted,wanted+" *"].includes(e.textContent.trim()));if(!e||e.disabled)return false;e.click();return true})()'
        assert js(code),text;pause(300)
    from ptb_desktop.updates import _atomic_json
    w.updates_bridge.service.set_preferences({'autoCheck':False})
    until('!!document.querySelector(".home-page")');click('设置');click('深色')
    until('document.documentElement.dataset.theme==="dark"')
    assert Path(w.profile.persistentStoragePath())==windows_workbench_storage()
    assert js('(()=>{const e=[...document.querySelectorAll(".nav-item")].find(e=>e.title==="汉字转国际音标");e.click();return true})()')
    until('!!document.querySelector("textarea[aria-label=待转换汉字文本]")')
    js('(()=>{const e=document.querySelector("textarea[aria-label=待转换汉字文本]");e.value="更新保存测试";e.dispatchEvent(new Event("input",{bubbles:true}));})()');pause(300);click('保存本机草稿')
    preferences=js('Object.fromEntries(Object.keys(localStorage).filter(k=>k.startsWith("ptb.v3.")).map(k=>[k,localStorage.getItem(k)]))')
    marker=user_data_root()/'project-do-not-remove.json';marker.write_text('{"fixture":"owned research project"}','utf8');marker_hash=hashlib.sha256(marker.read_bytes()).hexdigest()
    w.closing=True;w.close();pause(1000);w.page.deleteLater();app.processEvents()
    updates=user_data_root()/'updates';download=updates/'downloads'/uuid4().hex/args.package.name;download.parent.mkdir(parents=True);shutil.copyfile(args.package,download)
    with download.open('rb') as stream:
        digest=hashlib.file_digest(stream,'sha256').hexdigest()
    from ptb_desktop.update_apply import prepare_handoff,launch_handoff
    request,plan=prepare_handoff(updates,download,args.kind,download.stat().st_size,digest,'3.0.0-preview.1','3.0.0-preview.0',origin=args.origin)
    # Seeding is complete. The actual frozen helper and next process receive
    # Windows-only PATH and no checkout/runtime environment bindings.
    for name in list(os.environ):
        if name.startswith(('PTB_', 'PYTHON', 'CONDA', '_PYI')) or name == 'VIRTUAL_ENV':
            os.environ.pop(name)
    temporary = out / 'temp'; temporary.mkdir()
    os.environ.update(TEMP=str(temporary), TMP=str(temporary),
                      PTB_OWNED_BOOTSTRAP_LOG=str(out/'bootstrap-error.log'),
                      PATH=os.pathsep.join((os.environ['SystemRoot']+'/System32', os.environ['SystemRoot'])),
                      QTWEBENGINE_REMOTE_DEBUGGING=str(args.port))
    report=dict(scope='Real frozen helper and next process with Windows-only PATH; source Qt seeded persistent data, older version metadata fixture; no real user data',kind=args.kind,request=str(request),origin=str(args.origin),output=str(out),profile=str(profile),storage=str(windows_workbench_storage()),port=args.port,preferences=preferences,project_marker=str(marker),project_sha256=marker_hash,closeAfter=True,developerEnvironmentRemoved=True)
    _atomic_json(out/'handoff.json',report)
    helper=launch_handoff(request,plan);report['helper_pid']=helper.pid;_atomic_json(out/'handoff.json',report)
    print(json.dumps(dict(handoff=str(out/'handoff.json'),helper_pid=helper.pid)),flush=True)
    # Exiting this actual parent releases the inherited wait handle.
    return 0

if __name__=='__main__':raise SystemExit(main())
