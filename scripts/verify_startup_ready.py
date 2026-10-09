"""Actual Qt startup-readiness handshake in a new, disposable QA profile."""
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
for name in ('desktop/src','backend/src','packages/phonetic_core/src'):
    sys.path.insert(0,str(ROOT/name))


def main():
    out=Path(sys.argv[1]).absolute();out.mkdir(parents=True,exist_ok=False)
    profile=out/'profile';profile.mkdir()
    os.environ['LOCALAPPDATA']=str(profile)
    os.environ['QT_QPA_PLATFORM']='offscreen'
    os.environ['QTWEBENGINE_CHROMIUM_FLAGS']='--mute-audio --disable-gpu'
    os.environ['PYTHONPATH']=os.pathsep.join(str(ROOT/name) for name in ('desktop/src','backend/src','packages/phonetic_core/src'))
    from ptb_desktop.startup_ready import ReadyEvent,ENVIRONMENT
    from ptb_desktop.host import Workbench,register_scheme
    from ptb_worker.local_workspace import prepare_workspace
    from PyQt6.QtCore import QEventLoop,QTimer
    from PyQt6.QtWidgets import QApplication
    database,files=prepare_workspace(out/'workspace',ROOT/'backend/migrations')
    ready=ReadyEvent();os.environ[ENVIRONMENT]=ready.name
    register_scheme();app=QApplication(['PTB-startup-owned-QA'])
    window=Workbench(ROOT/'frontend/dist',test=True,jobs_path=database,local_files_root=files,
        vocal_profile=out/'vocal',vocal_resources=ROOT/'resources/vocal_tract/native',
        reaper_binary=ROOT/'phonetic_toolbox/core/acoustic/reaper.exe')
    def pause(ms):
        loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
    started=time.monotonic()
    try:
        assert not ready.is_set()
        window.show()
        while not ready.is_set() and time.monotonic()-started<40:pause(25)
        assert ready.is_set(),'Workbench readiness event not signalled'
        assert window._startup_presentation.done
        answer=[];loop=QEventLoop()
        window.page.runJavaScript('!!document.querySelector(".app-shell")&&!!document.querySelector(".home-page")&&document.fonts.status==="loaded"&&!!window.qt',lambda value:(answer.append(value),loop.quit()))
        QTimer.singleShot(5000,loop.quit);loop.exec();assert answer==[True],answer
        seconds=time.monotonic()-started;pause(200)
        assert window.grab().save(str(out/'ready.png'))
        window.closing=True;window.close();window.page.deleteLater();app.processEvents()
        assert window.service.exit_code==0
        report={'success':True,'readySeconds':seconds,'pageAndFontsReady':True,
                'scope':'actual Qt source; owned isolated workspace/profile; offscreen software rendering',
                'serviceExitCode':window.service.exit_code}
        (out/'report.json').write_text(json.dumps(report,indent=2),'utf8')
        print(json.dumps(report),flush=True)
    finally:
        ready.close();os.environ.pop(ENVIRONMENT,None)
        if window.service.process and window.service.process.poll() is None:
            window.closing=True;window.close()


if __name__=='__main__':
    try:main()
    except Exception:
        target=Path(sys.argv[1]);target.mkdir(parents=True,exist_ok=True)
        (target/'failure.txt').write_text(traceback.format_exc(),'utf8')
        raise SystemExit(1)
