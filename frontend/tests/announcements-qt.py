"""Owned source Qt window for the external announcements CDP recipe."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from workbench_source import configure

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--frontend-dist',type=Path,required=True);parser.add_argument('--scratch',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    configure()
    from PyQt6.QtCore import QCoreApplication,QEvent,QEventLoop,QTimer,Qt,PYQT_VERSION_STR,QT_VERSION_STR
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtGui import QDesktopServices
    from ptb_desktop.host import Workbench,register_scheme
    register_scheme();app=QApplication(['PTB-owned-announcements'])
    opened=[];original=QDesktopServices.openUrl
    QDesktopServices.openUrl=lambda url:opened.append(url.toString()) or True
    window=Workbench(args.frontend_dist,test=True,jobs_path=None,vocal_resources=ROOT/'resources/vocal_tract/native',vocal_profile=args.scratch/'vocal')
    assert window.profile.isOffTheRecord()
    window.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating,True);window.resize(1100,800);window.setWindowTitle('PhoneticToolbox · 公告自动验证');window.show()
    report=dict(success=False,scope='Actual current-source Workbench, independent production frontend, off-the-record Qt profile, default graphics; controlled public network fixture through external CDP; no physical device acceptance',qt=QT_VERSION_STR,pyqt=PYQT_VERSION_STR,indexSha256=hashlib.sha256((args.frontend_dist/'index.html').read_bytes()).hexdigest())
    def commands():
        if (args.scratch/'close-request.json').exists():window.closing=True;window.close()
        (args.out/'native-opened.json').write_text(json.dumps(opened,ensure_ascii=False),encoding='utf-8')
    timer=QTimer();timer.timeout.connect(commands);timer.start(100)
    def pause(ms):
        loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
    try:
        app.exec();report['normalEventLoopReturn']=True
    finally:
        timer.stop();window.closing=True;window.close();pause(500)
        window.page.deleteLater();QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete);pause(300)
        window.profile.deleteLater();QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete);pause(500)
        window.deleteLater();QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete);pause(100)
        QDesktopServices.openUrl=original
        report.update(success=True,cleanupComplete=True,openedExternal=opened)
        (args.out/'qt-process.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return 0

if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception:traceback.print_exc();raise
