"""Final ZIP through real Qt/native bridge/API, fresh registry, and MFA probe."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from uuid import uuid4

ROOT=Path(__file__).resolve().parents[1]
for part in ('desktop/src','backend/src','packages/phonetic_core/src','scripts'):
    sys.path.insert(0,str(ROOT/part))
os.environ['PYTHONPATH']=os.pathsep.join(str(ROOT/p) for p in ('desktop/src','backend/src','packages/phonetic_core/src'))
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS','--disable-gpu --disable-gpu-compositing --mute-audio')
for name in ('QT_ENABLE_HIGHDPI_SCALING','QT_SCALE_FACTOR','QT_SCREEN_SCALE_FACTORS'):
    os.environ.pop(name,None)


def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--screenshots-only',action='store_true');args=p.parse_args()
    from verification_artifacts import finish,write_json
    from ptb_worker.local_workspace import prepare_workspace
    from ptb_worker.mfa.components import digest
    from ptb_worker.mfa.release_catalog import RELEASE
    from ptb_desktop.host import Workbench,register_scheme
    from PyQt6.QtCore import QEventLoop,QTimer,Qt,QPoint
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication,QFileDialog
    out=ROOT/'output/validation'/('m11d-'+uuid4().hex[:6]);out.mkdir()
    scratch=out/'scratch';scratch.mkdir();components=scratch/'c'
    marker=dict(schema='ptb-test-run/1',owner_pid=os.getpid(),state='active',recipe='Final ZIP/native Qt import; bundled Mandarin MFA public probe; folder import; isolated state')
    write_json(out/'run.json',marker)
    os.environ['PTB_M11_COMPONENT_ROOT']=str(components);os.environ.pop('PTB_M11_BUNDLED_COMPONENTS',None)
    db,cache=prepare_workspace(scratch/'state',ROOT/'backend/migrations')
    assert digest(args.archive)==RELEASE['sha256']
    choice={'file':args.archive,'folder':None}
    QFileDialog.getOpenFileName=lambda *a,**k:(str(choice['file']),'')
    QFileDialog.getExistingDirectory=lambda *a,**k:str(choice['folder']) if choice['folder'] else ''
    register_scheme();app=QApplication(['PTB-owned-M11-component-check'])
    w=Workbench(ROOT/'frontend/dist',test=True,jobs_path=db,local_files_root=cache,vocal_profile=scratch/'vocal')
    w.resize(w.screen().geometry().width(),w.screen().geometry().height())
    w.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen);w.showMaximized();w.view.setZoomFactor(1.);w.page.setAudioMuted(True)
    report=dict(success=False,archive_sha256=RELEASE['sha256'],scope='Windows source Qt; actual final ZIP; isolated fresh registry; public synthesized Mandarin probe; no new-machine or live-server claim',captures=[],stages=[],errors=[])
    def pause(ms=150):
        loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
    def js(code):
        loop=QEventLoop();values=[];w.page.runJavaScript(code,lambda v:(values.append(v),loop.quit()));QTimer.singleShot(6000,loop.quit);loop.exec()
        if not values:raise RuntimeError('JS timeout')
        return values[0]
    def until(code,seconds=180):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            if js(code):return
            pause()
        raise AssertionError(code+'\n'+str(js('document.querySelector(".mfa-page")?.innerText')))
    def click(label):
        box=js('(()=>{const e=[...document.querySelectorAll("button")].find(b=>b.offsetParent&&b.textContent.trim()==='+json.dumps(label)+');if(!e||e.disabled)return null;e.scrollIntoView({block:"nearest"});const r=e.getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2};})()')
        if not box:raise AssertionError('Unavailable: '+label)
        QTest.mouseClick(w.view.focusProxy() or w.view,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,QPoint(round(box['x']),round(box['y'])));pause()
    def capture(name):
        pause(350);w.view.repaint();pix=w.view.grab();raw_size=[pix.width(),pix.height()]
        if raw_size==[2561,1601]:pix=pix.copy(0,0,2560,1600)
        if args.screenshots_only:assert [pix.width(),pix.height()]==[2560,1600] and pix.devicePixelRatio()==1.5
        path=out/(name+'.png');pix.save(str(path),'PNG')
        report['captures'].append(dict(name=name,size=[pix.width(),pix.height()],dpr=pix.devicePixelRatio(),maximized=w.isMaximized(),theme=js('document.documentElement.dataset.theme'),pageZoom=1,sha256=digest(path)))
    error=None
    try:
        until('!!document.querySelector("nav")&&document.fonts.status==="loaded"')
        js('document.documentElement.dataset.theme="light"');click('MFA 自动标注')
        until('document.querySelector(".mfa-page")?.getAttribute("aria-busy")==="false"')
        js('document.querySelector("[aria-label=\\"收起侧栏\\"]")?.click()')
        click('组件安装与环境检查');capture('component-install-light')
        assert not js('document.querySelector(".mfa-page")?.innerText.includes("可信发布清单 JSON")')
        if args.screenshots_only:
            report['scope']='Windows native Qt installation UI screenshot only; no installation in this run'
            report['success']=True
            print(str(out/'report.json'),flush=True)
            return
        click('选择组件压缩包')
        until('document.querySelector(".mfa-page")?.getAttribute("aria-busy")==="true"')
        started=time.monotonic()
        while js('document.querySelector(".mfa-page")?.getAttribute("aria-busy")==="true"'):
            stage=js('document.querySelector(".component-progress")?.innerText')
            if stage and (not report['stages'] or stage!=report['stages'][-1]):
                report['stages'].append(stage);print(stage,flush=True)
            if time.monotonic()-started>900:raise TimeoutError('Component import')
            pause(1000)
        assert js('document.querySelector(".mfa-page")?.innerText.includes("已检查并应用：mandarin_mfa.zip")'),js('document.querySelector(".mfa-page")?.innerText')
        registry=json.loads((components/'registry.json').read_text('utf8'))
        record=registry['runtimes'][0];report['zip_receipt']=json.loads(Path(record['receipt']).read_text('utf8'))
        assert report['zip_receipt']['success'] and any('正在检查 MFA' in s for s in report['stages'])
        report['zip_seconds']=time.monotonic()-started
        capture('component-ready-light')
        choice['folder']=Path(json.loads((components/'current.json').read_text('utf8'))['path'])
        click('选择已解压文件夹');until('document.querySelector(".mfa-page")?.getAttribute("aria-busy")==="true"')
        until('document.querySelector(".mfa-page")?.getAttribute("aria-busy")==="false"',600)
        assert js('document.querySelector(".mfa-page")?.innerText.includes("已检查并应用：mandarin_mfa.zip")'),js('document.querySelector(".mfa-page")?.innerText')
        after=json.loads((components/'registry.json').read_text('utf8'));assert len(after['runtimes'])==1
        report['folder_receipt']=json.loads(Path(after['runtimes'][0]['receipt']).read_text('utf8'))
        assert report['folder_receipt']['success']
        js('document.documentElement.dataset.theme="dark"');capture('component-ready-dark')
        report['success']=True
    except Exception as exc:
        error=exc;report['errors'].append(traceback.format_exc())
        try:capture('diagnostic')
        except Exception:pass
    finally:
        write_json(out/'report.json',report)
        w.closing=True;w.close();pause(500);app.quit()
        finish(out,marker,error=error)
    print(json.dumps(dict(success=report['success'],report=str(out/'report.json'),errors=report['errors']),ensure_ascii=False),flush=True)
    if error:raise error


if __name__=='__main__':main()
