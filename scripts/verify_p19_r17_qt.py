"""Owned source Qt window: native Enter, M10 iframe help and chapter return."""
import json
import os
import sys
import time
from pathlib import Path
from uuid import uuid4


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / 'output/validation/p19-r17' / ('qt-' + uuid4().hex)
    out.mkdir(parents=True)
    # The verifier owns a hidden window and a private browser profile.
    os.environ['QT_QPA_PLATFORM'] = 'windows'
    os.environ['QTWEBENGINE_CHROMIUM_FLAGS'] = '--mute-audio'
    from v3_local_preview_entry import configure
    bundle = configure()
    from PyQt6.QtCore import QEventLoop, QTimer, QPoint, Qt, qVersion
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication
    from ptb_desktop.host import Workbench, register_scheme
    from ptb_worker.local_workspace import prepare_workspace
    database, files = prepare_workspace(out / 'state', bundle / 'backend/migrations')
    register_scheme()
    app = QApplication(['P19R17-owned-Qt-check'])
    window = Workbench(bundle / 'frontend/dist', test=True, jobs_path=database,
                       local_files_root=files, vocal_profile=out / 'vocal',
                       vocal_resources=bundle / 'resources/vocal_tract/native')
    window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
    window.showMaximized()
    report = dict(success=False, scope='Windows source, hidden Qt, isolated state; no physical DPI/DWM claim',
                  qt=qVersion(), modules=[], navigation=[])

    def pause(ms=70):
        loop = QEventLoop(); QTimer.singleShot(ms, loop.quit); loop.exec()

    def js(code):
        loop, box = QEventLoop(), []
        window.page.runJavaScript(code, lambda value: (box.append(value), loop.quit()))
        QTimer.singleShot(6000, loop.quit); loop.exec()
        assert box, 'JavaScript timeout'
        return box[0]

    def until(code):
        end = time.monotonic() + 35
        while time.monotonic() < end:
            value = js(code)
            if value:
                return value
            pause()
        raise AssertionError('Timed out: ' + code)

    def enter():
        QTest.keyClick(window.view.focusProxy() or window.view, Qt.Key.Key_Return); pause()

    titles = ['参数估计','参数显示','EGG 信号分析','LPC 谱图','唇形提取','声学参数合成',
              '发声类型合成','变速变调','语谱图转音频','生理参数合成','MFA 自动标注',
              'TextGrid标注','汉字转国际音标','音系归纳','感知实验','录音','国际音标表Plus']
    project = json.loads((bundle / 'frontend/dist/manual/project.json').read_text('utf-8'))
    try:
        until('!!document.querySelector(".nav-item")')
        for index, title in enumerate(titles, 1):
            name = json.dumps(title)
            until(f'(()=>{{const b=[...document.querySelectorAll(".nav-item")].find(b=>b.title==={name});if(!b)return false;b.click();return true;}})()')
            assert js('document.querySelectorAll(".module-manual-entry").length') == 0
            if index == 10:
                until('typeof document.querySelector("iframe")?.contentDocument?.getElementById("sharedHelp")?.onclick==="function"')
                geometry = js('(()=>{const d=document.querySelector("iframe").contentDocument,h=d.getElementById("sharedHelp"),r=d.getElementById("aboutButton");const a=h.getBoundingClientRect(),b=r.getBoundingClientRect();return {dy:Math.abs(a.y-b.y),dh:Math.abs(a.height-b.height)};})()')
                assert geometry['dy'] < 2 and geometry['dh'] < 2, geometry
                point = js('(()=>{const f=document.querySelector("iframe"),a=f.getBoundingClientRect(),b=f.contentDocument.getElementById("sharedHelp").getBoundingClientRect();return [a.x+b.x+b.width/2,a.y+b.y+b.height/2];})()')
                QTest.mouseClick(window.view.focusProxy() or window.view, Qt.MouseButton.LeftButton,
                                 Qt.KeyboardModifier.NoModifier, QPoint(round(point[0]), round(point[1])))
                until('document.querySelector(".manual-chapter-header h1")?.textContent==="生理参数合成"')
                js('[...document.querySelectorAll(".manual-reading-toolbar button")].find(b=>b.textContent.trim()==="返回 生理参数合成").focus()')
                enter()
                until('document.querySelector(".nav-item.selected")?.title==="生理参数合成"')
                js('document.querySelector("iframe").contentDocument.getElementById("sharedHelp").focus()')
            else:
                until('(()=>{const h=[...document.querySelectorAll("main button")].filter(b=>b.offsetParent&&b.textContent.trim()==="帮助");if(h.length!==1)return false;h[0].focus();return true;})()')
            enter()
            chapter = next(c['title'] for c in project['chapters'] if c['id'] == f'm{index:02d}')
            until('document.querySelector(".manual-chapter-header h1")?.textContent===' + json.dumps(chapter))
            js('(()=>{const b=[...document.querySelectorAll(".manual-reading-toolbar button")].find(b=>b.textContent.trim()===' + json.dumps('返回 ' + title) + ');b.focus();})()')
            enter()
            until('document.querySelector(".nav-item.selected")?.title===' + name)
            report['modules'].append(dict(id=f'M{index:02d}', chapter=chapter, nativeEnter=True, returned=True))
        for mode in ['light', 'dark']:
            js('(()=>{[...document.querySelectorAll(".nav-item")].find(b=>b.title==="EGG 信号分析").click();document.documentElement.dataset.theme=' + json.dumps(mode) + ';document.documentElement.dataset.buttonStyle="plain";document.querySelector(".nav-item.selected").focus();})()')
            QTest.keyClick(window.view.focusProxy() or window.view, Qt.Key.Key_Tab); pause(250)
            style = js('(()=>{const n=getComputedStyle(document.querySelector(".nav-item.selected")),t=getComputedStyle(document.querySelector(".tab-wrap.active>button"));return {navOutline:n.outlineStyle,tabOutline:t.outlineStyle,tabShadow:t.boxShadow};})()')
            assert style == dict(navOutline='none', tabOutline='none', tabShadow='none'), style
            window.view.grab().save(str(out / ('navigation-' + mode + '.png')))
            report['navigation'].append(dict(mode=mode, **style))
        report['success'] = True
    except Exception:
        import traceback
        report['error'] = traceback.format_exc()
        report['focus'] = js('({parent:document.activeElement?.tagName,child:document.querySelector("iframe")?.contentDocument?.activeElement?.id})')
        window.view.grab().save(str(out / 'failure.png'))
    finally:
        window.closing = True; window.close(); window.page.deleteLater(); app.processEvents()
        (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(success=report['success'], modules=len(report['modules']), out=str(out), error=report.get('error')), ensure_ascii=False))
    return 0 if report['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
