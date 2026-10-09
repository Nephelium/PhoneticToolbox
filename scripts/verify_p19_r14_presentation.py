"""Owned visible Windows presentation checks, usable inside the frozen EXE."""
import ctypes
import json
import os
import sys
import time
import traceback
from pathlib import Path


def verify(bundle, out, stage='shown'):
    if stage not in {'shown', 'ready'}:
        raise ValueError('Unknown presentation stage')
    os.environ['QT_QPA_PLATFORM'] = 'windows'
    os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS', '--mute-audio')
    if '--disable-gpu' in os.environ['QTWEBENGINE_CHROMIUM_FLAGS']:
        raise ValueError('Presentation check requires default GPU rendering')
    from PyQt6.QtCore import QEventLoop, QTimer, QPoint, Qt, qVersion, QObject, QEvent
    from PyQt6.QtWidgets import QApplication
    from ptb_desktop.host import Workbench, register_scheme
    from ptb_worker.local_workspace import prepare_workspace
    out.mkdir(parents=True, exist_ok=False)
    database, files = prepare_workspace(out / 'state', bundle / 'backend/migrations')
    register_scheme()
    app = QApplication(['P19-R14-owned-window-check'])
    app.setApplicationName('P19-R14-owned-QA')
    window = Workbench(bundle / 'frontend/dist', test=True, jobs_path=database,
        local_files_root=files, vocal_profile=out / 'vocal',
        vocal_resources=bundle / 'resources/vocal_tract/native')
    window.setWindowTitle('PhoneticToolbox 3.0 · 窗口验证')
    window.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
    started = time.monotonic()
    report = dict(success=False, stage=stage, qt=qVersion(),
        mode='frozen' if getattr(sys, 'frozen', False) else 'source',
        scope='Windows visible owned window, default GPU, fresh private profile and workspace',
        flags=os.environ['QTWEBENGINE_CHROMIUM_FLAGS'], samples=[], events=[], terminations=[])
    window.page.renderProcessTerminated.connect(lambda status, code: report['terminations'].append([status.name, code]))
    window.view.loadFinished.connect(lambda ok: report['events'].append(dict(event='load-finished', ok=ok, ms=round((time.monotonic()-started)*1000))))
    class CoverEvents(QObject):
        def eventFilter(self, watched, event):
            if event.type() in {QEvent.Type.Show, QEvent.Type.Hide}:
                report['events'].append(dict(event='cover-'+('show' if event.type()==QEvent.Type.Show else 'hide'), ms=round((time.monotonic()-started)*1000), frames=window._first_maximize.frames))
            return False
    cover_events = CoverEvents(window)
    window._first_maximize.cover.installEventFilter(cover_events)

    def pause(ms):
        loop = QEventLoop()
        QTimer.singleShot(ms, loop.quit)
        loop.exec()

    def js(code):
        loop, box = QEventLoop(), []
        window.page.runJavaScript(code, lambda value: (box.append(value), loop.quit()))
        QTimer.singleShot(6000, loop.quit)
        loop.exec()
        assert box, 'JavaScript timeout'
        return box[0]

    def until(code):
        end = time.monotonic() + 40
        while time.monotonic() < end:
            if js(code):
                return
            pause(50)
        raise AssertionError(code)

    def click(text, selector='button'):
        assert js('(()=>{const b=[...document.querySelectorAll('+json.dumps(selector)+')].find(b=>b.offsetParent&&(b.matches(".nav-item")?b.querySelector(":scope>span:not([aria-hidden])")?.textContent.trim():b.textContent.trim())==='+json.dumps(text)+');if(!b)return false;b.click();return true;})()'), text

    user = ctypes.windll.user32
    user.WindowFromPoint.restype = ctypes.c_void_p
    user.GetAncestor.restype = ctypes.c_void_p
    user.GetForegroundWindow.restype = ctypes.c_void_p
    user.SendMessageW.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t, ctypes.c_ssize_t]
    class POINT(ctypes.Structure):
        _fields_ = [('x', ctypes.c_long), ('y', ctypes.c_long)]

    def sample(cycle, elapsed):
        guard = window._first_maximize
        value = dict(cycle=cycle, ms=round(elapsed), cover=guard.active, frames=guard.frames,
                     ready=guard.document_ready, captured=False)
        report['samples'].append(value)
        # Do not capture during the OS animation, or if any sampled corner is
        # occupied by another app. QWidget.grab is not used for this observation.
        if elapsed < 300:
            return
        point = window.view.mapToGlobal(QPoint(0, 0))
        ratio = window.devicePixelRatioF()
        for x, y in [(8, 8), (window.view.width()-8, 8), (8, window.view.height()-8),
                     (window.view.width()-8, window.view.height()-8), (window.view.width()//2, window.view.height()//2)]:
            hwnd = user.WindowFromPoint(POINT(round((point.x()+x)*ratio), round((point.y()+y)*ratio)))
            if user.GetAncestor(ctypes.c_void_p(hwnd or 0), 2) != int(window.winId()):
                return
        picture = window.screen().grabWindow(0, point.x(), point.y(), window.view.width(), window.view.height()).toImage()
        assert not picture.isNull()
        colors = [picture.pixelColor(x, y) for x in range(0, picture.width(), max(1, picture.width()//32))
                  for y in range(0, picture.height(), max(1, picture.height()//24))]
        value.update(captured=True, black_fraction=sum(max(c.red(), c.green(), c.blue())<12 for c in colors)/len(colors),
                     colors=len({c.rgb() for c in colors}), size=[picture.width(), picture.height()])
        assert value['black_fraction'] < .95, value
        if elapsed > 1100:
            picture.save(str(out / f'maximized-{cycle}.png'))

    try:
        window.show()
        pause(50)
        if stage == 'ready':
            until('!!document.querySelector(".home-page")&&document.fonts.status==="loaded"')
        for cycle in [0, 1]:
            start = time.monotonic()
            user.SendMessageW(int(window.winId()), 0x112, 0xF030, 0)
            end = time.monotonic()+1
            while not window.isMaximized() and time.monotonic() < end:
                pause(5)
            assert window.isMaximized() and window._first_maximize.used
            if cycle == 0:
                assert any(e['event']=='cover-show' for e in report['events']), 'First transition did not show its cover'
            for target in range(0, 1201, 100):
                elapsed = (time.monotonic()-start)*1000
                if target > elapsed:
                    pause(round(target-elapsed))
                sample(cycle, (time.monotonic()-start)*1000)
            until('!!document.querySelector(".home-page")&&document.fonts.status==="loaded"')
            end = time.monotonic()+3
            while window._first_maximize.active and time.monotonic() < end:
                pause(20)
            assert not window._first_maximize.active and window._first_maximize.frames >= 2, 'Cover did not retire after resized frames'
            renderer = window.page.renderProcessPid()
            report['events'].append(dict(event='maximized', cycle=cycle, ms=round((time.monotonic()-started)*1000), renderer=renderer, frames=window._first_maximize.frames))
            if cycle == 0:
                js('window.qaPresentationToken="same-page";window.qaPresentationGL=document.createElement("canvas").getContext("webgl");')
            else:
                assert js('qaPresentationToken==="same-page"')
                assert renderer == report['events'][-2]['renderer']
            user.SendMessageW(int(window.winId()), 0x112, 0xF120, 0)
            pause(250)
        report['webgl'] = js('!!qaPresentationGL&&!qaPresentationGL.isContextLost()')
        assert report['webgl'] and not report['terminations']
        assert sum(s['captured'] for s in report['samples']) >= 4, 'Owned window not visible for pixel checks'
        assert js('[...document.querySelectorAll(".nav-item")].some(e=>e.textContent.trim()==="声学参数合成")&&[...document.querySelectorAll(".nav-item")].some(e=>e.textContent.trim()==="生理参数合成")')
        click('声学参数合成', '.nav-item')
        until('!!document.querySelector("[aria-label=声学参数合成工作区]")')
        assert js('document.querySelector(".tab-wrap.active")?.textContent.includes("声学参数合成")')
        click('生理参数合成', '.nav-item')
        until('!!document.querySelector("iframe[title=生理参数合成]")?.contentDocument?.querySelector(".workspace")')
        report['titles'] = ['声学参数合成', '生理参数合成']
        frame = 'document.querySelector("iframe[title=生理参数合成]").contentDocument'
        assert js('(()=>{const d='+frame+';return ["aboutButton","sharedReferences"].every(id=>d.getElementById(id)?.textContent.trim()==="方法与引用"&&!!d.getElementById(id).querySelector("svg"));})()')
        report['m10_reference_icons'] = True
        titles = ['参数估计','参数显示','EGG 信号分析','LPC 谱图','唇形提取','声学参数合成','发声类型合成','变速变调','语谱图转音频','生理参数合成','MFA 自动标注','TextGrid标注','汉字转国际音标','音系归纳','感知实验','录音','国际音标表Plus']
        report['modules'] = []
        for index, title in enumerate(titles, 1):
            click(title, '.nav-item')
            if index == 10:
                continue  # Actual iframe controls checked above.
            until('[...document.querySelectorAll("main button")].some(b=>b.offsetParent&&b.textContent.trim()==="方法与引用")')
            buttons = js('([...document.querySelectorAll("main button")].filter(b=>b.offsetParent).map(b=>({text:b.textContent.trim(),icon:!!b.querySelector("svg")})))')
            methods = [b for b in buttons if b['text']=='方法与引用']
            assert methods and all(b['icon'] for b in methods), title
            directories = [b for b in buttons if b['text']=='打开音频目录']
            if index in {1,2,3,4,6,7,8,11,12}:
                assert len(directories)==1 and directories[0]['icon'], (title, directories)
            if index == 9:
                click('音频藏信息')
                until('[...document.querySelectorAll("main button")].some(b=>b.offsetParent&&b.textContent.trim()==="打开音频目录"&&!!b.querySelector("svg"))')
            assert not any(b['text'] in {'方法与来源','使用说明','操作说明','使用帮助','参数帮助','选择音频目录','打开 WAV 目录'} for b in buttons), title
            report['modules'].append(dict(id=f'M{index:02}', title=title, method_icon=True, audio_directories=1 if index==9 else len(directories)))
        report['success'] = True
    except Exception:
        report['error'] = traceback.format_exc()
    finally:
        window.closing = True
        window.close()
        window.page.deleteLater()
        app.processEvents()
        (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return 0 if report['success'] else 1


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--stage', choices=['shown', 'ready'], default='shown')
    args = parser.parse_args()
    from v3_local_preview_entry import configure
    raise SystemExit(verify(configure(), args.out.absolute(), args.stage))
