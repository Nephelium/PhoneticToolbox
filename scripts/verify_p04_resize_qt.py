"""P04-RESIZE: hidden actual Qt host + production bundle; no task DB or analysis.

Synthetic WAV/XLSX, real local FileProvider and read API. Native Qt mouse events
verify pointer capture. The owned service is stopped by Workbench.close().
"""
import json
import math
import struct
import time
import wave
from pathlib import Path
from uuid import uuid4

from openpyxl import Workbook
from PyQt6.QtCore import QEventLoop, QPoint, QTimer, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QFileDialog
from ptb_desktop.host import Workbench, register_scheme

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT / 'output/validation/p04-resize' / ('qt-' + uuid4().hex)
    inputs = out / 'inputs'
    inputs.mkdir(parents=True)
    with wave.open(str(inputs / 'synthetic.wav'), 'wb') as audio:
        audio.setparams((1, 2, 8000, 8000, 'NONE', 'not compressed'))
        audio.writeframes(b''.join(struct.pack('<h', round(10000 * math.sin(i * math.pi / 20))) for i in range(8000)))
    book = Workbook()
    book.active.append(['Time_s', 'pF0', 'Intensity'])
    for i in range(100):
        book.active.append([i / 100, 200 + i, 60 + i / 100])
    book.save(inputs / 'synthetic.xlsx')
    register_scheme()
    app = QApplication(['P04 resize verification'])
    window = Workbench(ROOT / 'frontend/dist', test=True, vocal_profile=out / 'vocal')
    window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    window.resize(1440, 900)
    window.show()
    QFileDialog.getExistingDirectory = lambda *args, **kwargs: str(inputs)
    checks = []
    report = {'success': False, 'checks': checks, 'platform': 'Windows Qt / hidden native window', 'schema_applied': []}

    def js(code):
        loop = QEventLoop()
        values = []
        window.page.runJavaScript(code, lambda value: (values.append(value), loop.quit()))
        QTimer.singleShot(5000, loop.quit)
        loop.exec()
        return values[0] if values else None

    def pause(ms=150):
        loop = QEventLoop()
        QTimer.singleShot(ms, loop.quit)
        loop.exec()

    def until(code):
        end = time.monotonic() + 20
        while time.monotonic() < end:
            if js(code):
                return
            pause()
        raise AssertionError('UI timeout: ' + code)

    def click(text):
        js('[...document.querySelectorAll("button")].find(b=>b.offsetParent&&b.textContent.trim()===' + json.dumps(text) + ')?.click()')
        pause()

    def width(selector):
        return js('document.querySelector(' + json.dumps(selector) + ').getBoundingClientRect().width/(document.documentElement.style.zoom||1)')

    def drag(label, delta):
        selector = '[role="separator"][aria-label="' + label + '"]'
        point = js('(()=>{const el=document.querySelector(' + json.dumps(selector) + ');el.scrollIntoView({block:"nearest"});const r=el.getBoundingClientRect();return [r.x+r.width/2,r.y+Math.min(50,r.height/2)];})()')
        target = window.view.focusProxy() or window.view
        start = target.mapFrom(window.view, QPoint(round(point[0]), round(point[1])))
        QTest.mousePress(target, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start)
        for i in range(1, 7):
            QTest.mouseMove(target, start + QPoint(round(delta * i / 6), 0), 20)
        QTest.mouseRelease(target, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start + QPoint(delta, 0))
        pause(250)

    try:
        until('document.querySelector(".host-badge")?.textContent==="本地桌面"')
        click('参数估计')
        until('document.querySelectorAll(".workbench-columns .panel-resize-handle:not([hidden])").length===2')
        assert abs(width('.workbench-columns>.file-panel') - 220) < 2
        drag('音频列表宽度', 45)
        assert abs(width('.workbench-columns>.file-panel') - 265) < 3
        drag('参数与结果宽度', -35)
        assert abs(width('.workbench-columns>.parameter-summary') - 285) < 3
        window.view.grab().save(str(out / 'M01-resized.png'))
        checks.append('Actual Qt native pointer drag and capture on both M01 boundaries')
        loaded = QEventLoop()
        window.view.loadFinished.connect(loaded.quit)
        window.view.reload()
        QTimer.singleShot(20000, loaded.quit)
        loaded.exec()
        window.view.loadFinished.disconnect(loaded.quit)
        until('document.querySelector(".host-badge")?.textContent==="本地桌面"')
        click('参数估计')
        until('!!document.querySelector(".workbench-columns")')
        assert abs(width('.workbench-columns>.file-panel') - 265) < 3
        checks.append('Qt profile reload restores width through ptbapp origin')
        click('参数显示')
        click('选择音频目录')
        until('document.querySelectorAll(".m02-files button").length===1')
        click('synthetic.wav')
        until('!!document.querySelector(".empty-plot")&&document.querySelectorAll(".m02-parameters input").length===2')
        assert js('document.querySelectorAll(".parameter-curve").length') == 0
        assert js('document.querySelectorAll(".m02-parameters input:checked").length') == 0
        drag('参数与图窗宽度', -30)
        assert abs(width('.m02-columns>.parameter-summary') - 290) < 3
        click('全选可见参数')
        click('将 2 项分配到图窗')
        until('document.querySelectorAll(".parameter-curve").length===2')
        click('保存绘图配置')
        window.view.grab().save(str(out / 'M02-assigned.png'))
        checks.append('Real local WAV/XLSX read starts empty, explicit selection plots two curves; M02 native boundary')
        click('设置')
        until('!!document.querySelector(".font-settings")')
        assert js('document.querySelectorAll("dialog[open]").length') == 0
        click('使用说明')
        assert js('document.querySelectorAll("dialog[open]").length') == 0
        click('设置')
        assert js('document.querySelectorAll("#tab-settings").length') == 1
        js('document.querySelector("[aria-label=放大页面]").click()')
        until('document.documentElement.style.zoom==="1.1"')
        click('参数显示')
        before = width('.m02-columns>.file-panel')
        report['zoom_before_drag'] = js('({zoom:document.documentElement.style.zoom,root:document.querySelector(".m02-columns").getBoundingClientRect().width,offset:document.querySelector(".m02-columns").offsetWidth,left:document.querySelector(".m02-columns>.file-panel").getBoundingClientRect().width,handle:[...document.querySelectorAll("[role=separator]")].filter(x=>!x.hidden).map(x=>({label:x.getAttribute("aria-label"),max:x.getAttribute("aria-valuemax"),value:x.getAttribute("aria-valuenow")}))})')
        drag('音频列表宽度', 33)
        report['zoom_drag_widths'] = [before, width('.m02-columns>.file-panel')]
        assert abs(width('.m02-columns>.file-panel') - before - 30) < 3, str(report['zoom_drag_widths'])
        checks.append('Settings/help use unique tabs; Qt 110% zoom uses logical-pixel drag distance')
        report['success'] = True
    except Exception as error:
        report['error'] = str(error)
        window.view.grab().save(str(out / 'failed.png'))
        raise
    finally:
        (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(out, flush=True)
        window.closing = True
        window.close()
        app.processEvents()


if __name__ == '__main__':
    main()
