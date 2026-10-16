"""Check native M10 dropdowns in an owned Windows Qt workbench."""
import json
import os
from pathlib import Path
import time

from workbench_source import configure
from verification_artifacts import verification_run, write_json

ROOT = Path(__file__).resolve().parents[1]
configure(ROOT)
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS', '--mute-audio')


def main():
    from PyQt6.QtCore import QCoreApplication, QEvent, QEventLoop, QPoint, QTimer, Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication
    from ptb_desktop.host import Workbench, register_scheme

    register_scheme()
    app = QApplication(['M10-dropdown-owned-QA'])
    with verification_run('m10-dropdown', 'qt', 'Native source-mode popup; all option colours, contrast, dark/light/dark switching, keyboard selection') as (out, scratch):
        w = Workbench(ROOT / 'frontend/dist', test=True,
                      vocal_profile=scratch / 'vocal', start_module='M10')
        w.resize(1440, 900)
        w.show()
        report = {'passed': False, 'cases': []}

        def pause(ms=100):
            loop = QEventLoop()
            QTimer.singleShot(ms, loop.quit)
            loop.exec()

        def js(code):
            loop, values = QEventLoop(), []
            w.page.runJavaScript(code, lambda v: (values.append(v), loop.quit()))
            QTimer.singleShot(5000, loop.quit)
            loop.exec()
            assert values, 'JavaScript timeout'
            return values[0]

        def local(code):
            return js('(()=>{const f=document.querySelector("iframe"),d=f?.contentDocument,win=f?.contentWindow;if(!d)return null;' + code + '})()')

        def until(code):
            end = time.monotonic() + 60
            while not local('return ' + code):
                assert time.monotonic() < end, code
                pause()

        def luminance(rgb):
            c = [float(v.strip()) / 255 for v in rgb.removeprefix('rgb(').removesuffix(')').split(',')]
            c = [v / 12.92 if v <= .04045 else ((v + .055) / 1.055) ** 2.4 for v in c]
            return sum(a * b for a, b in zip(c, [.2126, .7152, .0722]))

        try:
            until('d.body.dataset.engineState==="ready"')
            until('d.body.dataset.posePending==="false"')
            for mode in ['dark', 'light', 'dark']:
                js('document.documentElement.dataset.theme=' + json.dumps(mode))
                until('d.documentElement.dataset.theme===' + json.dumps(mode))
                local('d.querySelector("#tab-sound").click()')
                pause(300)
                rows = json.loads(local('return JSON.stringify([...d.querySelectorAll("select option")].map(e=>{const s=win.getComputedStyle(e);return {select:e.parentElement.id,text:e.textContent,hidden:e.hidden,disabled:e.disabled,bg:s.backgroundColor,fg:s.color}}))'))
                assert rows and all(r['bg'].startswith('rgb(') for r in rows), rows
                for row in rows:
                    a, b = sorted([luminance(row['bg']), luminance(row['fg'])])
                    row['contrast'] = (b + .05) / (a + .05)
                    if not row['hidden']:
                        assert row['contrast'] >= 4.5, row
                xy = local('const r=d.querySelector("#sourceMode").getBoundingClientRect(),fbox=f.getBoundingClientRect();return [fbox.x+r.x+r.width/2,fbox.y+r.y+r.height/2]')
                QTest.mouseClick(w.view.focusProxy(), Qt.MouseButton.LeftButton, pos=QPoint(round(xy[0]), round(xy[1])))
                pause(300)
                popups = [p for p in app.topLevelWidgets() if p.isVisible() and p != w]
                if not popups:
                    local('d.querySelector("#sourceMode").focus()')
                    QTest.keyClick(w.view.focusProxy(), Qt.Key.Key_Space)
                    pause(300)
                    popups = [p for p in app.topLevelWidgets() if p.isVisible() and p != w]
                write_json(out / 'widgets.json', [{'class':p.metaObject().className(),'visible':p.isVisible(),'type':int(p.windowType())} for p in app.topLevelWidgets()])
                assert popups, 'Native select popup did not open'
                popup = popups[0]
                image = popup.grab().toImage()
                image.save(str(out / f'{len(report["cases"])}-{mode}-popup.png'))
                # The lower unselected row has a wide text-free background area.
                pixel = image.pixelColor(image.width() - 20, image.height() - 20)
                if mode == 'dark':
                    assert max(pixel.red(), pixel.green(), pixel.blue()) < 100, pixel.name()
                else:
                    assert min(pixel.red(), pixel.green(), pixel.blue()) > 150, pixel.name()
                QTest.keyClick(popup, Qt.Key.Key_Down)
                QTest.keyClick(popup, Qt.Key.Key_Return)
                pause(300)
                assert local('return d.querySelector("#sourceMode").value') == 'voiceless'
                local('const e=d.querySelector("#sourceMode");e.value="voiced";e.dispatchEvent(new win.Event("change"))')
                report['cases'].append({'mode': mode, 'options': rows, 'popup_background': pixel.name(), 'keyboard_selection': True})
            report['passed'] = True
        finally:
            write_json(out / 'report.json', report)
            w.accept_close()
            w.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            pause(400)
        print(out / 'report.json', flush=True)


if __name__ == '__main__':
    main()
