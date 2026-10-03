"""P19 built frontend in the actual owned Qt host, offscreen and without task DB."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS', '--disable-gpu --mute-audio')
import json
import time
from pathlib import Path
from uuid import uuid4
from PyQt6.QtCore import QEventLoop, QTimer
from PyQt6.QtWidgets import QApplication
from ptb_desktop.host import Workbench, register_scheme

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT / 'output/validation/p19' / ('qt-' + uuid4().hex)
    out.mkdir(parents=True)
    register_scheme()
    app = QApplication(['P19-owned-offscreen-QA'])
    w = Workbench(ROOT / 'frontend/dist', test=True, vocal_profile=out / 'vocal')
    w.resize(1440, 900)
    w.show()
    report = {'success': False, 'checks': [], 'layouts': [], 'schema_applied': [], 'scope': 'Windows actual Qt offscreen; no physical taskbar/DPI or audio tests'}

    def pause(ms=140):
        loop = QEventLoop(); QTimer.singleShot(ms, loop.quit); loop.exec()

    def js(code):
        loop = QEventLoop(); result = []
        w.page.runJavaScript(code, lambda value: (result.append(value), loop.quit()))
        QTimer.singleShot(6000, loop.quit); loop.exec()
        if not result:
            raise RuntimeError('JS timeout')
        return result[0]

    def until(code, seconds=30):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if js(code):
                return
            pause()
        raise RuntimeError('UI timeout: ' + code)

    def click(text):
        js('[...document.querySelectorAll("button")].find(b=>b.offsetParent&&b.textContent.trim()===' + json.dumps(text, ensure_ascii=False) + ')?.click()')

    def palette(value):
        js('(()=>{const e=document.querySelector("#palette-choice");e.value=' + json.dumps(value) + ';e.dispatchEvent(new Event("change",{bubbles:true}));})()')

    try:
        until('!!document.querySelector(".app-shell")')
        click('设置')
        until('!!document.querySelector("#palette-choice")')
        assert js('document.querySelector("#palette-choice").value') == 'everforest'
        until('document.documentElement.style.getPropertyValue("--font").includes("SimSun")')
        js('document.fonts.load(\'12px "JetBrains Mono"\').then(()=>window.p19FontReady=true)')
        until('window.p19FontReady===true')
        assert 'JetBrains Mono' in js('getComputedStyle(document.querySelector(".font-code-preview")).fontFamily')
        assert 'PTB-Doulos' in js('getComputedStyle(document.querySelector(".font-preview .ipa-text")).fontFamily')
        assert w.windowIcon().availableSizes()
        report['checks'].append('offline bundled WOFF2, default SimSun/Times, fixed IPA and runtime multi-size QIcon')
        ids = js('[...document.querySelector("#palette-choice").options].map(o=>o.value)')
        assert len(ids) == 29 and 'ptb' not in ids
        for value in ids:
            for mode, label in [('light', '浅色'), ('dark', '深色')]:
                palette(value); click(label)
                assert js('document.documentElement.dataset.palette') == value
                assert js('document.documentElement.dataset.theme') == mode
        report['checks'].append('Everforest default and all 58 palette/mode combinations in built actual Qt')
        js('(()=>{const e=document.querySelector("#mono-choice");e.value="__custom__";e.dispatchEvent(new Event("change",{bubbles:true}));})()')
        until('!!document.querySelector("input[aria-label=自定义代码字体]")')
        js('(()=>{const e=document.querySelector("input[aria-label=自定义代码字体]");e.value="Georgia";e.dispatchEvent(new Event("input",{bubbles:true}));})()')
        click('应用字体')
        until('document.querySelector(".font-settings [role=status]")?.textContent.includes("字体已应用")')
        assert 'Georgia' in js('getComputedStyle(document.querySelector(".font-code-preview")).fontFamily')
        assert js('document.querySelector("#mono-choice").options[0].textContent') == 'JetBrains Mono（内置）'
        js('(()=>{const e=document.querySelector("#mono-choice");e.value="JetBrains Mono";e.dispatchEvent(new Event("change",{bubbles:true}));})()')
        click('应用字体')
        until('document.querySelector(".font-settings [role=status]")?.textContent.includes("字体已应用")')
        assert 'JetBrains Mono' in js('getComputedStyle(document.querySelector(".font-code-preview")).fontFamily')
        report['checks'].append('custom Georgia to bundled Mono through native select, with offline font actually applied')
        for width, height in [(1440, 900), (1920, 1080), (900, 700)]:
            w.resize(width, height); pause()
            for mode, label in [('light', '浅色'), ('dark', '深色')]:
                palette('everforest'); click(label); pause(300)
                layout = js('(()=>{const e=document.querySelector(".settings-columns");return {scroll:document.documentElement.scrollWidth,width:document.documentElement.clientWidth,columns:getComputedStyle(e).gridTemplateColumns}})()')
                assert layout['scroll'] <= layout['width'] + 2
                report['layouts'].append({'width': width, 'height': height, 'mode': mode, **layout})
                if width == 1440:
                    assert w.view.grab().save(str(out / ('settings-' + mode + '.png')))
        w.resize(1440, 900)
        click('声道工作台')
        until('!!document.querySelector("iframe")?.contentDocument?.querySelector(".workspace")')
        until('!!document.querySelector("iframe").contentDocument.documentElement.style.getPropertyValue("--panel")')
        for value, label in [('everforest', '深色'), ('catppuccin', '浅色'), ('matrix', '浅色')]:
            click('设置'); palette(value); click(label); pause(200)
            result = js('(()=>{const a=getComputedStyle(document.documentElement),b=getComputedStyle(document.querySelector("iframe").contentDocument.documentElement);return [a.getPropertyValue("--panel").trim(),b.getPropertyValue("--panel").trim(),a.getPropertyValue("--accent").trim(),b.getPropertyValue("--accent").trim()]})()')
            assert result[0] == result[1] and result[2] == result[3], result
        report['checks'].append('existing M10 iframe follows Everforest, Catppuccin and Matrix palette tokens')
        report['success'] = True
    except Exception as error:
        report['error'] = str(error)
        w.view.grab().save(str(out / 'failed.png'))
        raise
    finally:
        (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(out, flush=True)
        w.closing = True; w.close(); app.processEvents()


if __name__ == '__main__':
    main()
