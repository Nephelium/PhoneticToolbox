"""Actual Qt shared-shell checks, isolated files and synthetic PortAudio only."""
import json
import os
import time
from pathlib import Path
from uuid import uuid4

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS', '--disable-gpu --mute-audio')
from PyQt6.QtCore import QEventLoop, QTimer
from PyQt6.QtWidgets import QApplication
from ptb_desktop.host import Workbench, register_scheme
from m16_test_backend import Backend

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT / 'output/validation/m16-m17-integration' / ('qt-' + uuid4().hex)
    project = out / 'project'
    project.mkdir(parents=True)
    register_scheme()
    app = QApplication(['M16-M17-owned-QA'])
    window = Workbench(ROOT / 'frontend/dist', test=True, vocal_profile=out / 'vocal', start_module='M16')
    bridge = window.bridge.recording_bridge()
    bridge.service.backend = Backend()
    bridge.choose = lambda purpose: bridge.service.grant(project, purpose)
    window.resize(1500, 1000)
    window.show()
    report = {'success': False, 'scope': 'Windows Qt offscreen, built shared shell, synthetic PortAudio; no physical devices or database', 'checks': []}

    def pause(ms=100):
        loop = QEventLoop()
        QTimer.singleShot(ms, loop.quit)
        loop.exec()

    def js(code):
        loop = QEventLoop()
        values = []
        window.page.runJavaScript(code, lambda value: (values.append(value), loop.quit()))
        QTimer.singleShot(5000, loop.quit)
        loop.exec()
        return values[0] if values else None

    def until(code):
        end = time.monotonic() + 30
        while time.monotonic() < end:
            if js(code):
                return
            pause()
        raise AssertionError(code + '\n' + str(js('document.body.innerText.slice(-3000)')))

    def click(label, selector='button'):
        expr = '[...document.querySelectorAll(' + json.dumps(selector) + ')].find(e=>e.offsetParent && e.textContent.trim()===' + json.dumps(label) + ')'
        until('!!(' + expr + ') && !(' + expr + ').disabled')
        js(expr + '.click()')
        pause()

    try:
        until('!!document.querySelector(".recording-page")')
        click('新建工程')
        until('document.body.innerText.includes("本地录音工程已建立")')
        click('设备与录前检测')
        device = bridge.service.dispatch({'op': 'devices'})[0]['id']
        js('(()=>{const e=document.querySelector("select[aria-label=输入设备]");e.value=' + json.dumps(device) + ';e.dispatchEvent(new Event("change",{bubbles:true}));})()')
        until('document.querySelectorAll("select[aria-label^=物理输入]").length===2')
        assert js('[...document.querySelectorAll("select[aria-label^=物理输入]")].map(e=>e.value)') == ['microphone', 'microphone']
        click('收起', '.recording-page button')
        click('● 开始录音')
        until('!!document.querySelector(".topbar .recording-indicator")')
        assert bridge.service.capture.config['roles'] == ['microphone', 'microphone']
        assert bridge.service.capture.task is None
        report['checks'].append('untouched default captures two audio inputs without a task or implicit EGG role')
        report['navClick']=js('(()=>{const e=[...document.querySelectorAll(".nav-item")].find(e=>e.title==="国际音标 Plus");if(!e)return "missing";e.click();return e.outerHTML})()')
        until('document.querySelector(".m17-body")?.dataset.loaded==="true"')
        assert js('document.querySelector("#tab-M17").getAttribute("aria-selected")') == 'true'
        assert bridge.capturing
        text = '双音频录制中 ɑ̃ 𝼆 '
        js('(()=>{const e=document.querySelector(".m17-editor");e.focus();e.value=' + json.dumps(text) + ';e.setSelectionRange(e.value.length,e.value.length);e.dispatchEvent(new Event("input",{bubbles:true}));e.dispatchEvent(new KeyboardEvent("keydown",{key:" ",code:"Space",bubbles:true}));})()')
        pause(500)
        assert bridge.capturing and bridge.service.player is None
        assert js('document.querySelector(".m17-editor").value') == text
        report['checks'].append('M17 remains usable during recording, global indicator persists, typing Space does not control hidden M16 playback')
        js('document.querySelector(\'button[aria-label="关闭 录音"]\').click()')
        until('document.body.innerText.includes("停止并保存工程后关闭")')
        click('取消关闭')
        assert bridge.capturing
        js('document.querySelector(\'button[aria-label="关闭 录音"]\').click()')
        click('停止并保存工程后关闭')
        until('!document.querySelector("#tab-M16")')
        assert not bridge.capturing
        assert len(bridge.service.project.data['takes']) == 1
        take = bridge.service.project.data['takes'][0]
        assert take['config']['roles'] == ['microphone', 'microphone'] and take['task_snapshot'] is None
        assert take['versions'][0]['spans']
        assert js('document.querySelector(".m17-editor").value') == text
        report['checks'].append('closing hidden recorder can be cancelled; stop-and-save retains raw stereo and leaves IPA text intact')
        click('录音', '.nav-item')
        until('document.body.innerText.includes("录音历史（1 take）")')
        assert bridge.service.project.data['takes'][0]['id'] == take['id']
        report['checks'].append('reopening recorder recovers the same project and take in the shared desktop session')
        report['success'] = True
    finally:
        (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), 'utf8')
        js('window.onbeforeunload=null')
        window.closing = True
        window.close()
        pause(300)
        app.quit()
        print(json.dumps({'output': str(out), 'success': report['success'], 'checks': len(report['checks'])}, ensure_ascii=False))


if __name__ == '__main__':
    main()
