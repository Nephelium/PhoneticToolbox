"""Selected real Qt client workflows in an existing frozen workbench."""
import base64
import json
from pathlib import Path
import time


def verify(window, output, report):
    from PyQt6.QtCore import QEventLoop, QTimer, Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QFileDialog
    destination = Path(output) / 'client-exports'
    destination.mkdir()
    exports = []
    original_dialog = QFileDialog.getSaveFileName
    def save(parent, title, proposed, filters, **kwargs):
        path = destination / (str(len(exports)) + '-' + Path(proposed).name)
        exports.append(path)
        return str(path), filters
    QFileDialog.getSaveFileName = save
    def pause(ms=100):
        loop = QEventLoop(); QTimer.singleShot(ms, loop.quit); loop.exec()
    def js(code):
        loop = QEventLoop(); result = []
        window.page.runJavaScript(code, lambda value: (result.append(value), loop.quit()))
        QTimer.singleShot(6000, loop.quit); loop.exec()
        assert result, code
        return result[0]
    def until(code, seconds=40):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if js(code): return
            pause()
        raise AssertionError(code + '\n' + str(js('document.body.innerText.slice(-4000)')))
    def title_nav(title):
        assert js('(()=>{const e=[...document.querySelectorAll(".nav-item")].find(e=>e.title==='+json.dumps(title)+');if(!e)return false;e.click();return true})()')
        pause(250)
    def button(label):
        return '[...document.querySelectorAll("button")].find(e=>e.offsetParent&&!e.disabled&&e.textContent.trim()==='+json.dumps(label)+')'
    def click(label):
        until('!!'+button(label)); js(button(label)+'.click()'); pause()
    def fill(label, value):
        assert js('(()=>{const e=document.querySelector('+json.dumps('[aria-label="'+label+'"]')+');if(!e)return false;e.value='+json.dumps(value)+';e.dispatchEvent(new Event("input",{bubbles:true}));e.dispatchEvent(new Event("change",{bubbles:true}));return true})()')
        pause()
    try:
        title_nav('汉字转国际音标')
        until('!!document.querySelector("[aria-label=待转换汉字文本]")')
        fill('待转换汉字文本', '女略')
        until('document.querySelectorAll(".m13-mapped").length===2')
        assert js('[...document.querySelectorAll(".m13-mapped")].every(e=>e.dataset.value.length>0)')
        selector = '.mandarin-ipa-page .module-toolbar-primary>button'
        js('document.querySelector('+json.dumps(selector)+').click()')
        until('document.querySelector(".mandarin-ipa-page").innerText.length>0')
        deadline = time.monotonic() + 30
        while not any(p.exists() and p.suffix == '.png' for p in exports) and time.monotonic() < deadline: pause()
        assert any(p.exists() and p.suffix == '.png' for p in exports)
        report['checks'].append('Frozen M13 dictionary conversion and native PNG export')
        title_nav('感知实验')
        until('!!document.querySelector(".perception-page")')
        js('(()=>{const dt=new DataTransfer();for(const i of [1,2])dt.items.add(new File(["public QA stimulus "+i],"stimulus"+i+".txt",{type:"text/plain"}));const e=document.querySelector('+json.dumps('[aria-label="导入刺激 X"]')+');e.files=dt.files;e.dispatchEvent(new Event("change",{bubbles:true}));})()')
        until('document.querySelectorAll(".m15-asset").length===2')
        click('参数'); fill('试次间隔毫秒', '0')
        js('[...document.querySelectorAll("label")].find(e=>e.textContent.trim()==="提示音").querySelector("input").click()')
        click('预检与试音')
        until('document.body.textContent.includes("预检通过")')
        assert js('(()=>{for(const t of ["已听见试音，设备正确","已暂停其他录制、播放及重型任务"]){const e=[...document.querySelectorAll("label")].find(l=>l.textContent.trim()===t)?.querySelector("input");if(!e||e.disabled)return false;if(!e.checked)e.click();}return true})()')
        click('填写被试信息'); fill('姓名/编号', 'owned-compact-QA')
        js('document.querySelector("input[type=radio][value=女]").click()')
        click('建立独立会话'); click('开始实验')
        until('document.querySelector(".m15-run")?.dataset.phase==="responding"')
        QTest.keyClick(window.view.focusProxy() or window.view, Qt.Key.Key_J)
        pause(200)
        until('document.querySelector(".m15-run")?.dataset.phase==="responding"')
        QTest.keyClick(window.view.focusProxy() or window.view, Qt.Key.Key_F)
        until('document.querySelector(".m15-run")?.dataset.phase==="completed"')
        fill('结果格式', 'json'); click('导出结果')
        deadline = time.monotonic() + 30
        result = None
        while time.monotonic() < deadline:
            for path in exports:
                if path.exists() and path.suffix == '.json':
                    result = json.loads(path.read_text('utf8'))
            if result: break
            pause()
        assert result and result['status'] == 'completed' and len(result['attempts']) == 2
        report['checks'].append('Frozen M15 independent session, native response keys, IndexedDB save and complete JSON readback; textual stimuli, no hearing/timing accuracy claim')
        state = window.vocal.invoke('status')
        # Use the native model's own current pose through the saved keyframe API.
        sequence = window.vocal.invoke('keyframes/load')
        report['vocal_startup'] = dict(state=state, savedFrames=len(sequence['frames']))
        metadata = window.vocal.invoke('meta')
        frames = [dict(params=metadata['presets'][name], preset=name, duration=.1, f0=150) for name in ('a', 'i')]
        prepared = window.vocal.invoke('animation/prepare', dict(frames=frames))
        audio = window.vocal.invoke('animation/audio', dict(id=prepared['id']))
        raw = base64.b64decode(audio['base64'])
        assert audio['samples'] == 9600 and len(raw) == 9600 * 4
        (destination / 'vocal-a-i.f32').write_bytes(raw)
        report['checks'].append('Frozen native vocal-tract model generated 9,600 audio samples and animation frames without a playback device')
        report['client_exports'] = [dict(name=p.name, bytes=p.stat().st_size) for p in exports if p.is_file()]
    finally:
        QFileDialog.getSaveFileName = original_dialog
