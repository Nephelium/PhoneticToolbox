"""M15-R1 hidden Windows Qt, native end/Enter/retry and synthetic local sessions."""
from __future__ import annotations
import json
import os
import sys
import time
import traceback
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
for folder in ('packages/phonetic_core/src', 'backend/src', 'desktop/src'):
    sys.path.insert(0, str(ROOT / folder))
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS', '--mute-audio')
from PyQt6.QtCore import QEventLoop, QTimer, Qt, QPoint
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QFileDialog
from ptb_desktop.host import Workbench, register_scheme


def main():
    out = ROOT / 'output/validation/m15-r1' / ('qt-' + uuid4().hex)
    out.mkdir(parents=True)
    register_scheme()
    app = QApplication(['M15-R1-owned-QA'])
    w = Workbench(ROOT / 'frontend/dist', test=True, jobs_path=None)
    w.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    w.resize(1440, 900)
    w.show()
    report = {'success': False, 'checks': [], 'layouts': [], 'downloads': [],
              'scope': 'actual Windows hidden Qt; synthetic input; no task DB or physical timing',
              'heard_checkbox': 'automated; human hearing not verified'}

    def destination(parent, title, suggested, file_filter, **kwargs):
        target = out / (str(len(report['downloads']) + 1) + '-' + Path(suggested).name)
        report['downloads'].append(str(target))
        return str(target), file_filter
    QFileDialog.getSaveFileName = destination

    def pause(ms=100):
        loop = QEventLoop()
        QTimer.singleShot(ms, loop.quit)
        loop.exec()

    def js(code):
        loop, box = QEventLoop(), []
        w.page.runJavaScript(code, lambda value: (box.append(value), loop.quit()))
        QTimer.singleShot(6000, loop.quit)
        loop.exec()
        assert box, 'JavaScript timeout'
        return box[0]

    def until(code, seconds=20):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if js(code):
                return
            pause()
        raise AssertionError(code)

    def button(name):
        return '[...document.querySelectorAll("button")].find(e=>e.offsetParent&&e.textContent.trim()===' + json.dumps(name) + ')'

    def pointer(name):
        selector = button(name)
        assert js('(()=>{const e=' + selector + ';if(!e||e.disabled)return false;e.scrollIntoView({block:"nearest"});return true;})()'), name
        pause()
        point = js('(()=>{const r=' + selector + '.getBoundingClientRect();return [r.x+r.width/2,r.y+r.height/2];})()')
        target = w.view.focusProxy() or w.view
        p = target.mapFromGlobal(w.view.mapToGlobal(QPoint(round(point[0]), round(point[1]))))
        QTest.mouseMove(target, p)
        QTest.mouseClick(target, Qt.MouseButton.LeftButton, pos=p)
        pause()

    def phase(name):
        until('document.querySelector(".m15-run")?.dataset.phase===' + json.dumps(name))

    def idle():
        until('document.querySelector(".perception-page")?.getAttribute("aria-busy")==="false"')

    def initial():
        until('!!document.querySelector("[aria-label=实验范式]")&&!document.querySelector(".m15-run")')
        idle()

    def input_value(label, value):
        assert js('(()=>{const e=document.querySelector('+json.dumps('[aria-label="'+label+'"]')+');if(!e)return false;e.value='+json.dumps(value)+';e.dispatchEvent(new Event("input",{bubbles:true}));return true;})()')

    def prepare():
        pointer('预检与试音')
        idle()
        assert js('document.body.textContent.includes("预检通过")')
        assert js('(()=>{for(const text of ["已听见试音，设备正确","已暂停其他录制、播放及重型任务"]){const e=[...document.querySelectorAll("label")].find(l=>l.textContent.trim()===text)?.querySelector("input");if(!e||e.disabled)return false;if(!e.checked)e.click();}return true;})()')

    def run(name):
        pointer('填写被试信息')
        until(r'!!document.querySelector("[aria-label=\"姓名/编号\"]")')
        assert js(r'document.querySelector("[aria-label=\"姓名/编号\"]").value===""')
        assert js('!document.querySelector("input[type=radio][value=女]").checked')
        input_value('姓名/编号', name)
        js('document.querySelector("input[type=radio][value=女]").click()')
        pointer('建立独立会话')
        phase('intro')
        pointer('开始实验')
        phase('responding')

    def export_json(name):
        idle()
        js('(()=>{const e=document.querySelector("[aria-label=结果格式]");e.value="json";e.dispatchEvent(new Event("change",{bubbles:true}));})()')
        count = len(report['downloads'])
        pointer('导出结果')
        idle()
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if len(report['downloads']) > count:
                target = Path(report['downloads'][-1])
                if target.exists():
                    try:
                        value = json.loads(target.read_text(encoding='utf8'))
                        (out / (name + '-readback.json')).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')
                        return value
                    except (ValueError, OSError):
                        pass
            pause()
        raise AssertionError('native JSON save/readback')

    def layouts(stage):
        for theme in ['light', 'dark']:
            for width, height in [(1440, 900), (900, 700)]:
                for size in [14, 24]:
                    w.resize(width, height)
                    js('document.documentElement.dataset.theme='+json.dumps(theme)+';document.documentElement.style.fontSize='+json.dumps(str(size)+'px')+';document.documentElement.style.setProperty("--body-size",'+json.dumps(str(size)+'px')+')')
                    pause(220)
                    geometry = js('(()=>{const es=[...document.querySelectorAll(".perception-page button")].filter(e=>e.offsetParent&&e.textContent.trim()==="导出结果");const r=es[0]?.getBoundingClientRect();return {count:es.length,x:r?.x,right:r?.right,width:innerWidth};})()')
                    assert geometry['count'] == 1 and geometry['x'] >= 0 and geometry['right'] <= geometry['width'] + 1, geometry
                    filename = f'{stage}-{theme}-{width}-{size}.png'
                    w.view.grab().save(str(out / filename))
                    report['layouts'].append({'file': filename, **geometry})
        w.resize(1440, 900)
        js('document.documentElement.style.fontSize="14px";document.documentElement.style.setProperty("--body-size","14px")')
        pause()

    try:
        until('document.querySelector(".host-badge")?.textContent==="本地桌面"')
        pointer('感知实验')
        idle()
        js(r'(()=>{const dt=new DataTransfer();for(const i of [1,2])dt.items.add(new File(["Qt 刺激 "+i+" a̠ ŋ"],"刺激"+i+".txt",{type:"text/plain"}));const e=document.querySelector("[aria-label=\"导入刺激 X\"]");e.files=dt.files;e.dispatchEvent(new Event("change",{bubbles:true}));})()')
        until('document.querySelectorAll(".m15-asset").length===2')
        idle()
        pointer('参数')
        input_value('试次间隔毫秒', '0')
        js('[...document.querySelectorAll("label")].find(e=>e.textContent.trim()==="提示音").querySelector("input").click()')
        prepare()
        run('Qt 被试甲')
        target = w.view.focusProxy() or w.view
        QTest.keyClick(target, Qt.Key.Key_J)
        phase('responding')
        pause(150)
        QTest.keyClick(target, Qt.Key.Key_F)
        phase('completed')
        idle()
        natural = export_json('natural')
        assert natural['status'] == 'completed' and len(natural['attempts']) == 2
        pointer('确认导出文件已保存')
        idle()
        layouts('completed')
        pointer('结束实验')
        initial()
        assert js('document.querySelectorAll(".m15-asset").length===2')
        assert js(button('填写被试信息') + '.disabled')
        count = len(report['downloads'])
        pointer('查看结果')
        phase('completed')
        idle()
        pause(250)
        assert len(report['downloads']) == count
        assert js(button('确认导出文件已保存') + '.disabled')
        pointer('结束实验')
        initial()
        report['checks'].append('native natural completion, JSON readback, end returns initial, immediate recovery has no duplicate download')
        prepare()
        run('Qt 被试乙')
        js(button('结束实验') + '.focus()')
        QTest.keyClick(target, Qt.Key.Key_Return)
        initial()
        ended = export_json('early-end')
        assert ended['status'] == 'ended' and ended['attempts'][0]['status'] == 'interrupted'
        assert ended['answers']['q1'] == 'Qt 被试乙' and ended['participantId'] != natural['participantId']
        report['checks'].append('native Enter ends focused session, clears previous participant form, and retains interrupted result')
        prepare()
        run('Qt 被试丙')
        js('window.__m15Put=IDBObjectStore.prototype.put;IDBObjectStore.prototype.put=function(...args){if(this.name==="sessions")throw new DOMException("M15-R1 isolated quota injection","QuotaExceededError");return window.__m15Put.apply(this,args);}')
        pointer('结束实验')
        phase('saving-error')
        idle()
        assert js(button('结束实验')+'.disabled&&'+button('返回设计器')+'.disabled')
        memory = export_json('failed-save-memory')
        assert memory['status'] == 'ended' and memory['attempts'][0]['status'] == 'interrupted'
        w.view.grab().save(str(out / 'save-error.png'))
        js('IDBObjectStore.prototype.put=window.__m15Put')
        pointer('重试本地保存')
        phase('completed')
        idle()
        pointer('结束实验')
        initial()
        durable = export_json('retry-durable')
        assert durable['attempts'][0]['id'] == memory['attempts'][0]['id']
        report['checks'].append('native failed-save exit protection, memory JSON, retry and durable JSON preserve the same attempt')
        layouts('initial')
        report['success'] = True
    except Exception as exc:
        report['error'] = repr(exc)
        report['traceback'] = traceback.format_exc()
        report['body'] = js('document.body.innerText')
        w.view.grab().save(str(out / 'failure.png'))
    finally:
        (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
        w.closing = True
        w.close()
        pause(300)
    print(out / 'report.json')
    if not report['success']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
