"""Actual Windows Qt native keyboard QA with an off-record profile and owned scratch."""
from pathlib import Path
import json
import os
import time
import traceback
import zipfile
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from verification_artifacts import verification_run
from workbench_source import bind_sources

ROOT = Path(__file__).resolve().parents[1]
bind_sources(ROOT)
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS', '--mute-audio')
from PyQt6.QtCore import QEventLoop, QTimer, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QFileDialog
from ptb_desktop.host import Workbench, register_scheme


def run(out, scratch):
    register_scheme()
    app = QApplication(['M15-keyboard-analysis-owned-QA'])
    w = Workbench(ROOT / 'frontend/dist', test=True, jobs_path=None, vocal_profile=scratch / 'vocal')
    w.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    w.resize(1440, 900)
    w.show()
    report = {'success': False, 'checks': [], 'downloads': [], 'physicalTimingMeasured': False}

    def destination(parent, title, suggested, file_filter, **kwargs):
        file = scratch / (str(len(report['downloads'])) + '-' + Path(suggested).name)
        report['downloads'].append(str(file))
        return str(file), file_filter
    QFileDialog.getSaveFileName = destination

    def pause(ms=60):
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
        return '[...document.querySelectorAll("button")].find(e=>e.textContent.trim()===' + json.dumps(name) + ')'

    def click(name):
        assert js('(()=>{const e=' + button(name) + ';if(!e||e.disabled)return false;e.click();return true;})()'), name
        pause()

    def set_value(label, value, event='input'):
        code = '(()=>{const e=document.querySelector(' + json.dumps('[aria-label="'+label+'"]') + ');if(!e)return false;e.value=' + json.dumps(value) + ';e.dispatchEvent(new Event(' + json.dumps(event) + ',{bubbles:true}));return true;})()'
        assert js(code), label

    def toggle(text, wanted=True):
        assert js('(()=>{const l=[...document.querySelectorAll("label")].find(e=>e.textContent.trim()===' + json.dumps(text) + '),e=l?.querySelector("input");if(!e||e.disabled)return false;if(e.checked!==' + str(wanted).lower() + ')e.click();return true;})()'), text

    def phase(value):
        until('document.querySelector(".m15-run")?.dataset.phase===' + json.dumps(value))

    def idle():
        until('document.querySelector(".perception-page")?.getAttribute("aria-busy")==="false"')

    def key(value):
        QTest.keyClick(w.view.focusProxy(), value)
        pause()

    try:
        until('document.querySelector(".host-badge")?.textContent==="本地桌面"')
        click('感知实验')
        idle()
        js('(()=>{const dt=new DataTransfer();for(const i of [1,2])dt.items.add(new File(["Qt 刺激 "+i],"刺激"+i+".txt",{type:"text/plain"}));const e=document.querySelector("[aria-label=\\"导入刺激 X\\"]");e.files=dt.files;e.dispatchEvent(new Event("change",{bubbles:true}));})()')
        until('document.querySelectorAll(".m15-asset").length===2')
        idle()
        click('参数')
        set_value('推进方式', 'manual', 'change')
        toggle('自动播放', False)
        toggle('提示音', False)
        set_value('作答页提示语', '请按 F 或 J 判断。')
        toggle('显示作答页提示语')
        click('序列')
        set_value('试次 1 标准答案', 'j', 'change')
        set_value('试次 2 标准答案', 'j', 'change')
        click('预检与试音')
        idle()
        toggle('已听见试音，设备正确')
        toggle('已暂停其他录制、播放及重型任务')
        click('填写被试信息')
        until('!!document.querySelector(".m15-questionnaire input")')
        # Subject input uses actual native Qt keyboard events from here.
        QTest.keyClicks(w.view.focusProxy(), 'Qt-P01')
        key(Qt.Key.Key_Tab)
        key(Qt.Key.Key_Down)
        key(Qt.Key.Key_Space)
        key(Qt.Key.Key_Tab)
        key(Qt.Key.Key_Tab)
        key(Qt.Key.Key_Return)
        phase('intro')
        key(Qt.Key.Key_Return)
        phase('interstitial')
        for index in range(2):
            key(Qt.Key.Key_Return)
            phase('ready')
            key(Qt.Key.Key_Space)
            phase('responding')
            assert js('document.querySelector(".m15-response-prompt")?.textContent==="请按 F 或 J 判断。"')
            if index == 0:
                w.view.grab().save(str(out / 'qt-response.png'))
            key(Qt.Key.Key_J if index == 0 else Qt.Key.Key_F)
            phase('completed' if index == 1 else 'interstitial')
        idle()
        set_value('结果格式', 'json', 'change')
        click('导出结果')
        idle()
        until('true')
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and not (report['downloads'] and Path(report['downloads'][-1]).is_file()):
            pause()
        session = json.loads(Path(report['downloads'][-1]).read_text('utf8'))
        assert [a['key'] for a in session['attempts']] == ['j', 'f']
        assert session['answers']['q1'] == 'Qt-P01'
        click('确认导出文件已保存')
        idle()
        click('结束实验')
        idle()
        click('分析')
        idle()
        assert js('document.querySelectorAll(".analysis-table tbody tr").length===1&&document.querySelector(".analysis-table").textContent.includes("50.00%")')
        for theme in ('light', 'dark'):
            js('document.documentElement.dataset.theme=' + json.dumps(theme))
            pause()
            w.view.grab().save(str(out / ('qt-analysis-' + theme + '.png')))
        click('导出被试分析')
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and not (report['downloads'] and Path(report['downloads'][-1]).suffix == '.xlsx' and Path(report['downloads'][-1]).is_file()):
            pause()
        with zipfile.ZipFile(report['downloads'][-1]) as book:
            names = book.read('xl/workbook.xml').decode()
            assert 'Participants' in names and 'Scoring' in names
        report['checks'] = ['native Qt keyboard questionnaire/start/enter/manual play/answers', 'custom prompt in response page', 'two durable responses and 50 percent accuracy', 'real JSON/XLSX download readback; light/dark analysis']
        report['success'] = True
    except Exception as exc:
        report['error'] = repr(exc)
        report['traceback'] = traceback.format_exc()
        report['body'] = js('document.body.innerText')
        w.view.grab().save(str(out / 'failure.png'))
        raise
    finally:
        (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
        w.closing = True
        w.close()
        pause(300)
        print(out, flush=True)

if __name__ == '__main__':
    import subprocess
    import sys
    if '--worker' in sys.argv:
        run(Path(os.environ['M15_QA_OUT']), Path(os.environ['M15_QA_SCRATCH']))
    else:
        with verification_run('m15-keyboard-analysis', 'qt', 'Current-source Qt off-record profile; two synthetic UTF-8 TXT; native QTest keyboard, prompt and participant analysis') as (out, scratch):
            env = dict(os.environ, M15_QA_OUT=str(out), M15_QA_SCRATCH=str(scratch))
            result = subprocess.run([sys.executable, '-B', '-X', 'utf8', str(Path(__file__).resolve()), '--worker'], env=env, capture_output=True, text=True, encoding='utf8', errors='replace')
            (out / 'console.log').write_text(result.stdout + result.stderr, encoding='utf8')
            print(out, flush=True)
            if result.returncode:
                print(result.stderr, flush=True)
                raise RuntimeError('Qt QA failed; see report.json')
