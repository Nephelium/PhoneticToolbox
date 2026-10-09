"""Owned actual Qt host: settings, native font list and synthetic EGG display."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'windows')
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS', '--disable-gpu --mute-audio')
import json
import sqlite3
import time
from pathlib import Path
from uuid import uuid4
import numpy as np
from scipy.io import wavfile
from PyQt6.QtCore import QEventLoop, QTimer, Qt, QPoint
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QFileDialog
from ptb_desktop.host import Workbench, register_scheme
from ptb_worker.local_acoustic_files import initialize_local_files
ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT / 'output/validation/p19-r5' / ('qt-' + uuid4().hex)
    out.mkdir(parents=True)
    db = out / 'jobs.sqlite3'
    with sqlite3.connect((ROOT / 'output/validation/p06/local-state.sqlite3').as_uri() + '?mode=ro', uri=True) as source, sqlite3.connect(db) as target:
        assert not source.execute("SELECT 1 FROM jobs WHERE state IN ('queued','running','cancel_requested')").fetchone()
        source.backup(target)
    cache = out / 'cache'; cache.mkdir(); initialize_local_files(cache)
    inputs = out / 'inputs'; inputs.mkdir()
    t = np.arange(44100 * 6) / 44100
    gain = np.where(t < 2, .8, np.where(t < 4, .04, 0.))
    audio = gain * (np.sin(2 * np.pi * 200 * t) + .2 * np.sin(2 * np.pi * 400 * t))
    wavfile.write(inputs / 'db-ranges.wav', 44100, np.column_stack([.5 * np.sin(2 * np.pi * 200 * t), audio]).astype(np.float32))
    target_time = np.arange(44100 * 2) / 44100
    wavfile.write(inputs / 'target.wav', 44100, (.4 * np.sin(2 * np.pi * 800 * target_time)).astype(np.float32))
    os.environ['PTB_EGG_PYTHON'] = str(ROOT / '.venv/m03-compatible/python.exe')
    register_scheme(); app = QApplication(['P19-R5-M03-R6-owned-QA'])
    w = Workbench(ROOT / 'frontend/dist', test=True, jobs_path=db, local_files_root=cache, vocal_profile=out / 'vocal')
    w.resize(1440, 1000); w.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen); w.show()
    QFileDialog.getExistingDirectory = lambda *a, **k: str(inputs)
    report = {'success': False, 'checks': [], 'layouts': [], 'schema_applied': [], 'scope': 'Windows actual Qt windows platform, hidden test window, synthetic EGG; no hardware or physical DPI'}

    def pause(ms=120):
        loop = QEventLoop(); QTimer.singleShot(ms, loop.quit); loop.exec()

    def js(code):
        loop = QEventLoop(); box = []; w.page.runJavaScript(code, lambda value: (box.append(value), loop.quit()))
        QTimer.singleShot(6000, loop.quit); loop.exec()
        if not box: raise RuntimeError('JS timeout')
        return box[0]

    def until(code, seconds=60):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if js(code): return
            pause()
        raise RuntimeError('UI timeout: ' + code)

    def click(text):
        assert js('(()=>{const e=[...document.querySelectorAll("button")].find(e=>e.offsetParent&&e.textContent.trim()===' + json.dumps(text, ensure_ascii=False) + ');if(!e)return false;e.click();return true;})()'), text
        pause()

    def palette(value):
        js('document.querySelector("#palette-choice").click()'); pause()
        js('document.querySelector(' + json.dumps('[data-palette-option="' + value + '"]') + ').click()'); pause()

    def idle():
        pause()
        until('document.querySelector(".egg-live-status")?.textContent==="实时预览"&&[...document.querySelectorAll(".egg-page button")].some(e=>e.textContent==="保存 CSV / 三图"&&!e.disabled)')

    def roi(value):
        js('(()=>{const e=document.querySelectorAll(".egg-bottom-bar .selection-controls input"),a=' + str(value) + ',b=' + str(value + .5) + ';for(const [i,v]of (a<Number(e[0].value)?[[0,a],[1,b]]:[[1,b],[0,a]])){e[i].value=String(v);e[i].dispatchEvent(new Event("input",{bubbles:true}));e[i].dispatchEvent(new Event("change",{bubbles:true}));}})()'); idle()

    db_range = '([...document.querySelectorAll("input[aria-label=\\\"EGG dB 下限\\\"],input[aria-label=\\\"EGG dB 上限\\\"]")].map(e=>Number(e.value)))'
    try:
        until('!!document.querySelector(".app-shell")'); click('设置')
        until('document.documentElement.style.getPropertyValue("--font").includes("SimSun")')
        assert js('document.documentElement.dataset.palette') == 'codex'
        assert js('document.querySelector("#waveform-color-choice").value') == 'theme'
        click('读取本机字体列表')
        until('document.querySelector(".font-settings [role=status]")?.textContent.includes("已读取")')
        count = js('document.querySelector("select[aria-label=英文与数字字体]").options.length'); assert count > 100
        assert js('document.querySelector("select[aria-label=英文与数字字体]").value') == 'Times New Roman'
        assert js('(()=>{const e=document.querySelector("select[aria-label=英文与数字字体]");if(![...e.options].some(o=>o.value==="Georgia"))return false;e.value="Georgia";e.dispatchEvent(new Event("change",{bubbles:true}));return true;})()')
        click('应用字体'); until('document.querySelector(".font-settings [role=status]")?.textContent.includes("字体已应用")')
        assert js('document.querySelector("#ptb-font-faces").textContent.includes(\'local("Georgia")\')')
        report['checks'].append({'native_font_count': count, 'selection': 'filled Times -> full actual list -> Georgia applied'})
        js('document.querySelector(".font-follow input").click()'); pause()
        assert js('document.querySelector("select[aria-label=图表中文字体]").value') == 'SimSun'
        assert js('document.querySelector("select[aria-label=图表英文字体]").value') == 'Times New Roman'
        report['checks'].append('independent figure defaults populated, fixed IPA retained')
        for width, height, scale in [(1440, 1000, 100), (900, 700, 150)]:
            w.resize(width, height); pause()
            js('document.documentElement.style.zoom=' + json.dumps(str(scale / 100)) + ';document.documentElement.style.setProperty("--page-scale",' + json.dumps(str(scale / 100)) + ')')
            for mode, label in [('light', '浅色'), ('dark', '深色')]:
                click(label); js('document.querySelector("#palette-choice").scrollIntoView();document.querySelector("#palette-choice").click()'); pause()
                js('(()=>{const e=document.querySelector("[data-palette-option=catppuccin]");e.focus();e.dispatchEvent(new PointerEvent("pointerenter"));})()'); pause()
                until('!!document.querySelector(".palette-hover-preview")')
                geometry = js('([...document.querySelectorAll(".palette-list,.palette-hover-preview")].map(e=>{const r=e.getBoundingClientRect();return {x:r.x,y:r.y,right:r.right,bottom:r.bottom};}))')
                assert all(g['x'] >= -1 and g['y'] >= -1 and g['right'] <= width + 1 and g['bottom'] <= height + 1 for g in geometry), geometry
                assert js('document.querySelector(".palette-hover-preview").dataset.previewMode') == mode
                w.view.repaint(); pause(400)
                w.view.grab().save(str(out / f'preview-{width}-{mode}.png'))
                report['layouts'].append({'width': width, 'height': height, 'scale': scale, 'mode': mode, 'geometry': geometry})
                js('document.dispatchEvent(new KeyboardEvent("keydown",{key:"Escape",bubbles:true}))'); pause()
        w.resize(1440, 1000); js('document.documentElement.style.zoom="1";document.documentElement.style.setProperty("--page-scale","1")'); palette('codex')
        click('EGG 信号分析'); until('!!document.querySelector(".egg-page")'); click('打开 WAV 目录')
        js('(()=>{const e=document.querySelector(".egg-source select");e.value=[...e.options].find(o=>o.textContent==="db-ranges.wav").value;e.dispatchEvent(new Event("change",{bubbles:true}));})()'); idle()
        loud = js(db_range); roi(2.5); quiet = js(db_range)
        assert quiet[1] <= loud[1] - 20 and quiet[1] - quiet[0] == 50, (loud, quiet)
        auto_selector = '[...document.querySelectorAll(".egg-page button")].find(e=>e.textContent==="自动 dB")'
        assert js('(' + auto_selector + ').getAttribute("aria-pressed")') == 'true'
        w.view.grab().save(str(out / 'egg-auto.png'))
        click('自动 dB'); assert js('(' + auto_selector + ').getAttribute("aria-pressed")') == 'false'
        js('(()=>{for(const [label,v]of [["EGG dB 下限",-90],["EGG dB 上限",-20]]){const e=document.querySelector("input[aria-label=\\\""+label+"\\\"]");e.value=String(v);e.dispatchEvent(new Event("input",{bubbles:true}));}})()'); idle(); roi(.7)
        assert js(db_range) == [-90, -20]
        click('自动 dB'); idle(); assert js(db_range) != [-90, -20]
        before = js(db_range); roi(4.5); assert js(db_range) == before
        assert js('document.querySelector(".db-mode-note").textContent.includes("静音")')
        report['checks'].append({'actual_egg': 'strong/quiet/manual/re-enabled/silent', 'loud': loud, 'quiet': quiet})
        roi(2.3456789)
        assert js('[...document.querySelectorAll(".egg-bottom-bar .selection-controls input")].map(e=>e.value)') == ['2.34568', '2.84568']
        micro_selector = json.dumps('input[aria-label="EGG 微观窗口"]', ensure_ascii=False)
        js('(()=>{const e=document.querySelector(' + micro_selector + ');e.value="55.678901";e.dispatchEvent(new Event("input",{bubbles:true}));e.dispatchEvent(new Event("change",{bubbles:true}));})()'); idle()
        assert js('document.querySelector(' + micro_selector + ').value') == '55.68'
        report['checks'].append({'time_fields': 'actual Qt seconds five decimals / milliseconds two, user input and scientific preview complete'})
        click('发声类型合成'); until('!!document.querySelector(".m07-page")'); click('打开音频目录')
        def choose_audio(label, name):
            assert js('(()=>{const e=document.querySelector(' + json.dumps('select[aria-label="' + label + '"]') + ');const o=[...e.options].find(o=>o.textContent===' + json.dumps(name) + ');if(!o)return false;e.value=o.value;e.dispatchEvent(new Event("change",{bubbles:true}));return true;})()')
            pause(500)
        choose_audio('源音频', 'db-ranges.wav'); choose_audio('目标音频', 'target.wav')
        until('document.querySelectorAll(".m07-page .input-pair .wave-line").length===2')
        assert js('[...document.querySelectorAll(".m07-page .input-pair .wave-selection")].map(e=>Number(e.getAttribute("width")))') == [0, 0]
        js('window.__audioStarts=[];window.__nativeEvents=[];for(const type of ["pointerdown","pointerup","keydown"]){document.addEventListener(type,e=>window.__nativeEvents.push({type:e.type,target:e.target.tagName,key:e.key,x:e.clientX,y:e.clientY,trusted:e.isTrusted}),true);}window.__sourceFactory=AudioContext.prototype.createBufferSource;AudioContext.prototype.createBufferSource=function(){const n=window.__sourceFactory.call(this),s=n.start;n.start=function(w,o,d){window.__audioStarts.push({offset:o,duration:d,sample:this.buffer.getChannelData(0)[10]});return s.call(this,w,o,d);};return n;}')
        native_target = w.view.focusProxy() or w.view
        def drag_wave(index):
            box = js('(()=>{const e=document.querySelectorAll(".m07-page .input-pair .wave-track>svg")[' + str(index) + '];e.scrollIntoView({block:"nearest"});const r=e.getBoundingClientRect();return {x:r.x,y:r.y,width:r.width,height:r.height};})()')
            a = QPoint(round(box['x'] + box['width'] * .1), round(box['y'] + box['height'] * .5))
            b = QPoint(round(box['x'] + box['width'] * .6), a.y())
            QTest.mousePress(native_target, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, a); pause(60)
            QTest.mouseMove(native_target, b, 50); QTest.mouseRelease(native_target, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, b); pause(100)
        def space():
            QTest.keyClick(native_target, Qt.Key.Key_Space); pause(200)
        drag_wave(0); space()
        report['native_diagnostics'] = js('({events:window.__nativeEvents,starts:window.__audioStarts,focus:document.activeElement?.tagName,selection:[...document.querySelectorAll(".m07-page .wave-selection")].map(e=>e.getAttribute("width")),errors:[...document.querySelectorAll(".error-text")].map(e=>e.textContent)})')
        until('window.__audioStarts.length===1', seconds=8)
        space(); drag_wave(1); space(); until('window.__audioStarts.length===2')
        assert js('[...document.querySelectorAll(".m07-page .input-pair .wave-selection")].map(e=>Number(e.getAttribute("width"))>0)') == [False, True]
        space(); drag_wave(0); space(); until('window.__audioStarts.length===3')
        assert js('[...document.querySelectorAll(".m07-page .input-pair .wave-selection")].map(e=>Number(e.getAttribute("width"))>0)') == [True, False]
        starts = js('window.__audioStarts'); assert starts[0]['sample'] == starts[2]['sample'] and starts[0]['sample'] != starts[1]['sample'], starts
        space(); report['checks'].append({'native_qtest': 'M07 source/target/source pointer selections and Space start/stop, sole selected waveform', 'starts': starts})
        w.view.repaint(); pause(400); w.view.grab().save(str(out / 'm07-native-selection.png'))
        report['success'] = True
    except Exception as error:
        report['error'] = str(error); w.view.grab().save(str(out / 'failed.png')); raise
    finally:
        (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(out, flush=True); w.closing = True; w.close(); app.processEvents()


if __name__ == '__main__': main()
