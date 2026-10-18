"""Actual isolated Qt/QWebChannel/native TextGrid writes, synthetic audio only."""
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sqlite3
import time
import wave

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS', '--disable-gpu --mute-audio')
from PyQt6.QtCore import QEventLoop, QPoint, Qt, QTimer
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QFileDialog
from verification_artifacts import verification_run
from workbench_source import configure

ROOT = Path(__file__).resolve().parents[1]


def main():
    configure()
    from ptb_desktop.host import Workbench, register_scheme
    from ptb_worker.local_acoustic_files import initialize_local_files
    from phonetic_core.annotation import parse_document
    register_scheme()
    app = QApplication(['M12 history QA'])
    with verification_run('m12', 'history-qt', 'Two synthetic 2s mono PCM16 16kHz WAVs, two same-domain interval tiers; actual Qt native writes and Delete key; no physical devices') as (out, scratch):
        print(out, flush=True)
        inputs = scratch / 'inputs'
        inputs.mkdir()
        text = '"ooTextFile short" "TextGrid" 0 2 <exists> 2 ' + ' '.join(
            f'"IntervalTier" "{name}" 0 2 2 0 1 "a" 1 2 "b"' for name in ('words', 'phones'))
        for name in ('A', 'B'):
            with wave.open(str(inputs / (name + '.wav')), 'wb') as stream:
                stream.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
                stream.writeframes(b''.join(struct.pack('<h', round(6000 * math.sin(i * 2 * math.pi * 200 / 16000))) for i in range(32000)))
            (inputs / (name + '.TextGrid')).write_text(text, encoding='utf-8')
        originals = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.iterdir()}
        cache = scratch / 'cache'
        cache.mkdir()
        initialize_local_files(cache)
        database = scratch / 'jobs.sqlite3'
        with sqlite3.connect(database) as connection:
            for migration in ('002_jobs_sqlite.sql', '005_acoustic_batches_sqlite.sql'):
                connection.executescript((ROOT / 'backend/migrations' / migration).read_text('utf-8'))
        window = Workbench(ROOT / 'frontend/dist', test=True, jobs_path=database, local_files_root=cache, vocal_profile=scratch / 'vocal', start_module='M12')
        window.resize(1600, 1100)
        window.show()
        picker = QFileDialog.getExistingDirectory
        QFileDialog.getExistingDirectory = lambda *args, **kwargs: str(inputs)
        report = {'success': False, 'scope': 'Windows actual isolated Qt/QWebChannel/native files; synthetic audio', 'checks': []}
        text_input = 'document.querySelector(' + json.dumps('[aria-label="编辑选中标注文本"]') + ')'
        history_input = 'document.querySelector(' + json.dumps('[aria-label="标注历史版本"]') + ')'

        def pause(ms=80):
            loop = QEventLoop()
            QTimer.singleShot(ms, loop.quit)
            loop.exec()

        def js(code):
            loop, values = QEventLoop(), []
            window.page.runJavaScript(code, lambda value: (values.append(value), loop.quit()))
            QTimer.singleShot(5000, loop.quit)
            loop.exec()
            if not values:
                raise AssertionError('JavaScript timeout: ' + code)
            return values[0]

        def until(code):
            deadline = time.monotonic() + 25
            while time.monotonic() < deadline:
                if js(code):
                    return
                pause()
            raise AssertionError(code + '\n' + str(js('document.body.innerText.slice(-5000)')))

        def click(name):
            expression = '[...document.querySelectorAll(".annotation-page button")].find(b=>b.textContent.trim().replace(/\\s*\\*$/,"")===' + json.dumps(name) + ')'
            until(f'!!({expression})&&!({expression}).disabled')
            js(expression + '.click()')
            pause()

        def choose(name):
            js('[...document.querySelectorAll(".annotation-file-list button")].find(b=>b.textContent.trim()===' + json.dumps(name) + ').click()')
            until('document.querySelector(".annotation-page").getAttribute("aria-busy")==="false"&&document.querySelector(".annotation-file-list button.selected")?.textContent.trim()===' + json.dumps(name))
            until('!document.querySelector(".annotation-page [role=status]")?.textContent.includes("正在读取")')
            pause(250)

        def select(fraction):
            js('document.querySelector(".annotation-grid").scrollIntoView({block:"center"})')
            pause()
            coordinates = js('(()=>{const r=document.querySelector(".annotation-grid").getBoundingClientRect();return [r.x+r.width*' + str(fraction) + ',r.y+r.height*.2]})()')
            QTest.mouseClick(window.view.focusProxy() or window.view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(round(coordinates[0]), round(coordinates[1])))
            pause()

        def label(value):
            js(text_input + '.focus()')
            target = window.view.focusProxy() or window.view
            QTest.keyClick(target, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
            QTest.keyClicks(target, value)
            QTest.keyClick(target, Qt.Key.Key_Return)
            pause()
            until(text_input + '.value===' + json.dumps(value))

        try:
            until('!!document.querySelector(".annotation-page")')
            click('打开音频目录')
            until('document.querySelectorAll(".annotation-file-list button").length===2')
            choose('A.wav')
            select(.2)
            label('Qt-current-A')
            choose('B.wav')
            saved = inputs / 'A_自动保存.TextGrid'
            assert 'Qt-current-A' in saved.read_text('utf-8')
            choose('A.wav')
            select(.2)
            assert js(text_input + '.value') == 'Qt-current-A'
            report['checks'].append('switch writes current TextGrid before opening B; returning A restores current edit')
            select(.5)
            QTest.keyClick(window.view.focusProxy() or window.view, Qt.Key.Key_Delete)
            pause()
            click('保存 TextGrid')
            until('document.querySelector(".annotation-page").getAttribute("aria-busy")==="false"&&!document.querySelector("#tab-M12")?.textContent.includes("*")')
            pause(300)
            document = parse_document(saved.read_text('utf-8'))
            assert all(len(t['intervals']) == 1 for t in document['tiers'])
            report['checks'].append('native Delete removes shared word/phone boundary and actual disk file has one interval per tier')
            js('(()=>{const e=' + history_input + ';e.value=[...e.options].find(o=>o.textContent.includes("原始标注")).value;e.dispatchEvent(new Event("change",{bubbles:true}));})()')
            click('恢复此版本')
            until('document.querySelector(".annotation-page").getAttribute("aria-busy")==="false"')
            pause(300)
            assert all(len(t['intervals']) == 2 for t in parse_document(saved.read_text('utf-8'))['tiers'])
            report['checks'].append('history restoration writes a new current native TextGrid while preserving original source')
            for theme in ('light', 'dark'):
                js('document.documentElement.dataset.theme=' + json.dumps(theme))
                pause(250)
                window.view.grab().save(str(out / (theme + '.png')))
            assert all(hashlib.sha256((inputs / name).read_bytes()).hexdigest() == digest for name, digest in originals.items())
            report['checks'].append('all original WAV/TextGrid hashes unchanged')
            report['success'] = True
        except Exception as exc:
            report['error'] = repr(exc)
            report['editor'] = js('(()=>{const p=document.querySelector(".annotation-grid")?.__vueParentComponent;return p?{text:p.props.editor.state.textgrid,selected:p.props.editor.state.selected,dirty:p.props.editor.state.dirty}:null;})()')
            window.view.grab().save(str(out / 'failure.png'))
            raise
        finally:
            (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            QFileDialog.getExistingDirectory = picker
            window.closing = True
            window.close()
            app.processEvents()
        print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
