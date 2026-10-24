"""Owned Windows Qt and real browser-Python M06 review regression checks."""
import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import time
from workbench_source import ROOT, configure
from verification_artifacts import verification_run, write_json


def oracles(path):
    from phonetic_core.synthesis.klatt.api import defaults, export_parameters
    from ptb_worker.m06_child import compute
    def config(sequence, duration):
        c = defaults()
        c.update(sequence=sequence, duration=duration)
        for curve in c['curves'].values():
            curve['points'][-1][0] = duration
        return c
    values = []
    def case(label, action, c):
        parameters = export_parameters(c)
        output = dict(compute(dict(parameters=parameters, action=action, seed=23,
                                  input_sha256=hashlib.sha256(b'').hexdigest()), b''))
        metadata = json.loads(output['m06.ptb.json'])
        values.append(dict(case=label, action=action, parameters=parameters, seed=23,
                           config=metadata['config'],
                           wav=base64.b64encode(output['synthesis.wav']).decode() if 'synthesis.wav' in output else None))
        return metadata['config']
    old = case('generate-with-pause', 'generate', config('a a', .6))
    old['sequence'] = 'a'
    plain = case('regenerate-without-pause', 'generate', old)
    case('synthesize-without-old-hole', 'synthesize', plain)
    short = case('generate-short-pause', 'generate', config('a //////i+', .3))
    case('synthesize-35-sample-pause', 'synthesize', short)
    write_json(path, values)


def qt(out, scratch):
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS', '--disable-gpu')
    from PyQt6.QtCore import QEventLoop, QTimer, QCoreApplication, QEvent
    from PyQt6.QtWidgets import QApplication, QFileDialog
    from ptb_desktop.host import Workbench, register_scheme
    from ptb_worker.local_acoustic_files import initialize_local_files
    from phonetic_core.synthesis.klatt.api import import_parameters
    from scipy.io import wavfile
    import numpy as np
    db = scratch / 'jobs.sqlite3'
    with sqlite3.connect((ROOT / 'output/validation/p06/local-state.sqlite3').as_uri() + '?mode=ro', uri=True) as source, sqlite3.connect(db) as dest:
        source.backup(dest)
    cache = scratch / 'cache'
    cache.mkdir()
    initialize_local_files(cache)
    saved = scratch / 'saved'
    saved.mkdir()
    register_scheme()
    app = QApplication(['M06-R6-owned-QA'])
    app.setQuitOnLastWindowClosed(False)
    window = Workbench(ROOT / 'frontend/dist', test=True, jobs_path=db,
                       local_files_root=cache, vocal_profile=scratch / 'vocal-profile', start_module='M06')
    window.resize(1440, 1000)
    window.show()
    report = dict(success=False, checks=[], scope='Actual Windows Qt offscreen, built shared frontend and real durable worker; synthetic input')
    def pause(ms=100):
        loop = QEventLoop()
        QTimer.singleShot(ms, loop.quit)
        loop.exec()
    def js(code):
        loop = QEventLoop()
        result = []
        window.page.runJavaScript(code, lambda value: (result.append(value), loop.quit()))
        QTimer.singleShot(5000, loop.quit)
        loop.exec()
        return result[0] if result else None
    def until(code, seconds=120):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if js(code):
                return
            pause()
        raise AssertionError(code + '\n' + str(js('document.body.innerText.slice(-4000)')))
    def click(text):
        js('[...document.querySelectorAll("button")].find(b=>b.offsetParent&&b.textContent.trim()===' + json.dumps(text) + ')?.click()')
    def field(label, value):
        js('(()=>{const e=document.querySelector(' + json.dumps('[aria-label="' + label + '"]') + ');e.value=' + json.dumps(value) + ';e.dispatchEvent(new Event("input",{bubbles:true}));})()')
    count = 0
    def snapshot():
        nonlocal count
        count += 1
        target = scratch / f'parameters-{count}.csv'
        QFileDialog.getSaveFileName = lambda *a, **k: (str(target), 'CSV')
        click('导出参数')
        deadline = time.monotonic() + 15
        while not target.exists() and time.monotonic() < deadline:
            pause()
        return import_parameters(target.read_text('utf-8-sig'))
    def generate(sequence):
        field('IPA 元音序列', sequence)
        click('生成元音')
        until('document.querySelector(".m06-page")?.getAttribute("aria-busy")==="false"&&document.body.innerText.includes("元音曲线已生成")')
    def synthesize():
        click('合成音频')
        until('document.querySelector(".m06-page")?.getAttribute("aria-busy")==="false"&&document.body.innerText.includes("合成完成")')
        QFileDialog.getExistingDirectory = lambda *a, **k: str(saved)
        click('导出音频')
        until('document.body.innerText.includes("已导出合成结果")')
    try:
        until('!!document.querySelector(".m06-page")')
        field('总时长', '.6')
        click('应用时长')
        original = snapshot()['curves']['AV']
        generate('a a')
        assert snapshot()['curves']['AV'] == original
        generate('a')
        current = snapshot()
        assert current['curves']['AV'] == original and not current['silence']
        synthesize()
        rate, samples = wavfile.read(saved / 'synthesis.wav')
        assert rate == 16000 and len(samples) == 9600
        assert np.sqrt(np.mean(samples[4000:5600].astype(float)**2)) > 100
        report['checks'].append('actual QWebChannel generation a a -> a, complete AV retained and real saved PCM has no old central hole')
        field('总时长', '.3')
        click('应用时长')
        generate('a //////i+')
        short = snapshot()
        assert len(short['silence']) == 1
        synthesize()
        matching = [(p, json.loads(p.read_text('utf-8'))) for p in saved.glob('*m06.ptb.json')]
        path, metadata = next((p, v) for p, v in matching if v['config']['sequence'] == 'a //////i+')
        prefix = path.name[:-len('m06.ptb.json')]
        rate, samples = wavfile.read(saved / (prefix + 'synthesis.wav'))
        start, end = short['silence'][0]
        i0, i1 = round(start * rate), round(end * rate)
        assert i1 - i0 == 35 and not np.any(samples[i0:i1])
        assert metadata['diagnostics']['silence_render_revision'] == 'm06-silence/2'
        report['checks'].append('actual Qt worker/export keeps the requested 35 PCM samples exactly silent and records sample-boundary diagnostics')
        assert js('[...document.querySelectorAll(".parameter-list button")].filter(b=>/^A[1-5] /.test(b.textContent)).every(b=>b.disabled)')
        js('[...document.querySelectorAll(".parameter-list button")].find(b=>b.textContent.startsWith("HNR ")).click()')
        field('曲线覆盖', '100')
        click('应用覆盖')
        until('document.body.innerText.includes("HNR 的有效编辑范围为 0–60 dB")')
        field('曲线覆盖', '60')
        click('应用覆盖')
        legacy = snapshot()
        legacy['curves']['HNR']['override'] = 100
        payload = json.dumps(json.dumps(legacy, ensure_ascii=False))
        js('(()=>{const e=document.querySelector("input[accept=\\".csv,.json\\"]"),d=new DataTransfer();d.items.add(new File([' + payload + '],"legacy.json",{type:"application/json"}));e.files=d.files;e.dispatchEvent(new Event("change",{bubbles:true}));})()')
        until('!!document.querySelector("dialog[open]")')
        click('应用并覆盖')
        click('保存参数草稿')
        until('document.body.innerText.includes("完整参数草稿已保存")')
        js('[...document.querySelectorAll(".parameter-list button")].find(b=>b.textContent.startsWith("F0 ")).click()')
        assert snapshot()['curves']['HNR']['override'] == 100
        report['checks'].append('actual Qt disables A1-A5, rejects new HNR=100 and retains/navigates/saves imported legacy HNR=100')
        for theme in ('light', 'dark'):
            js('document.documentElement.dataset.theme=' + json.dumps(theme))
            js('[...document.querySelectorAll(".parameter-list button")].find(b=>b.textContent.startsWith("HNR ")).click()')
            pause(300)
            window.view.grab().save(str(out / f'qt-{theme}.png'))
        report['checks'].append('light and dark current built Qt interface captured')
        report['success'] = True
    finally:
        write_json(out / 'qt-report.json', report)
        if not report['success']:
            window.view.grab().save(str(out / 'qt-failure.png'))
            (out / 'qt-failure.txt').write_text(str(js('document.body.innerText')), encoding='utf-8')
        window.closing = True
        window.close()
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()
        pause(300)
        app.quit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--science', type=Path, required=True)
    parser.add_argument('--overlay', type=Path, required=True)
    parser.add_argument('--browser-only', action='store_true')
    parser.add_argument('--qt-only', action='store_true')
    args = parser.parse_args()
    configure()
    with verification_run('m06-r6', 'qt' if args.qt_only else 'browser' if args.browser_only else 'acceptance',
                          'Current shared M06, synthetic 0.6 s a a -> a and 0.3 s a //////i+, seed 23; owned Qt/Chrome profiles, copied existing SQLite schema, original Pyodide and native PCM comparison') as (out, scratch):
        print(str(out), flush=True)
        if not args.qt_only:
            oracle = scratch / 'native-oracles.json'
            oracles(oracle)
            config = scratch / 'browser.json'
            write_json(config, dict(out=str(out), scratch=str(scratch), science=str(args.science.resolve()), overlay=str(args.overlay.resolve()), oracles=str(oracle)))
            result = subprocess.run(['node', str(ROOT / 'tests/e2e/m06-r6.cjs'), str(config)], cwd=ROOT,
                                    capture_output=True, text=True, encoding='utf-8', creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            (out / 'browser.log').write_text(result.stdout + '\n' + result.stderr, encoding='utf-8')
            if result.returncode:
                raise RuntimeError('Browser M06 verification failed; see browser.log')
        if not args.browser_only:
            qt(out, scratch)
        write_json(out / 'scope.json', dict(success=True, windows_qt=not args.browser_only, chrome_pyodide=not args.qt_only,
                                          macos='shared implementation only; no physical macOS/Safari', production='not deployed', packages='not rebuilt'))


if __name__ == '__main__':
    main()
