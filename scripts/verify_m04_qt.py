"""M04-E actual Qt host and two previously authorized P03 natural WAVs, local only."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
from uuid import uuid4

import numpy as np
from scipy.io import wavfile
from PyQt6.QtCore import QEventLoop, QTimer, Qt
from PyQt6.QtWidgets import QApplication, QFileDialog

from ptb_desktop.host import Workbench, register_scheme
from ptb_worker.local_acoustic_files import initialize_local_files

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT / 'output/validation/m04-e' / ('qt-' + uuid4().hex)
    out.mkdir(parents=True)
    db = out / 'jobs.sqlite3'
    with sqlite3.connect((ROOT / 'output/validation/p06/local-state.sqlite3').as_uri() + '?mode=ro', uri=True) as source, sqlite3.connect(db) as target:
        assert not source.execute("SELECT 1 FROM jobs WHERE state IN ('queued','running','cancel_requested')").fetchone()
        source.backup(target)
    cache = out / 'cache'
    cache.mkdir()
    initialize_local_files(cache)
    inputs = out / 'inputs'
    inputs.mkdir()
    saved = out / 'saved'
    saved.mkdir()
    baseline = ROOT / 'output/validation/p03/20260909-165413'
    natural = []
    for key in ('LOCAL-01', 'LOCAL-12'):
        request = json.loads((baseline / key / '1/request.json').read_text('utf-8'))
        original = Path(request['input'])
        grid = Path(request['textgrid'])
        with gzip.open(baseline / key / '1/result.json.gz', 'rt', encoding='utf-8') as stream:
            frozen = json.load(stream)
        audio_hash = hashlib.sha256(original.read_bytes()).hexdigest()
        assert audio_hash == frozen['input_sha256']
        originals = [(p, hashlib.sha256(p.read_bytes()).hexdigest()) for p in (original, grid)]
        (inputs / (key + '.wav')).write_bytes(original.read_bytes())
        (inputs / (key + '.TextGrid')).write_bytes(grid.read_bytes())
        rate, samples = wavfile.read(original)
        mono = samples.astype(np.float64).mean(axis=1) if samples.ndim == 2 else samples.astype(np.float64)
        block = round(rate * .2)
        count = len(mono) // block
        energies = np.abs(mono[:count * block]).reshape(count, block).mean(axis=1)
        first = int(np.argmax(energies)) * block
        natural.append(dict(key=key, originals=originals, audio_hash=audio_hash, rate=rate,
                            start=first / rate, end=(first + block) / rate, samples=block))
    os.environ['PTB_EGG_PYTHON'] = str(ROOT / '.venv/m03-compatible/python.exe')
    register_scheme()
    app = QApplication(['M04-owned-QA'])
    app.setApplicationName('M04-owned-QA')
    window = Workbench(ROOT / 'frontend/dist', test=True, jobs_path=db, local_files_root=cache, vocal_profile=out / 'vocal-profile')
    window.resize(1440, 1000)
    window.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
    window.show()
    QFileDialog.getExistingDirectory = lambda *a, **k: str(saved if '结果' in str(a) else inputs)
    report = dict(success=False, checks=[], natural=[], schema_applied=[])

    def pause(milliseconds=100):
        loop = QEventLoop()
        QTimer.singleShot(milliseconds, loop.quit)
        loop.exec()

    def js(code):
        loop = QEventLoop()
        box = []
        window.page.runJavaScript(code, lambda value: (box.append(value), loop.quit()))
        QTimer.singleShot(5000, loop.quit)
        loop.exec()
        return box[0] if box else None

    def until(code, seconds=60):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if js(code):
                return
            pause()
        raise AssertionError('UI timeout: ' + code + '; ' + str(js('document.querySelector(".lpc-page")?.innerText.slice(0,1300)')))

    def click(text):
        js('[...document.querySelectorAll("button")].find(b=>b.offsetParent&&b.textContent.trim()===' + json.dumps(text, ensure_ascii=False) + ')?.click()')

    def fill(label, value):
        js('(()=>{const e=document.querySelector(' + json.dumps('input[aria-label="' + label + '"]') + ');e.value=' + json.dumps(str(value)) + ';e.dispatchEvent(new Event("input",{bubbles:true}));e.dispatchEvent(new Event("change",{bubbles:true}));})()')

    def select(name):
        js('(()=>{const e=document.querySelector(".lpc-files select");e.value=[...e.options].find(o=>o.textContent===' + json.dumps(name, ensure_ascii=False) + ').value;e.dispatchEvent(new Event("change",{bubbles:true}));})()')

    try:
        until('document.documentElement.style.getPropertyValue("--font").length>0')
        click('LPC 谱图')
        until('!!document.querySelector(".lpc-page")')
        click('打开 WAV 目录')
        until('document.querySelectorAll(".lpc-files select:first-of-type option").length>=3')
        for case in natural:
            select(case['key'] + '.wav')
            until('!!document.querySelector(".lpc-page .wave-track svg")')
            until('!document.querySelector(".lpc-files").innerText.includes("正在读取音频")')
            assert js('document.querySelector(".lpc-files").innerText.includes("44100 Hz")')
            until("!!document.querySelector('[aria-label=\"LPC 标注层\"]')?.value")
            until('!document.querySelector(".lpc-files").innerText.includes("正在读取 TextGrid")')
            fill('LPC 选区起点', case['start'])
            fill('LPC 选区终点', case['end'])
            until('Math.abs(Number(document.querySelector(\'[aria-label="LPC 选区起点"]\').value)-'+str(case['start'])+')<1e-6 && Math.abs(Number(document.querySelector(\'[aria-label="LPC 选区终点"]\').value)-'+str(case['end'])+')<1e-6')
            click('开始分析')
            until('!!document.querySelector(".lpc-spectrum svg") && document.querySelector(".lpc-spectrum")?.offsetParent!==null && !document.querySelector(".view-tabs").innerText.includes("正在读取结果") && document.querySelector(".lpc-main").innerText.includes('+json.dumps(case['key']+'.wav · '+f"{case['start']:.6f}",ensure_ascii=False)+')', 90)
            with sqlite3.connect(db) as conn:
                rows = conn.execute("SELECT snapshot,result_manifest FROM jobs WHERE state='succeeded' ORDER BY created_at DESC").fetchall()
            snapshot, manifest = next((json.loads(a), json.loads(b)) for a, b in rows if case['key'] + '.wav' in a)
            meta_file = next(f for f in manifest['files'] if f['name'] == 'lpc.ptb.json')
            metadata = json.loads((cache / (meta_file['id'] + '.bin')).read_text('utf-8'))
            assert metadata['input_sha256'] == case['audio_hash']
            assert metadata['selection']['end_sample'] - metadata['selection']['start_sample'] == case['samples']
            assert len(metadata['spectrum']['frequencies_hz']) == 1024
            assert not js('!![...document.querySelectorAll(".lpc-page .notice")].find(e=>e.textContent.includes("已改变"))')
            assert all(hashlib.sha256(p.read_bytes()).hexdigest() == digest for p, digest in case['originals'])
            window.view.grab().save(str(out / (case['key'] + '-spectrum.png')))
            if case['key'] == 'LOCAL-01':
                click('波形')
                js('document.querySelector(".lpc-page .view-tabs input[type=checkbox]")?.click()')
                until('document.querySelector(".lpc-page .spectrogram-canvas canvas")?.width>500 && !document.querySelector(".lpc-page .spectrogram-view").innerText.includes("正在计算")', 45)
                pause(800)
                until('document.querySelector(".lpc-page .spectrogram-canvas canvas")?.width>500 && !document.querySelector(".lpc-page .spectrogram-view").innerText.includes("正在计算")', 45)
                window.view.grab().save(str(out / 'LOCAL-01-wave-spectrogram.png'))
                js('document.querySelector(".lpc-page .view-tabs input[type=checkbox]")?.click()')
            report['natural'].append(dict(case=case['key'], source_sha256=case['audio_hash'], sample_rate=case['rate'],
                                          roi=metadata['selection'], spectrum_points=1024,
                                          original_hashes_unchanged=True, original_textgrid_associated=True))
        click('选择目录保存完整结果')
        until('document.querySelector(".lpc-page [role=status]")?.textContent.includes("已保存 3")')
        saved_files = list(saved.iterdir())
        assert len(saved_files) == 3
        image = next(p.read_bytes() for p in saved_files if p.suffix == '.png')
        assert image[:8] == b'\x89PNG\r\n\x1a\n' and int.from_bytes(image[16:20], 'big') == 2400 and int.from_bytes(image[20:24], 'big') == 1350
        audio_file = next(p for p in saved_files if p.suffix == '.wav')
        rate, audio = wavfile.read(audio_file)
        assert rate == natural[-1]['rate'] and len(audio) == natural[-1]['samples'] and audio.dtype == np.float64
        saved_meta = json.loads(next(p.read_text('utf-8') for p in saved_files if p.suffix == '.json'))
        assert saved_meta['input_sha256'] == natural[-1]['audio_hash']
        assert all(hashlib.sha256(p.read_bytes()).hexdigest() == digest for case in natural for p, digest in case['originals'])
        report['checks'].append('actual Qt host loads two authorized natural WAV/TextGrid files, calculates short high-energy ROIs, saves PNG/JSON/WAV')
        report['success'] = True
    except Exception as exc:
        report['error'] = str(exc)
        window.view.grab().save(str(out / 'failed.png'))
        raise
    finally:
        window.close()
        app.processEvents()
        (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(out / 'report.json')


if __name__ == '__main__':
    main()
