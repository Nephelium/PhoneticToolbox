"""Recreate M06 manual examples through actual Qt generate/synthesize/export.

No UI or algorithm changes. Check each result's effective formant curves before
installing it under the existing manual asset IDs. Test files live in scratch.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from workbench_source import bind_sources
bind_sources(ROOT)
from verification_artifacts import verification_run

os.environ['QT_QPA_PLATFORM'] = 'windows'
os.environ['QTWEBENGINE_CHROMIUM_FLAGS'] = '--mute-audio --autoplay-policy=no-user-gesture-required'
for variable in ('QT_ENABLE_HIGHDPI_SCALING', 'QT_SCALE_FACTOR', 'QT_SCREEN_SCALE_FACTORS'):
    os.environ.pop(variable, None)

CASES = [('a', '常态浊声', [730, 1090, 2440]),
         ('i', '常态浊声', [270, 2290, 3010]),
         ('u', '常态浊声', [300, 870, 2240]),
         ('a', '气声', [730, 1090, 2440])]
LESSONS = [dict(key='m06-vowel-syntax-complex', sequence='a-i/- ///u+/e++o/-', duration=2., fade_in=50, fade_out=100),
           dict(key='m06-fade-100-100', sequence='a a', duration=1., fade_in=100, fade_out=100),
           dict(key='m06-fade-30-80', sequence='a a', duration=1., fade_in=30, fade_out=80)]


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', 'utf-8')


def effective(curve):
    return [curve['override']] if curve['override'] is not None else [v for _, v in curve['points']]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', help='Install the selected examples and figures in the manual')
    parser.add_argument('--lessons-only', action='store_true', help='Prepare complex IPA and the two fade comparisons')
    parser.add_argument('--run-directory', type=Path, help=argparse.SUPPRESS)
    options = parser.parse_args()
    if options.run_directory is None:
        # Qt's renderer helpers end when the capture interpreter exits. Own that
        # whole child before the outer scratch finalizer checks live processes.
        with verification_run('m06-manual', 'qt-examples', 'Actual hidden maximized Windows Qt, native DPR 1.5, page 100%; a/i/u at 150 Hz, 0.8 s, modal preset; a breathy comparison; generate then synthesize then native export') as (out, scratch):
            command = [sys.executable, '-X', 'utf8', str(Path(__file__).resolve()), '--run-directory', str(out)]
            if options.apply:
                command.append('--apply')
            if options.lessons_only:
                command.append('--lessons-only')
            completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding='utf-8', creationflags=subprocess.CREATE_NO_WINDOW)
            print(completed.stdout, end='', flush=True)
            if completed.returncode:
                print(completed.stderr, file=sys.stderr, flush=True)
                completed.check_returncode()
        return
    import numpy as np
    import soundfile as sf
    from PyQt6.QtCore import Qt, QEvent, QEventLoop, QTimer
    from PyQt6.QtWidgets import QApplication, QFileDialog
    from ptb_desktop.host import Workbench, register_scheme
    from ptb_worker.local_acoustic_files import initialize_local_files

    project_path = ROOT / 'manual/project.json'
    initial_project = json.loads(project_path.read_text('utf-8'))
    old_assets = {a['id']: a for a in initial_project['assets']}
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        raise RuntimeError('Existing FFmpeg is required')
    out = options.run_directory.resolve()
    if not out.is_relative_to(ROOT / 'output/validation/m06-manual') or not (out / 'run.json').is_file():
        raise ValueError('Capture requires its owned validation run')
    with nullcontext((out, out / 'scratch')) as (out, scratch):
        db = scratch / 'jobs.sqlite3'
        with sqlite3.connect((ROOT / 'output/validation/p06/local-state.sqlite3').as_uri() + '?mode=ro', uri=True) as source, sqlite3.connect(db) as target:
            assert not source.execute("SELECT 1 FROM jobs WHERE state IN ('queued','running','cancel_requested')").fetchone()
            source.backup(target)
        cache = scratch / 'cache'
        cache.mkdir()
        initialize_local_files(cache)
        register_scheme()
        app = QApplication(['M06-owned-manual-examples'])
        window = Workbench(ROOT / 'frontend/dist', test=True, jobs_path=db,
                           local_files_root=cache, vocal_profile=scratch / 'vocal')
        window.resize(window.screen().geometry().width(), window.screen().geometry().height())
        window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
        window.showMaximized()
        window.view.setZoomFactor(1.)
        window.page.setAudioMuted(True)
        report = dict(success=False, applied=False, cases=[], old_examples=[],
                      scope='Actual Qt tasks and native export; digital verification; no physical listening, frozen EXE or other platform verification')
        staged = []

        def pause(ms=100):
            loop = QEventLoop()
            QTimer.singleShot(ms, loop.quit)
            loop.exec()

        def js(code):
            values = []
            loop = QEventLoop()
            window.page.runJavaScript(code, lambda v: (values.append(v), loop.quit()))
            QTimer.singleShot(5000, loop.quit)
            loop.exec()
            if not values:
                raise RuntimeError('Qt JavaScript timeout')
            return values[0]

        def until(code, seconds=90):
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                if js(code):
                    return
                pause()
            raise AssertionError(code + '\n' + str(js("document.querySelector('.m06-page')?.innerText")))

        def click(label):
            assert js("(()=>{const e=[...document.querySelectorAll('button')].find(e=>e.offsetParent&&e.textContent.trim()===" + json.dumps(label) + ");if(!e||e.disabled)return false;e.click();return true})()"), label
            pause()

        def fill(label, value, select=False):
            selector = ('select' if select else 'input') + '[aria-label="' + label + '"]'
            assert js("(()=>{const e=document.querySelector(" + json.dumps(selector) + ");if(!e)return false;e.value=" + json.dumps(str(value)) + ";e.dispatchEvent(new Event('input',{bubbles:true}));e.dispatchEvent(new Event('change',{bubbles:true}));return true})()")
            pause()

        def fade(label, value):
            assert js("(()=>{const l=[...document.querySelectorAll('.base-settings label')].find(e=>e.textContent.includes(" + json.dumps(label) + "));const e=l?.querySelector('input');if(!e)return false;e.value=" + json.dumps(str(value)) + ";e.dispatchEvent(new Event('input',{bubbles:true}));return true})()")
            pause()

        try:
            until("!!document.querySelector('.home-page')&&document.fonts.status==='loaded'")
            click('设置')
            click('浅色')
            click('声学参数合成')
            until("!!document.querySelector('.m06-page')")
            js("document.querySelector('[aria-label=\"收起侧栏\"]')?.click()")
            cases = LESSONS if options.lessons_only else [dict(key='m06-' + v + '-' + ('modal' if p == '常态浊声' else 'breathy') + '-150', sequence=v, duration=.8, fade_in=50, fade_out=100, preset=p, formants=f) for v, p, f in CASES]
            for case in cases:
                vowel, phonation, formants = case['sequence'], case.get('preset', '常态浊声'), case.get('formants')
                key, duration = case['key'], case['duration']
                js("document.querySelector('.parameter-list button')?.click()")
                pause()
                click('清空参数')
                click('应用并覆盖')
                fill('总时长', duration)
                click('应用时长')
                fade('淡入', case['fade_in'])
                fade('淡出', case['fade_out'])
                fill('曲线覆盖', 150)
                click('应用覆盖')
                fill('发声类型预设', phonation, select=True)
                click('应用预设')
                click('应用并覆盖')
                fill('IPA 元音序列', vowel)
                click('生成元音')
                until("document.querySelector('.m06-page').getAttribute('aria-busy')==='false'&&document.querySelector('.m06-page').innerText.includes('元音曲线已生成')")
                # Only a generated configuration may be sent to synthesis.
                click('合成音频')
                until("document.querySelector('.m06-page').getAttribute('aria-busy')==='false'&&document.querySelector('.m06-page').innerText.includes('合成完成')")
                saved = scratch / key
                saved.mkdir()
                QFileDialog.getExistingDirectory = lambda *args, directory=saved, **kwargs: str(directory)
                click('导出音频')
                until("document.querySelector('.m06-page').innerText.includes('已导出合成结果')")
                metadata = json.loads((saved / 'm06.ptb.json').read_text('utf-8'))
                config = metadata['config']
                assert config['sequence'] == vowel and config['duration'] == duration
                assert config['sample_rate'] == 16000 and effective(config['curves']['F0']) == [150]
                assert config['fade_in'] == case['fade_in'] and config['fade_out'] == case['fade_out']
                if formants:
                    for name, value in zip(('F1', 'F2', 'F3'), formants):
                        assert np.allclose(effective(config['curves'][name]), value, rtol=0, atol=1e-6), (key, name)
                elif key == 'm06-vowel-syntax-complex':
                    weights = [.9, .45, .125, .55, 1.21, .45]
                    boundaries = np.cumsum(np.array(weights) * duration / 3.685)
                    assert np.allclose(config['boundaries'], boundaries[:-1], rtol=0, atol=1e-12)
                    assert np.allclose(config['silence'], [[boundaries[1], boundaries[2]]], rtol=0, atol=1e-12)
                    for i, values in zip((0, 1, 3, 4, 5), ([730,1090,2440], [270,2290,3010], [300,870,2240], [390,2030,2600], [460,800,2250])):
                        midpoint = ((boundaries[i-1] if i else 0) + boundaries[i]) / 2
                        for name, value in zip(('F1','F2','F3'), values):
                            points = np.array(config['curves'][name]['points'])
                            assert abs(np.interp(midpoint, points[:,0], points[:,1])-value) < 1e-6, (i, name)
                else:
                    assert np.allclose(config['boundaries'], [1/3, 2/3], rtol=0, atol=1e-12)
                    assert np.allclose(config['silence'], [[1/3, 2/3]], rtol=0, atol=1e-12)
                assert metadata['computation_revision'] == 'klatt/2'
                audio, rate = sf.read(saved / 'synthesis.wav')
                frames = round(duration * rate)
                assert rate == 16000 and audio.shape == (frames,) and np.isfinite(audio).all() and np.any(audio)
                for start, end in config['silence']:
                    assert not np.any(audio[round(start*rate):round(end*rate)])
                mp3 = saved / 'listening.mp3'
                subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-nostdin', '-i', str(saved / 'synthesis.wav'), '-map_metadata', '-1', '-c:a', 'libmp3lame', '-q:a', '2', str(mp3)], check=True, creationflags=subprocess.CREATE_NO_WINDOW)
                decoded = subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-nostdin', '-i', str(mp3), '-f', 'f32le', '-ac', '1', 'pipe:1'], capture_output=True, check=True, creationflags=subprocess.CREATE_NO_WINDOW).stdout
                decoded = np.frombuffer(decoded, dtype='<f4')
                assert len(decoded) == frames and np.isfinite(decoded).all()
                assert np.corrcoef(audio, decoded)[0, 1] > .97
                if key == 'm06-vowel-syntax-complex':
                    # Show formant curves and their generated segment boundaries.
                    js("document.querySelectorAll('.parameter-list button')[9].click()")
                    pause(200)
                image = window.view.grab().toImage()
                native_size = [image.width(), image.height()]
                dpr = image.devicePixelRatio()
                assert abs(dpr - 1.5) < .01 and native_size in ([2560, 1600], [2561, 1601]), (native_size, dpr)
                if native_size != [2560, 1600]:
                    image = image.copy(0, 0, 2560, 1600)
                figure = out / (key + '.png')
                assert image.save(str(figure))
                # Preserve the exact task snapshot/seed, omit redundant raster data.
                settings = {k: deepcopy(v) for k, v in metadata.items() if k not in ('spectrograms', 'curve_time_axis', 'audio_time_axis')}
                wav_id = key + ('-wav' if options.lessons_only else '-approved-r20261007')
                mp3_id = key + '-mp3' if options.lessons_only else wav_id + '-mp3-r20261007'
                if wav_id in old_assets:
                    old = old_assets[wav_id]
                    old_config = old.get('settings', {}).get('config', {})
                    report['old_examples'].append(dict(id=wav_id, sha256=old['sha256'], sequence=old_config.get('sequence'),
                        formants={n: effective(old_config['curves'][n]) for n in ('F1', 'F2', 'F3')}))
                record = dict(id=key, vowel=vowel, preset=phonation, control_formants_hz=formants,
                    sample_rate_hz=rate, frames=len(audio), seed=metadata['seed'], config=config,
                    wav_sha256=sha((saved / 'synthesis.wav').read_bytes()), mp3_sha256=sha(mp3.read_bytes()),
                    pcm_sha256=sha(audio.tobytes()), mp3_roundtrip_correlation=float(np.corrcoef(audio, decoded)[0, 1]),
                    figure=figure.relative_to(ROOT).as_posix(), figure_sha256=sha(figure.read_bytes()),
                    native_size=native_size, render_size=[2560, 1600], device_pixel_ratio=dpr)
                report['cases'].append(record)
                staged.append((wav_id, saved / 'synthesis.wav', settings, record))
                staged.append((mp3_id, mp3, settings, record))
                if options.lessons_only or vowel == 'a':
                    staged.append((key + ('-light' if options.lessons_only else '-light-scaled-r20261007'), figure, None, record))
                print('Verified ' + key + ' F1–F3=' + str(formants), flush=True)

            assert len({r['pcm_sha256'] for r in report['cases'][:3]}) == 3
            if options.apply:
                # Re-read the current index to preserve concurrent chapter edits.
                current = json.loads(project_path.read_text('utf-8'))
                for asset_id, source, settings, record in staged:
                    asset = next((a for a in current['assets'] if a['id'] == asset_id), None)
                    if asset is None:
                        assert options.lessons_only
                        asset = dict(id=asset_id, path='assets/software-only/m06-examples/' + asset_id + source.suffix,
                            kind='image' if settings is None else 'audio', mime='image/png' if settings is None else 'audio/wav' if source.suffix=='.wav' else 'audio/mpeg',
                            distribution='software-only', git=False, caption=record['vowel'] + '；常态浊声；F0 控制值 150 Hz；时长 ' + str(record['config']['duration']) + ' 秒。')
                        if settings is None:
                            asset.update(width=2560, height=1600, sourceType='实际应用截图')
                        elif source.suffix == '.mp3':
                            asset.update(originalAssetId=record['id']+'-wav', listeningCopy=True)
                        current['assets'].append(asset)
                    else:
                        assert asset['sha256'] == old_assets[asset_id]['sha256'], 'Asset changed concurrently: ' + asset_id
                    destination = ROOT / 'manual' / asset['path']
                    assert destination.resolve().is_relative_to((ROOT / 'manual').resolve())
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    payload = source.read_bytes()
                    destination.write_bytes(payload)
                    asset['sha256'] = sha(payload)
                    asset['source'] = '当前 M06 分别生成元音曲线后合成并导出；元音、F0、时长及声源预设均已核对。'
                    asset['captureReport'] = (out / 'report.json').relative_to(ROOT).as_posix()
                    if settings is not None:
                        asset['settings'] = settings
                        asset.update(duration=record['config']['duration'], sampleRate=16000, channels=1, sourceType='Klatt 合成示例')
                        if source.suffix == '.mp3':
                            asset['originalSha256'] = record['wav_sha256']
                    else:
                        asset['capture'] = dict(report=asset['captureReport'], theme='light', renderSize=[2560, 1600],
                            nativeSize=record['native_size'], devicePixelRatio=record['device_pixel_ratio'],
                            navigationCollapsed=True, maximized=True)
                        asset.pop('regions', None)
                        asset.pop('originalSha256', None)
                write_json(project_path, current)
                report['applied'] = True
            report['success'] = True
        except Exception as error:
            report['error'] = str(error)
            window.view.grab().save(str(out / 'failed.png'))
            raise
        finally:
            write_json(out / 'report.json', report)
            window.closing = True
            window.close()
            pause(500)
            window.deleteLater()
            app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            app.processEvents()
            app.quit()
            print(json.dumps(dict(out=str(out), success=report['success'], applied=report['applied'])), flush=True)


if __name__ == '__main__':
    main()
