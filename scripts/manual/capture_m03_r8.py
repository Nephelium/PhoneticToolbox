"""M03-R8 manual figures from the existing author-approved natural EGG recording."""
import hashlib
import argparse
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
DIRS = ('desktop/src', 'backend/src', 'packages/phonetic_core/src', 'scripts')
for directory in DIRS:
    sys.path.insert(0, str(ROOT / directory))
os.environ['PYTHONPATH'] = os.pathsep.join(str(ROOT / directory) for directory in DIRS)
for name in ('QT_ENABLE_HIGHDPI_SCALING', 'QT_SCALE_FACTOR', 'QT_SCREEN_SCALE_FACTORS'):
    os.environ.pop(name, None)
os.environ['QT_QPA_PLATFORM'] = 'windows'
os.environ['QTWEBENGINE_CHROMIUM_FLAGS'] = '--mute-audio --autoplay-policy=no-user-gesture-required'
os.environ['PTB_EGG_PYTHON'] = str(ROOT / '.venv/m03-compatible/python.exe')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main(args):
    from verification_artifacts import verification_run
    from manual.capture_locale import CaptureLocale
    from PyQt6 import sip
    from PyQt6.QtCore import Qt, QEventLoop, QTimer
    from PyQt6.QtWidgets import QApplication, QFileDialog
    from ptb_desktop.host import Workbench, register_scheme
    from ptb_worker.local_acoustic_files import initialize_local_files

    ui = CaptureLocale(args.locale, ('egg-analysis', 'workbench', 'components'))
    source_paths = [Path(__file__), ROOT / 'scripts/manual/capture_locale.py', args.frontend_dist / 'index.html',
                    ROOT / 'desktop/src/ptb_desktop/host.py', ROOT / 'desktop/src/ptb_desktop/presentation.py']
    if args.extra:
        source_paths.append(ROOT / 'scripts/manual/capture_m03_extra.py')
    source_paths += list((ROOT / 'frontend/src/modules/egg-analysis').glob('*.vue'))
    source_paths += [ROOT / f'frontend/src/i18n/locales/{language}/egg-analysis.json' for language in ('zh-CN', args.locale)]
    sources = {str(i): {'path': str(path), 'sha256': digest(path)} for i, path in enumerate(dict.fromkeys(source_paths))}
    source = Path(json.loads((ROOT / 'local-data/manual-authoring/sources.json').read_text('utf-8'))['egg'])
    source_sha = digest(source)
    with verification_run('m03-r8', args.locale, 'Existing author-approved natural stereo EGG recording; 40.00–40.50 s; hidden maximized Qt, native screen DPI, page 100%') as (out, scratch):
        inputs = scratch / 'inputs'
        inputs.mkdir()
        copied = inputs / source.name
        shutil.copyfile(source, copied)
        assert digest(copied) == source_sha
        db = scratch / 'jobs.sqlite3'
        with sqlite3.connect((ROOT / 'output/validation/p06/local-state.sqlite3').as_uri() + '?mode=ro', uri=True) as template, sqlite3.connect(db) as target:
            assert not template.execute("SELECT 1 FROM jobs WHERE state IN ('queued','running','cancel_requested')").fetchone()
            template.backup(target)
        template.close()
        target.close()
        cache = scratch / 'cache'
        cache.mkdir()
        initialize_local_files(cache)
        register_scheme()
        app = QApplication(['M03-R8-owned-manual'])
        app.setQuitOnLastWindowClosed(False)
        window = Workbench(args.frontend_dist.resolve(), test=True, jobs_path=db, local_files_root=cache,
                           vocal_profile=scratch / 'vocal', reaper_binary=ROOT / 'phonetic_toolbox/core/acoustic/reaper.exe')
        window.resize(window.screen().geometry().width(), window.screen().geometry().height())
        window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
        window.showMaximized()
        window.view.setZoomFactor(1.)
        window.page.setAudioMuted(True)
        folder_choice = {'path': inputs}
        QFileDialog.getExistingDirectory = lambda *args, **kwargs: str(folder_choice['path'])
        report = dict(success=False, source_sha256=source_sha, natural_recording=True, original_unchanged=False,
                      locale=args.locale, sourceChapter='m03', sources=sources,
                      captures=[], scope='Actual Windows Qt; screen DPR and page zoom observed; no physical audio/device test')

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
            if not values:
                raise RuntimeError('Qt JavaScript timeout')
            return values[0]

        def until(code, seconds=120):
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                if js(code):
                    return
                pause()
            raise AssertionError('Qt timeout: ' + code + '; ' + str(js("document.querySelector('.egg-page')?.innerText.slice(0,1200)")))

        def click(text):
            text = ui.label(text)
            assert js("(()=>{const b=[...document.querySelectorAll('button')].find(b=>b.offsetParent&&(b.textContent.trim()||b.title)===" + json.dumps(text, ensure_ascii=False) + ");if(!b||b.disabled)return false;b.click();return true})()"), text
            pause()

        def idle():
            pause(100)
            until("document.querySelector('.egg-live-status')?.textContent===" + json.dumps(ui.key('modules.egg-analysis.live_preview')) + "&&[...document.querySelectorAll('.egg-page button')].some(b=>b.textContent.trim()===" + json.dumps(ui.key('modules.egg-analysis.save_csv_three_plots')) + "&&!b.disabled)")

        def fill(label, value):
            label = ui.label(label)
            assert js("(()=>{const e=document.querySelector('input[aria-label=" + json.dumps(label, ensure_ascii=False) + "]');if(!e)return false;e.value=" + json.dumps(str(value)) + ";e.dispatchEvent(new Event('input',{bubbles:true}));e.dispatchEvent(new Event('change',{bubbles:true}));return true})()")
            idle()

        def tracks(praat=False, gci=False, reaper=False):
            js("(()=>{const boxes=document.querySelectorAll('.egg-checks input');const desired=" + json.dumps([praat, gci, reaper]) + ";boxes.forEach((e,i)=>{if(e.checked!==desired[i])e.click()})})()")
            idle()

        def capture(key, *, ready=True):
            if ready:
                idle()
            until("document.fonts.status==='loaded'")
            pause(350)
            image = window.view.grab().toImage()
            original_size = [image.width(), image.height()]
            assert abs(image.devicePixelRatio() - 1.5) < .01, image.devicePixelRatio()
            assert original_size in ([2560, 1600], [2561, 1601]), original_size
            if original_size != [2560, 1600]:
                image = image.copy(0, 0, 2560, 1600)
            path = out / (key + '.png')
            assert image.save(str(path))
            metrics = ui.metrics(js)
            report['captures'].append(dict(id=key, name=key, file=path.name, sha256=digest(path), image=[2560, 1600],
                locale=metrics['locale'], fontLocale=metrics['fontLocale'], fontFamily=metrics['fontFamily'],
                raw_image=original_size, dpr=metrics['dpr'], pageZoom=window.view.zoomFactor(),
                maximized=window.isMaximized(), theme=js('document.documentElement.dataset.theme'),
                navigationCollapsed=js("!!document.querySelector('[aria-label=' + CSS.escape(" + json.dumps(ui.label('展开侧栏')) + ") + ']')"),
                controls=js("Object.fromEntries([...document.querySelectorAll('.egg-parameter-row input,.egg-control-row input')].map(e=>[e.getAttribute('aria-label')||e.closest('label')?.textContent,e.type==='checkbox'?e.checked:e.value]))")))
            print('Captured ' + key, flush=True)

        try:
            until("!!document.querySelector('.home-page')&&document.fonts.status==='loaded'")
            ui.prepare_qt(js, until)
            click('设置')
            click('浅色')
            click('EGG 信号分析')
            js("document.querySelector('[aria-label=' + CSS.escape(" + json.dumps(ui.label('收起侧栏')) + ") + ']')?.click()")
            click('打开音频目录')
            until("document.querySelectorAll('.egg-source select option').length>1")
            assert js("(()=>{const e=document.querySelector('.egg-source select'),o=[...e.options].find(o=>o.textContent.trim()===" + json.dumps(source.name, ensure_ascii=False) + ");if(!o)return false;e.value=o.value;e.dispatchEvent(new Event('change',{bubbles:true}));return true})()")
            idle()
            js("(()=>{const inputs=document.querySelectorAll('.egg-bottom-bar .selection-controls input');for(const [i,value] of [[1,40.5],[0,40]]){inputs[i].value=value;inputs[i].dispatchEvent(new Event('input',{bubbles:true}));inputs[i].dispatchEvent(new Event('change',{bubbles:true}))}})()")
            idle()
            if args.extra:
                from manual.capture_m03_extra import run
                run(ui=ui, js=js, until=until, pause=pause, click=click, idle=idle, capture=capture,
                    report=report, out=out, scratch=scratch, source_sha=source_sha,
                    folder_choice=folder_choice, database=db, window=window, selected=args.extra_scene)
                report['original_unchanged'] = digest(source) == source_sha
                assert report['original_unchanged']
                report['success'] = True
                return
            capture('m03-r8-analysis-roi')
            fill('EGG 微观窗口', 10)
            capture('m03-r8-micro-events')
            fill('EGG 微观窗口', 50)
            fill('EGG F0 下限 Hz', 30)
            fill('EGG F0 上限 Hz', 800)
            tracks(True)
            capture('m03-r8-praat-only')
            tracks(gci=True)
            capture('m03-r8-gci-only')
            tracks()
            js("(()=>{const e=document.querySelector('.egg-spectral-grid input');e.value=50;e.dispatchEvent(new Event('input',{bubbles:true}))})()")
            capture('m03-r8-spec-50ms')
            js("(()=>{const e=document.querySelector('.egg-spectral-grid input');e.value=5;e.dispatchEvent(new Event('input',{bubbles:true}))})()")
            capture('m03-r8-spec-5ms')
            js("(()=>{const e=document.querySelector('.egg-spectral-grid input');e.value=20;e.dispatchEvent(new Event('input',{bubbles:true}))})()")
            tracks(True, True, True)
            capture('m03-r8-three-f0')
            fill('EGG F0 下限 Hz', 80)
            fill('EGG F0 上限 Hz', 350)
            capture('m03-r8-shared-f0-range')
            assert not js("document.querySelector('.egg-colorbar')")
            assert js("document.querySelectorAll('.egg-four-plots .plot-highlight').length===2")
            report['original_unchanged'] = digest(source) == source_sha
            assert report['original_unchanged']
            report['success'] = True
        except Exception as error:
            report['error'] = str(error)
            window.view.grab().save(str(out / 'failed.png'))
            raise
        finally:
            report['sourceChangesDuringRun'] = [item['path'] for item in sources.values() if digest(Path(item['path'])) != item['sha256']]
            (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', 'utf-8')
            print(json.dumps({'out': str(out), 'captures': len(report['captures']), 'success': report['success']}), flush=True)
            js('window.onbeforeunload=null')
            window.closing = True
            window.close()
            app.processEvents()
            sip.delete(window)
            app.processEvents()
            sip.delete(app)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--locale', choices=('zh-CN', 'en', 'ja'), default='zh-CN')
    parser.add_argument('--frontend-dist', type=Path, default=ROOT / 'frontend/dist')
    parser.add_argument('--extra', action='store_true', help='Remaining batch/export/inverse and guard states using the same authorised recording')
    parser.add_argument('--extra-scene', action='append', default=[], choices=('inverse-full-egg', 'batch-saved'), help='Only retain selected final correction scenes; required real setup still runs')
    main(parser.parse_args())
