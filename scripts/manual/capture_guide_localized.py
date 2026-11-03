"""Capture the actual home and shared manual using an isolated author preview."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from verification_artifacts import verification_run
from workbench_source import configure
from manual.author_preview import prepare
from manual.capture_locale import CaptureLocale


def main(args):
    configure()
    os.environ['QT_QPA_PLATFORM'] = 'windows'
    os.environ['QTWEBENGINE_CHROMIUM_FLAGS'] = '--mute-audio'
    for key in ('QT_ENABLE_HIGHDPI_SCALING', 'QT_SCALE_FACTOR', 'QT_SCREEN_SCALE_FACTORS'):
        os.environ.pop(key, None)
    from PyQt6 import sip
    from PyQt6.QtCore import QEventLoop, QTimer, Qt
    from PyQt6.QtWidgets import QApplication
    from ptb_desktop.host import Workbench, register_scheme
    from ptb_worker.local_workspace import prepare_workspace

    ui = CaptureLocale(args.locale, ('workbench', 'common'))
    paths = [Path(__file__), ROOT / 'scripts/manual/author_preview.py', ROOT / 'scripts/manual/capture_locale.py',
             args.frontend_dist / 'index.html', ROOT / 'desktop/src/ptb_desktop/host.py',
             ROOT / 'desktop/src/ptb_desktop/presentation.py', ROOT / 'frontend/src/app/HelpPage.vue',
             ROOT / 'frontend/src/manual/ManualReader.vue', ROOT / 'manual/project.json']
    paths += list((ROOT / 'manual/chapters').glob('*.json'))
    for locale in ('zh-CN', args.locale):
        paths += [ROOT / f'frontend/src/i18n/locales/{locale}/{namespace}.json' for namespace in ('manual', 'workbench')]
    snapshots = {str(i): {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                 for i, path in enumerate(dict.fromkeys(paths))}
    with verification_run('guide-localized', args.locale, 'Actual Qt home and same manual renderer; complete translated author-only preview installed in owned scratch; not a final language pack') as (out, scratch):
        preview = prepare(scratch, args.locale, args.frontend_dist)
        register_scheme()
        app = QApplication(['owned-localized-guide-capture'])
        db, cache = prepare_workspace(scratch / 'state', ROOT / 'backend/migrations')
        window = Workbench(preview['frontend'], test=True, jobs_path=db,
                           local_files_root=cache, vocal_profile=scratch / 'vocal')
        # Inject only the task-owned installed author preview. No personal
        # installed-manual path is ever accessed by this capture host.
        window.bridge.tasks._manual_packs = preview['manager']
        window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
        window.resize(window.screen().geometry().width(), window.screen().geometry().height())
        window.showMaximized()
        window.view.setZoomFactor(1)
        window.page.setAudioMuted(True)
        report = dict(success=False, locale=args.locale, sourceChapter='getting-started', captures=[],
                      sources=snapshots, ownedPid=os.getpid(), authoringPreview=preview['receipt'],
                      physicalCompositorVerified=False, checks=[])

        def pause(ms=120):
            loop = QEventLoop()
            QTimer.singleShot(ms, loop.quit)
            loop.exec()

        def js(code):
            loop, values = QEventLoop(), []
            window.page.runJavaScript(code, lambda value: (values.append(value), loop.quit()))
            QTimer.singleShot(10000, loop.quit)
            loop.exec()
            if not values:
                raise RuntimeError('JavaScript timeout: ' + code)
            return values[0]

        def until(code):
            end = time.monotonic() + 60
            while time.monotonic() < end:
                if js(code):
                    return
                pause()
            raise AssertionError(code + '\n' + str(js('document.body.innerText.slice(-1600)')))

        def click(key):
            label = json.dumps(ui.key(key), ensure_ascii=False)
            expression = '[...document.querySelectorAll("button")].find(b=>b.offsetParent&&(b.getAttribute("aria-label")===' + label + '||b.textContent.trim()===' + label + '))'
            until('!!(' + expression + ')&&!(' + expression + ').disabled')
            js(expression + '.click()')
            pause(200)

        def capture(name):
            until("document.fonts.status==='loaded'")
            pause(500)
            # A new owned profile can receive an automatic announcement after
            # returning home. Dismiss only that actual notice through its UI;
            # never capture an obscured page or click through a modal overlay.
            for _ in range(10):
                if not js("!!document.querySelector('.announcements-panel')"):
                    break
                click('components.got_it')
                pause(250)
            assert not js("!!document.querySelector('dialog[open],[aria-modal=true]')"), 'Unexpected dialog obscures capture'
            metrics = ui.metrics(js)
            image = window.view.grab().toImage()
            raw = [image.width(), image.height()]
            assert image.devicePixelRatio() == 1.5 and raw in ([2560, 1600], [2561, 1601]), raw
            if raw != [2560, 1600]:
                image = image.copy(0, 0, 2560, 1600)
            path = out / (name + '.png')
            assert image.save(str(path))
            report['captures'].append(dict(name=name, file=path.name, image=[2560, 1600], raw_image=raw,
                                           maximized=window.isMaximized(), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                           **{key: metrics[key] for key in ('locale', 'fontLocale', 'fontFamily', 'dpr', 'pageZoom', 'theme')}))
            print('Captured ' + name, flush=True)

        try:
            until("!!document.querySelector('#tool-sidebar')&&document.fonts.status==='loaded'")
            ui.prepare_qt(js, until)
            click('workbench.settings')
            js("document.querySelector('#settings-tab-appearance').click()")
            light = json.dumps(ui.label('浅色'), ensure_ascii=False)
            js('[...document.querySelectorAll("button")].find(b=>b.offsetParent&&b.textContent.trim()===' + light + ')?.click()')
            until("document.documentElement.dataset.theme==='light'")
            click('workbench.home')
            capture('home')
            click('workbench.user_guide')
            until("!!document.querySelector('.manual-document')")
            until("document.querySelectorAll('button[data-manual-chapter]').length===21")
            # Desktop intentionally omits only the Web introduction chapter.
            assert js("document.querySelector('.manual-language-toolbar select').value") == args.locale
            assert js("document.querySelector('.manual-language-toolbar').innerText.includes('author-preview-')")
            js("document.querySelector('[data-manual-chapter=getting-started]').click()")
            until("document.querySelector('[data-manual-chapter=getting-started]').getAttribute('aria-current')==='page'")
            first = js("document.querySelector('.manual-document').innerText.slice(0,500)")
            assert '下载与安装' not in first and '快速开始' not in first
            capture('manual')
            report['checks'] = ['Real application language control and completed locale font phase',
                                'Home with actual localized module names', '21 desktop chapters from the full 22-chapter translated reader; Web introduction omitted by the real host adapter',
                                'Author-preview ZIP installed through native import protocol in owned scratch; actual HelpPage native list/readJson and shared Chinese asset resolution',
                                'Reading language independent of UI, explicit author-preview version displayed; not a final delivered pack']
            report['success'] = True
        finally:
            report['sourceChangesDuringRun'] = [item['path'] for item in snapshots.values()
                                                if hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() != item['sha256']]
            (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            window.closing = True
            window.close()
            app.processEvents()
            sip.delete(window)
            app.processEvents()
            sip.delete(app)
            print(json.dumps({'success': report['success'], 'report': str(out / 'report.json')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--locale', choices=('en', 'ja'), required=True)
    parser.add_argument('--frontend-dist', type=Path, default=ROOT / 'frontend/dist')
    main(parser.parse_args())
