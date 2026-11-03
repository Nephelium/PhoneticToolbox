"""Capture current settings in an isolated, locale-controlled Windows Qt host."""
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
from manual.capture_locale import CaptureLocale


def main(args):
    configure()
    os.environ['QT_QPA_PLATFORM'] = 'windows'
    os.environ['QTWEBENGINE_CHROMIUM_FLAGS'] = '--mute-audio'
    for key in ('QT_ENABLE_HIGHDPI_SCALING', 'QT_SCALE_FACTOR', 'QT_SCREEN_SCALE_FACTORS'):
        os.environ.pop(key, None)
    from PyQt6.QtCore import QEventLoop, QTimer, Qt
    from PyQt6 import sip
    from PyQt6.QtWidgets import QApplication
    from ptb_desktop.host import Workbench, register_scheme
    from ptb_worker.local_workspace import prepare_workspace

    ui = CaptureLocale(args.locale, ('workbench', 'components', 'common'))
    sources = [Path(__file__), ROOT / 'scripts/manual/capture_locale.py', args.frontend_dist / 'index.html',
               ROOT / 'desktop/src/ptb_desktop/host.py', ROOT / 'desktop/src/ptb_desktop/presentation.py',
               ROOT / 'frontend/src/app/AppShell.vue']
    sources += [ROOT / 'frontend/src/components' / name for name in
                ('AppearanceSettings.vue', 'IconSettings.vue', 'IconCropDialog.vue', 'FontSettings.vue', 'StorageSettings.vue')]
    sources += [ROOT / 'frontend/src/i18n/locales' / locale / namespace for locale in ('zh-CN', args.locale)
                for namespace in ('components.json', 'workbench.json')]
    snapshots = {str(i): {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                 for i, path in enumerate(dict.fromkeys(sources))}
    register_scheme()
    app = QApplication(['owned-localized-settings-capture'])
    with verification_run('settings-localized', args.locale, 'Actual current Qt; isolated settings/database; four referenced scenes; generated red/white crop fixture; native DPR1.5, maximized, page100, light') as (out, scratch):
        db, cache = prepare_workspace(scratch / 'state', ROOT / 'backend/migrations')
        window = Workbench(args.frontend_dist.resolve(), test=True, jobs_path=db,
                           local_files_root=cache, vocal_profile=scratch / 'vocal')
        window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
        window.resize(window.screen().geometry().width(), window.screen().geometry().height())
        window.showMaximized()
        window.view.setZoomFactor(1)
        window.page.setAudioMuted(True)
        report = dict(success=False, locale=args.locale, sourceChapter='settings', captures=[], sources=snapshots,
                      ownedPid=os.getpid(), physicalCompositorVerified=False, checks=[])

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
            end = time.monotonic() + 45
            while time.monotonic() < end:
                if js(code):
                    return
                pause()
            raise AssertionError(code + '\n' + str(js('document.body.innerText.slice(-1600)')))

        def click_key(key):
            label = json.dumps(ui.key(key), ensure_ascii=False)
            expression = '[...document.querySelectorAll("button")].find(b=>b.offsetParent&&(b.getAttribute("aria-label")===' + label + '||b.textContent.trim()===' + label + '))'
            until('!!(' + expression + ')&&!(' + expression + ').disabled')
            js(expression + '.click()')
            pause(180)

        def tab(name):
            js('document.querySelector("#settings-tab-' + name + '").click()')
            until('!!document.querySelector("#settings-panel-' + name + '")?.offsetParent')
            pause(200)

        def capture(name,regions=None):
            until("document.fonts.status==='loaded'")
            pause(450)
            metrics = ui.metrics(js)
            image = window.view.grab().toImage()
            raw = [image.width(), image.height()]
            assert image.devicePixelRatio() == 1.5 and raw in ([2560, 1600], [2561, 1601]), raw
            if raw != [2560, 1600]:
                image = image.copy(0, 0, 2560, 1600)
            path = out / (name + '.png')
            assert image.save(str(path))
            boxes={}
            for key,selector in (regions or {}).items():
                box=js('(()=>{const r=document.querySelector('+json.dumps(selector)+')?.getBoundingClientRect();return r?[r.x,r.y,r.width,r.height]:null})()')
                assert box and box[2]>0 and box[3]>0
                boxes[key]={'selector':selector,'box':box,'units':'css-px'}
            report['captures'].append(dict(regions=boxes,name=name, file=path.name, image=[2560, 1600], raw_image=raw,
                                           maximized=window.isMaximized(), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                           **{key: metrics[key] for key in ('locale', 'fontLocale', 'fontFamily', 'dpr', 'pageZoom', 'theme')}))
            print('Captured ' + name, flush=True)

        try:
            until("!!document.querySelector('#tool-sidebar')&&document.fonts.status==='loaded'")
            ui.prepare_qt(js, until)
            click_key('workbench.settings')
            tab('appearance')
            # Theme values are stable product identities, independent of labels.
            js("document.querySelector('[data-theme-mode=light]')?.click()")
            if js('document.documentElement.dataset.theme') != 'light':
                label = ui.label('浅色')
                js('[...document.querySelectorAll("button")].find(b=>b.offsetParent&&b.textContent.trim()===' + json.dumps(label) + ')?.click()')
            until("document.documentElement.dataset.theme==='light'")
            if js("document.querySelector('#tool-sidebar').offsetWidth>100"):
                click_key('workbench.collapse_sidebar')
            if 'theme-and-zoom' in (args.scene or []):
                capture('theme-and-zoom-parent',{'themes':'.theme-mode-setting','zoom':'.settings-column:nth-child(2)>.setting-block:first-child'})
            if not args.scene or 'mode-cards' in args.scene:
                capture('mode-cards')
            if not args.scene or 'fonts' in args.scene:
                tab('fonts')
                capture('fonts')
            if not args.scene or 'storage' in args.scene:
                tab('storage')
                until("!!document.querySelector('.storage-settings dl')")
                # A fresh isolated context must never expose personal caches.
                assert not js("document.querySelector('.storage-settings').innerText.includes('浏览器图形缓存')")
                capture('storage')
            if not args.scene or 'crop-white-key' in args.scene:
                tab('icons')
                js("(()=>{const c=document.createElement('canvas');c.width=400;c.height=240;const x=c.getContext('2d');x.fillStyle='#fff';x.fillRect(0,0,400,240);x.fillStyle='#dc281e';x.fillRect(140,70,120,100);x.fillStyle='#fff';x.fillRect(190,105,20,20);c.toBlob(blob=>{const d=new DataTransfer();d.items.add(new File([blob],'manual-icon-example.png',{type:'image/png'}));const e=document.querySelector('.icon-file-input');e.files=d.files;e.dispatchEvent(new Event('change',{bubbles:true}));});})()")
                until("!!document.querySelector('.crop-selection')")
                js("(()=>{const e=document.querySelector('.crop-remove-toggle input');e.checked=true;e.dispatchEvent(new Event('change',{bubbles:true}));})()")
                until("!!document.querySelector('.crop-background-controls')")
                capture('crop-white-key')
                click_key('components.cancel')
                until("!document.querySelector('.icon-crop-dialog')")
            report['checks'] = ['Selected real settings scenes with six current category tabs', 'Live native storage status when requested; no cleanup invoked',
                                'When requested, actual File/Canvas icon import and background removal preview; cancelled without persisting icon',
                                'Actual UI locale and completed Japanese font phase; no user preferences or data modified']
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
    parser.add_argument('--scene', action='append', choices=('mode-cards', 'fonts', 'storage', 'crop-white-key','theme-and-zoom'), help='Capture only named scenes; repeat to select more')
    main(parser.parse_args())
