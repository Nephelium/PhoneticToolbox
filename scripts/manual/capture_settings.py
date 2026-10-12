"""Capture settings navigation from the current Windows source host."""
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
for part in ('scripts', 'desktop/src', 'backend/src', 'packages/phonetic_core/src'):
    sys.path.insert(0, str(ROOT / part))
os.environ['PYTHONPATH'] = os.pathsep.join(str(ROOT / part) for part in ('scripts', 'desktop/src', 'backend/src', 'packages/phonetic_core/src'))
for name in ('QT_ENABLE_HIGHDPI_SCALING', 'QT_SCALE_FACTOR', 'QT_SCREEN_SCALE_FACTORS'):
    os.environ.pop(name, None)
os.environ['QT_QPA_PLATFORM'] = 'windows'
os.environ['QTWEBENGINE_CHROMIUM_FLAGS'] = '--mute-audio'


def main():
    from verification_artifacts import verification_run
    from PyQt6.QtCore import Qt, QEventLoop, QTimer
    from PyQt6.QtWidgets import QApplication
    from ptb_desktop.host import Workbench, register_scheme
    from ptb_worker.local_acoustic_files import initialize_local_files

    with verification_run('settings-navigation', 'manual-qt', 'Current source host; isolated existing-schema database copy; native light DPR1.5 page100 maximized; no cache deletion') as (out, scratch):
        db = scratch / 'jobs.sqlite3'
        with sqlite3.connect((ROOT / 'output/validation/p06/local-state.sqlite3').as_uri() + '?mode=ro', uri=True) as template, sqlite3.connect(db) as target:
            template.backup(target)
        cache = scratch / 'cache'
        cache.mkdir()
        initialize_local_files(cache)
        register_scheme()
        app = QApplication(['settings-navigation-owned-manual'])
        window = Workbench(ROOT / 'frontend/dist', test=True, jobs_path=db, local_files_root=cache, vocal_profile=scratch / 'vocal')
        window.resize(window.screen().geometry().width(), window.screen().geometry().height())
        window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
        window.showMaximized()
        window.view.setZoomFactor(1.)
        window.page.setAudioMuted(True)
        report = dict(success=False, captures=[], checks=[])

        def pause(ms=100):
            loop = QEventLoop()
            QTimer.singleShot(ms, loop.quit)
            loop.exec()

        def js(code):
            loop, values = QEventLoop(), []
            window.page.runJavaScript(code, lambda value: (values.append(value), loop.quit()))
            QTimer.singleShot(8000, loop.quit)
            loop.exec()
            if not values:
                raise RuntimeError('JavaScript timeout')
            return values[0]

        def until(code):
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if js(code):
                    return
                pause()
            raise AssertionError(code)

        def click(label):
            expression = '[...document.querySelectorAll("button")].find(b=>b.offsetParent&&(b.getAttribute("aria-label")||b.textContent.trim())===' + json.dumps(label) + ')'
            until('!!(' + expression + ')&&!(' + expression + ').disabled')
            js(expression + '.click()')
            pause(180)

        def capture(name):
            until("document.fonts.status==='loaded'")
            pause(450)
            image = window.view.grab().toImage()
            raw = [image.width(), image.height()]
            dpr = image.devicePixelRatio()
            assert abs(dpr - 1.5) < .01 and raw in ([2560, 1600], [2561, 1601]), (raw, dpr)
            if raw != [2560, 1600]:
                image = image.copy(0, 0, 2560, 1600)
            path = out / (name + '.png')
            assert image.save(str(path))
            report['captures'].append(dict(name=name, file=str(path), image=[2560, 1600], raw_image=raw, dpr=dpr, pageZoom=window.view.zoomFactor(), maximized=window.isMaximized(), theme=js('document.documentElement.dataset.theme'), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            print('Captured ' + name, flush=True)

        try:
            until("!!document.querySelector('#tool-sidebar')&&document.fonts.status==='loaded'")
            assert js("!!document.querySelector('[aria-label=还原窗口]')")
            click('设置');click('浅色');click('首页')
            capture('home')
            click('使用说明')
            until("!!document.querySelector('.manual-document')")
            js("document.querySelector('[data-manual-chapter=getting-started]').click()")
            until("!!document.querySelector('[data-manual-section]')")
            js("document.querySelector('[data-manual-section]').click()")
            capture('manual')
            click('设置')
            click('浅色')
            click('收起侧栏')
            assert js("document.documentElement.dataset.palette==='everforest'")
            click('图标')
            until("document.querySelectorAll('.builtin-icon-options .icon-choice').length===31")
            assert window.icon_preferences.read()['selected'] == 'default'
            js("document.querySelector('[data-icon-id=builtin-033]').click()")
            until("document.querySelector('[data-icon-id=builtin-033]').getAttribute('aria-pressed')==='true'")
            assert window.icon_preferences.read()['selected'] == 'builtin-033'
            js("(()=>{const c=document.createElement('canvas');c.width=200;c.height=100;c.getContext('2d').fillRect(0,0,200,100);fetch(c.toDataURL('image/png')).then(r=>r.blob()).then(blob=>{const dt=new DataTransfer();dt.items.add(new File([blob],'测试图标.png',{type:'image/png'}));const input=document.querySelector('.icon-file-input');input.files=dt.files;input.dispatchEvent(new Event('change',{bubbles:true}));});})()")
            until("document.querySelectorAll('.custom-icon-item').length===1&&!document.querySelector('.icon-settings button:disabled')")
            state = window.icon_preferences.read()
            assert state['selected'].startswith('custom-')
            from ptb_desktop.icon_preferences import IconPreferences
            assert IconPreferences(window.icon_preferences.path.parent, window.windowIcon(), dist=ROOT / 'frontend/dist').read() == state
            assert not window.windowIcon().isNull()
            click('移除图标 测试图标.png')
            until("document.querySelector('[data-icon-id=default]').getAttribute('aria-pressed')==='true'")
            assert window.icon_preferences.read()['selected'] == 'default'
            for name, label in [('appearance', '外观'), ('fonts', '字体'), ('icons', '图标'), ('storage', '本机存储与自动清理')]:
                click(label)
                until("[...document.querySelectorAll('.settings-panel')].filter(e=>e.offsetParent).length===1")
                if name == 'storage':
                    until("!!document.querySelector('.storage-settings dl')")
                capture(name)
            js("document.querySelector('#tool-sidebar button[title=\"LPC 谱图\"]').click()")
            until("!!document.querySelector('.module-frame')")
            assert js("!([...document.querySelectorAll('main *')].filter(e=>{const r=e.getBoundingClientRect();return r.width>120&&r.height>60&&getComputedStyle(e).backgroundColor==='rgb(255, 255, 255)'}).length)")
            capture('lpc')
            js("document.querySelector('#tool-sidebar button[title=\"参数显示\"]').click()")
            until("!!document.querySelector('.m02-page .workbench-left')")
            layers = js("(()=>{const color=s=>getComputedStyle(document.querySelector(s)).backgroundColor;return {navigation:color('#tool-sidebar'),tabs:color('.topbar'),rail:color('.m02-page .workbench-left'),card:color('.m02-page .workbench-card'),center:color('main')}})()")
            assert layers['navigation'] == layers['tabs'] and layers['rail'] != layers['card'] and layers['navigation'] != layers['center']
            report['layeredColours'] = layers
            capture('parameter-display')
            click('设置')
            js("(()=>{const e=document.querySelector('.storage-fields input');e.value='45';e.dispatchEvent(new Event('input',{bubbles:true}));})()")
            click('字体')
            js("(()=>{const e=document.querySelector('input[aria-label=正文基础字号]');e.value='18';e.dispatchEvent(new Event('input',{bubbles:true}));})()")
            click('外观')
            click('字体')
            assert js("document.querySelector('input[aria-label=正文基础字号]').value==='18'")
            click('本机存储与自动清理')
            assert js("document.querySelector('.storage-fields input').value==='45'")
            report['checks'] = ['Four mutually exclusive source Qt settings pages', 'Native storage status from isolated database without cleaning', 'Font and retention drafts retained across navigation', 'Default Everforest and edited W1; actual Qt bridge builtin selection, File/Canvas import, preference reopening and selected removal restoring default', 'Four settings, LPC and parameter display light maximized DPR1.5 page100 screenshots; actual native LPC interface has no large hardcoded white surfaces; M02 chrome/rail/card colours are subtly distinct']
            report['success'] = True
        finally:
            (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            # Discard only this isolated instance's unapplied font edit before exit.
            js("document.querySelector('#settings-tab-fonts')?.click()")
            pause()
            js("[...document.querySelectorAll('.font-settings button')].find(b=>b.textContent.trim()==='取消字体编辑')?.click()")
            pause()
            window.close()
            pause(500)
            app.quit()
            print(json.dumps({'success': report['success'], 'report': str(out / 'report.json')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
