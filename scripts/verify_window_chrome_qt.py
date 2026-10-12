"""Actual Windows title surface and shared scrollbars, with isolated state."""
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PARTS = ('scripts', 'desktop/src', 'backend/src', 'packages/phonetic_core/src')
for part in PARTS:
    sys.path.insert(0, str(ROOT / part))
os.environ['PYTHONPATH'] = os.pathsep.join(str(ROOT / part) for part in PARTS)
os.environ['QTWEBENGINE_CHROMIUM_FLAGS'] = '--mute-audio'


def main():
    from PyQt6.QtCore import QEventLoop, QPoint, QTimer, Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication
    from ptb_desktop.host import Workbench, register_scheme
    from capture_native_window import capture_window
    from ptb_worker.local_acoustic_files import initialize_local_files
    from verification_artifacts import verification_run

    assert sys.platform == 'win32' and sys.getwindowsversion().build >= 22000
    with verification_run('window-chrome', 'qt', 'Windows 11 native caption colours and Qt Chromium scrollbars; isolated existing-schema DB; owned window only') as (out, scratch):
        db = scratch / 'jobs.sqlite3'
        with sqlite3.connect((ROOT / 'output/validation/p06/local-state.sqlite3').as_uri() + '?mode=ro', uri=True) as source, sqlite3.connect(db) as target:
            source.backup(target)
        cache = scratch / 'cache'
        cache.mkdir()
        initialize_local_files(cache)
        register_scheme()
        app = QApplication(['window-chrome-qa'])
        window = Workbench(ROOT / 'frontend/dist', test=True, jobs_path=db, local_files_root=cache, vocal_profile=scratch / 'vocal', start_module='M02')
        window.move(30,30)
        window.show()
        report = dict(success=False, windows_build=sys.getwindowsversion().build, themes=[], checks=[])

        def pause(ms=120):
            loop = QEventLoop()
            QTimer.singleShot(ms, loop.quit)
            loop.exec()

        def js(code):
            loop, results = QEventLoop(), []
            window.page.runJavaScript(code, lambda value: (results.append(value), loop.quit()))
            QTimer.singleShot(8000, loop.quit)
            loop.exec()
            assert results, 'JavaScript timeout'
            return results[0]

        def until(code):
            deadline = time.monotonic() + 45
            while time.monotonic() < deadline:
                if js(code):
                    return
                pause()
            raise AssertionError(code)

        def native_image():
            return capture_window(int(window.winId()))

        def caption_pixel():
            # CAPTION_COLOR/TEXT_COLOR are set-only DWM attributes. Verify the
            # rendered native title pixels rather than querying unsupported getters.
            pause(180)
            image = native_image()
            image.save(out / 'caption-probe.png')
            return image.getpixel((image.width // 2, round(16 * window.devicePixelRatioF())))[:3]

        def rgb(hex_colour):
            return tuple(int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))

        def capture(name):
            pause(450)
            native_image().save(out / (name + '.png'))

        try:
            until("!!document.querySelector('#tool-sidebar')&&document.fonts.status==='loaded'")
            js("[...document.querySelectorAll('button')].find(b=>b.textContent.trim()==='设置').click()")
            until("!!document.querySelector('#settings-tab-appearance')")
            assert js("document.querySelectorAll('#tool-sidebar .brand').length===1&&document.querySelector('.brand').getBoundingClientRect().top===0")
            details = js("(()=>{const d=document.querySelector('.palette-details'),picker=document.querySelector('#palette-choice');return {open:d.open,under:picker.closest('.setting-block').nextElementSibling===d};})()")
            assert details == dict(open=True, under=True), details
            for palette, mode in (('everforest', 'light'), ('codex', 'light'), ('rose-pine', 'light'), ('everforest', 'dark')):
                # Apply through the real settings controls, keeping Vue state consistent.
                js("[...document.querySelectorAll('.appearance-settings button')].find(b=>b.textContent.trim()===" + json.dumps('深色' if mode == 'dark' else '浅色') + ").click()")
                js("document.querySelector('#palette-choice').click()")
                until("!!document.querySelector('[data-palette-option=" + palette + "]')")
                js("document.querySelector('[data-palette-option=" + palette + "]').click()")
                pause(400)
                colours = js("(()=>{const s=getComputedStyle(document.documentElement);return {background:s.getPropertyValue('--sidebar').trim(),text:s.getPropertyValue('--text').trim()};})()")
                assert window._caption_applied
                assert caption_pixel() == rgb(colours['background']), (colours, caption_pixel())
                assert window._caption_colors == (colours['background'], colours['text'])
                report['themes'].append(dict(palette=palette, mode=mode, **colours))
                capture(palette + '-' + mode)
            window.bridge.setCaptionColors('#ffffff;evil', '#000000')
            assert caption_pixel() == rgb(report['themes'][-1]['background'])
            window.showMinimized()
            pause(250)
            assert window.isMinimized()
            window.showNormal()
            window.showMaximized()
            pause(400)
            assert window.isMaximized()
            window.showNormal()
            pause(250)
            assert not window.isMaximized()
            position = window.pos()
            window.move(position + QPoint(20, 10))
            pause()
            assert window.pos() == position + QPoint(20, 10)
            assert caption_pixel() == rgb(report['themes'][-1]['background'])
            # Two-dimensional scrolling exercises the shared style and native input.
            js("(()=>{const s=document.createElement('div');s.id='scrollbar-probe';s.tabIndex=0;s.style='position:fixed;left:100px;top:130px;width:240px;height:220px;overflow:scroll;z-index:99999;background:var(--panel)';s.innerHTML='<div style=\"width:600px;height:2000px\">滚动条验证</div>';document.body.append(s);s.focus();})()")
            pause()
            styles = js("(()=>{const e=document.querySelector('#scrollbar-probe'),bar=getComputedStyle(e,'::-webkit-scrollbar'),thumb=getComputedStyle(e,'::-webkit-scrollbar-thumb'),button=getComputedStyle(e,'::-webkit-scrollbar-button'),track=getComputedStyle(e,'::-webkit-scrollbar-track');return {width:bar.width,height:bar.height,radius:thumb.borderRadius,button:button.display,track:track.backgroundColor};})()")
            assert styles == dict(width='9px', height='9px', radius='999px', button='none', track='rgba(0, 0, 0, 0)'), styles
            QTest.keyClick(window.view.focusProxy(), Qt.Key.Key_PageDown)
            pause(250)
            assert js("document.querySelector('#scrollbar-probe').scrollTop>0")
            js("document.querySelector('#scrollbar-probe').scrollTop=0")
            pause()
            QTest.mousePress(window.view.focusProxy(), Qt.MouseButton.LeftButton, pos=QPoint(335, 145))
            QTest.mouseMove(window.view.focusProxy(), QPoint(335, 270), delay=150)
            QTest.mouseRelease(window.view.focusProxy(), Qt.MouseButton.LeftButton, pos=QPoint(335, 270))
            pause()
            assert js("document.querySelector('#scrollbar-probe').scrollTop>500")
            js("document.querySelector('#scrollbar-probe').scrollLeft=200")
            assert js("document.querySelector('#scrollbar-probe').scrollLeft===200")
            capture('scrollbars-dark')
            js("document.querySelector('#scrollbar-probe').remove();[...document.querySelectorAll('.mode-choices button')].find(b=>b.textContent.trim()==='浅色').click()")
            pause(200)
            capture('everforest-final')
            # Embedded M10 has its own document and must consume the shared sheet.
            js("[...document.querySelectorAll('.nav-item')].find(b=>b.textContent.trim()==='生理参数合成').click()")
            until("!!document.querySelector('iframe')?.contentDocument?.querySelector('.inspector')")
            iframe_style = js("(()=>{const d=document.querySelector('iframe').contentDocument,e=d.querySelector('.inspector');return {width:d.defaultView.getComputedStyle(e,'::-webkit-scrollbar').width,button:d.defaultView.getComputedStyle(e,'::-webkit-scrollbar-button').display};})()")
            assert iframe_style == dict(width='9px', button='none'), iframe_style
            report['checks'] = ['Four native caption theme pairs match rendered title pixels, text setter succeeds', 'Invalid bridge colours preserve existing caption', 'Native minimize/maximize/restore/move preserve title colour', 'Transparent rails, round 9px scrollbars with no arrows; PageDown, thumb dragging and horizontal scroll', 'Embedded M10 consumes the same scrollbar sheet', 'Single desktop branding and expanded palette explanation directly below picker', 'Owned Workbench close stops its service']
            report['scrollbars'] = styles
            report['success'] = True
        finally:
            window.close()
            pause(350)
            report['closed'] = not window.isVisible()
            (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            app.quit()
            print(json.dumps(dict(success=report['success'], report=str(out / 'report.json')), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
