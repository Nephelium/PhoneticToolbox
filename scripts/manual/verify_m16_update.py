"""Check current recording manual and a missing lazy-page resource in owned Qt."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from workbench_source import configure
from verification_artifacts import verification_run

os.environ['QT_QPA_PLATFORM'] = 'windows'
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS', '--mute-audio')


def main():
    configure()
    from PyQt6.QtCore import QEvent, QEventLoop, QTimer, Qt
    from PyQt6.QtWidgets import QApplication
    from ptb_desktop.host import Workbench, register_scheme
    from ptb_desktop.web_assets import LocalOnly

    project = json.loads((ROOT / 'manual/project.json').read_text('utf8'))
    chapter = json.loads((ROOT / 'manual/chapters/m16.json').read_text('utf8'))
    expected = [a for a in project['assets'] if 'software-only/m16-20261018/' in a['path']]
    assert len(expected) == 11
    for asset in expected:
        hashes = [hashlib.sha256((ROOT / folder / asset['path']).read_bytes()).hexdigest()
                  for folder in ['manual', 'frontend/public/manual', 'frontend/dist/manual']]
        assert hashes == [asset['sha256']] * 3, asset['id']
    for folder in ['frontend/public/manual', 'frontend/dist/manual']:
        assert json.loads((ROOT / folder / 'chapters/m16.json').read_text('utf8')) == chapter
    register_scheme()
    app = QApplication(['PTB owned manual and lazy-page QA'])
    with verification_run('m16', 'manual-reader', 'Actual Windows Qt; 11 source/public/dist image hashes, light/dark reader, blocked lazy CSS and normal M05; no devices or user project') as (out, scratch):
        report = dict(success=False, blocked=[], themes=[], byteChecks=[a['id'] for a in expected])
        window = None

        class BlockLipCss(LocalOnly):
            def interceptRequest(self, info):
                name = Path(info.requestUrl().path()).name
                if name.startswith('LipExtractionPage-') and name.endswith('.css'):
                    report['blocked'].append(name)
                    info.block(True)
                else:
                    super().interceptRequest(info)

        def pause(ms=100):
            loop = QEventLoop()
            QTimer.singleShot(ms, loop.quit)
            loop.exec()

        def js(code):
            loop, values = QEventLoop(), []
            window.page.runJavaScript(code, lambda value: (values.append(value), loop.quit()))
            QTimer.singleShot(6000, loop.quit)
            loop.exec()
            assert values, 'JavaScript callback timeout: ' + code
            return values[0]

        def until(code):
            deadline = time.monotonic() + 35
            while time.monotonic() < deadline:
                if js(code):
                    return
                pause()
            raise AssertionError(code + '\n' + str(js('document.body.innerText.slice(-2000)')))

        def click(label):
            expression = '[...document.querySelectorAll("button")].find(e=>e.offsetParent&&e.textContent.trim()===' + json.dumps(label) + ')'
            until('!!(' + expression + ')&&!(' + expression + ').disabled')
            js(expression + '.click()')
            pause(120)

        def open_window(name, block=False):
            nonlocal window
            window = Workbench(ROOT / 'frontend/dist', test=True, vocal_profile=scratch / name, start_module='M05')
            if block:
                window.interceptor = BlockLipCss(window.service.url, window.profile)
                window.profile.setUrlRequestInterceptor(window.interceptor)
            window.resize(1500, 1000)
            window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
            window.show()
            window.page.setAudioMuted(True)

        def close_window():
            nonlocal window
            if window is None:
                return
            window.closing = True
            window.close()
            window.page.deleteLater()
            pause(120)
            window.profile.deleteLater()
            app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            app.processEvents()
            window.deleteLater()
            pause(150)
            window = None

        try:
            open_window('blocked', block=True)
            until('document.querySelector(".module-status-error")?.textContent.includes("唇形提取页面载入失败，请重启工作台后再打开。")')
            assert report['blocked']
            report['failureHint'] = js('document.querySelector(".module-status-error").textContent')
            window.view.grab().save(str(out / 'lazy-css-failure.png'))
            close_window()
            open_window('normal')
            until('!!document.querySelector(".module-workbench .workbench-left[aria-label=设备与采集]")')
            report['normalM05'] = True
            click('使用说明')
            until('!!document.querySelector(".manual-toc [data-manual-chapter=m16]")')
            js('document.querySelector(".manual-toc [data-manual-chapter=m16]").click()')
            until('document.querySelector(".manual-document")?.dataset.manualChapter==="m16"')
            for phrase in ['下载 XLSX 示例', 'Ctrl', '谱减', '全部历史版本']:
                assert js('document.querySelector(".manual-document").innerText.includes(' + json.dumps(phrase) + ')'), phrase
            assert js('(()=>{const ids=[...document.querySelectorAll(".manual-document [id]")].map(e=>e.id);return new Set(ids).size===ids.length})()')
            for theme in ['light', 'dark']:
                js('document.documentElement.dataset.theme=' + json.dumps(theme))
                for asset in expected:
                    expression = 'document.querySelector(' + json.dumps('[data-asset-id="' + asset['id'] + '"]') + ')'
                    js(expression + '.scrollIntoView({block:"center"})')
                    until('(()=>{const i=' + expression + '.querySelector("img");return i?.complete&&i.naturalWidth===' + str(asset['width']) + '&&i.naturalHeight===' + str(asset['height']) + '})()')
                    js(expression + '.querySelector("button").click()')
                    until('document.querySelector(".manual-image-overlay img")?.complete')
                    assert js('document.querySelector(".manual-image-overlay img").naturalWidth') == asset['width']
                    click('关闭图片')
                    assert not js('!!document.querySelector(".manual-image-overlay")')
                assert not js('!!document.querySelector(".manual-media-error")')
                assert js('document.documentElement.scrollWidth<=innerWidth+1')
                js('document.getElementById("manual-m16--m16-processing").scrollIntoView({block:"start"})')
                pause(300)
                window.view.grab().save(str(out / ('reader-processing-' + theme + '.png')))
                report['themes'].append(dict(theme=theme, loaded=11, zoomed=11, errors=0))
            audio = js('[...document.querySelectorAll(".manual-document audio")].map(e=>({src:e.currentSrc||e.src,preload:e.preload}))')
            assert len(audio) == 2
            js('document.querySelectorAll(".manual-document audio").forEach(e=>e.load())')
            until('[...document.querySelectorAll(".manual-document audio")].every(e=>e.readyState>=1&&e.duration>0&&!e.error)')
            report['audioMetadata'] = js('[...document.querySelectorAll(".manual-document audio")].map(e=>({duration:e.duration,paused:e.paused,error:e.error}))')
            assert all(a['paused'] and not a['error'] for a in report['audioMetadata'])
            report['success'] = True
        except Exception as error:
            report['error'] = repr(error)
            if window:
                window.view.grab().save(str(out / 'failed.png'))
            raise
        finally:
            close_window()
            (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', 'utf8')
            print(json.dumps(dict(out=str(out), success=report['success'], error=report.get('error')), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
