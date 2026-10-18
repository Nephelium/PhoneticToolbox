"""Verify current M05 image bytes and actual Qt reader load/zoom in both themes."""
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
os.environ['QTWEBENGINE_CHROMIUM_FLAGS'] = '--disable-gpu --mute-audio'


def main():
    configure()
    from PyQt6.QtCore import QEvent, QEventLoop, QTimer, Qt
    from PyQt6.QtWidgets import QApplication
    from ptb_desktop.host import Workbench, register_scheme
    from ptb_worker.local_workspace import prepare_workspace

    project = json.loads((ROOT / 'manual/project.json').read_text('utf8'))
    chapter = json.loads((ROOT / 'manual/chapters/m05.json').read_text('utf8'))
    images = [n for n in chapter['body']['content'] if n['type'] == 'image']
    assert len(images) == 9
    assets = {a['id']: a for a in project['assets']}
    expected = []
    for node in images:
        asset = assets[node['attrs']['assetId']]
        assert asset['git'] is False and asset['distribution'] == 'software-only'
        assert 'm05-current-20261018/' in asset['path']
        hashes = [hashlib.sha256((ROOT / folder / asset['path']).read_bytes()).hexdigest() for folder in ['manual', 'frontend/public/manual', 'frontend/dist/manual']]
        assert hashes == [asset['sha256']] * 3
        expected.append(dict(id=asset['id'], width=asset['width'], height=asset['height']))
    for folder in ['frontend/public/manual', 'frontend/dist/manual']:
        assert json.loads((ROOT / folder / 'chapters/m05.json').read_text('utf8')) == chapter
    register_scheme()
    app = QApplication(['PTB-M05-manual-reader'])
    with verification_run('m05-manual', 'reader-20261018', 'Current nine image hashes across source/public/dist, actual isolated Qt reader load and image zoom; light/dark; no devices') as (out, scratch):
        print(out, flush=True)
        db, cache = prepare_workspace(scratch / 'state', ROOT / 'backend/migrations')
        window = Workbench(ROOT / 'frontend/dist', test=True, jobs_path=db, local_files_root=cache, vocal_profile=scratch / 'vocal', start_module='M05')
        window.resize(window.screen().geometry().width(), window.screen().geometry().height())
        window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
        window.showMaximized()
        window.view.setZoomFactor(1)
        report = dict(success=False, byteChecks=expected, themes=[], errors=[])

        def pause(ms=100):
            loop = QEventLoop()
            QTimer.singleShot(ms, loop.quit)
            loop.exec()

        def js(code):
            wrapped = '(()=>{try{return {ok:true,value:eval(' + json.dumps(code) + ')??null};}catch(e){return {ok:false,error:String(e)};}})()'
            loop, values = QEventLoop(), []
            window.page.runJavaScript(wrapped, lambda value: (values.append(value), loop.quit()))
            QTimer.singleShot(6000, loop.quit)
            loop.exec()
            if not values or not values[0] or not values[0].get('ok'):
                raise RuntimeError(str(values))
            return values[0].get('value')

        def until(code):
            deadline = time.monotonic() + 45
            while time.monotonic() < deadline:
                if js(code):
                    return
                pause()
            raise AssertionError(code + '\n' + str(js('document.body.innerText.slice(-2000)')))

        def click(label):
            text = json.dumps(label)
            expression = '[...document.querySelectorAll("button")].find(e=>e.offsetParent&&(e.getAttribute("aria-label")===' + text + '||e.getAttribute("title")===' + text + '||e.textContent.trim()===' + text + '))'
            until('!!(' + expression + ')&&!(' + expression + ').disabled')
            js(expression + '.click()')
            pause(120)

        try:
            until('!!document.querySelector(".lip-page")')
            for theme, label in [('light', '浅色'), ('dark', '深色')]:
                click('设置')
                click(label)
                click('使用说明')
                until('!!document.querySelector("[data-manual-chapter=m05]")')
                js('document.querySelector("[data-manual-chapter=m05]").click()')
                until('document.querySelector(".manual-chapter-header h1")?.textContent.includes("唇形提取")')
                until('document.fonts.status==="loaded"&&!document.documentElement.hasAttribute("data-theme-switching")')
                assert js('document.documentElement.dataset.theme') == theme
                assert js('document.querySelectorAll(".manual-media-image").length') == 9
                for asset in expected:
                    selector = '[data-asset-id="' + asset['id'] + '"]'
                    expression = 'document.querySelector(' + json.dumps(selector) + ')'
                    js(expression + '.scrollIntoView({block:"center"})')
                    until('(()=>{const i=' + expression + '.querySelector("img");return i?.complete&&i.naturalWidth===' + str(asset['width']) + '&&i.naturalHeight===' + str(asset['height']) + ';})()')
                    js('window.manualImageDecoded=false;' + expression + '.querySelector("img").decode().then(()=>window.manualImageDecoded=true)')
                    until('window.manualImageDecoded')
                    js(expression + '.querySelector("button").click()')
                    until('(()=>{const a=' + expression + '.querySelector("img"),b=document.querySelector(".manual-image-overlay img");return b?.complete&&b.currentSrc===a.currentSrc&&b.naturalWidth===' + str(asset['width']) + ';})()')
                    js('window.manualZoomDecoded=false;document.querySelector(".manual-image-overlay img").decode().then(()=>requestAnimationFrame(()=>requestAnimationFrame(()=>window.manualZoomDecoded=true)))')
                    until('window.manualZoomDecoded')
                    assert js('document.documentElement.scrollWidth<=innerWidth+1')
                    if 'offline-replay' in asset['id']:
                        pause(300)
                        window.view.repaint()
                        window.view.grab()
                        pause(250)
                        window.view.grab().save(str(out / ('reader-replay-' + theme + '.png')))
                    click('关闭图片')
                    until('!document.querySelector(".manual-image-overlay")')
                assert not js('!!document.querySelector(".manual-media-error")')
                report['themes'].append(dict(theme=theme, loaded=9, zoomed=9, imageErrors=0, overflow=False))
            report['success'] = True
        except Exception as error:
            report['errors'].append(repr(error))
            window.view.grab().save(str(out / 'failed.png'))
            raise
        finally:
            (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', 'utf8')
            window.accept_close()
            pause(150)
            window.page.deleteLater()
            app.processEvents()
            window.profile.deleteLater()
            app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            app.processEvents()
        print(json.dumps(dict(success=report['success'], images=9, themes=2, out=str(out))), flush=True)


if __name__ == '__main__':
    main()
