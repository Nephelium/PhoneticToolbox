"""Check original PDF colours in an owned, visible Qt source workbench."""
import hashlib
from io import BytesIO
import json
import threading
import time
from pathlib import Path

from workbench_source import configure
from verification_artifacts import verification_run, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    configure(ROOT)
    from PyQt6.QtCore import QCoreApplication, QEvent, QEventLoop, QTimer
    from PyQt6.QtWidgets import QApplication
    from ptb_desktop.host import Workbench, register_scheme
    from ptb_desktop.papers import BASE_URL, PaperService

    register_scheme()
    app = QApplication(['M18-paper-background-owned-QA'])
    with verification_run('m18', 'paper-background-qt',
                          'Local licensed P2A original/translation/analysis PDFs; native raster, '
                          'visible default-GPU Qt; light/dark with wallpaper on/off') as (out, scratch):
        source = ROOT / 'output/paper-reading/2408.05641v1/layout-20261012/publish'
        catalog = json.loads((source / 'catalog.json').read_text('utf8'))
        paper = next(p for p in catalog['papers'] if p['id'] == 'arxiv-2408-05641-v1')
        catalog = dict(schema='ptb-papers/1', papers=[paper])
        items = {'catalog.json': json.dumps(catalog).encode()}
        for field in ('original', 'translation', 'analysisPdf'):
            asset = paper[field]
            data = (source / asset['path']).read_bytes()
            assert hashlib.sha256(data).hexdigest() == asset['sha256']
            items[asset['path']] = data

        class Response(BytesIO):
            def geturl(self):
                return self.url

        class Opener:
            def open(self, request, timeout):
                result = Response(items[request.full_url.removeprefix(BASE_URL)])
                result.url = request.full_url
                return result

        service = PaperService(scratch / 'papers', opener=Opener())
        service.set_auto_receive(False)
        service.refresh(threading.Event())
        service.download('0001-01-01', threading.Event(), lambda _: None, [paper['id']])
        w = Workbench(ROOT / 'frontend/dist', test=True, vocal_profile=scratch / 'vocal',
                      start_module='M18')
        w.papers_bridge.service = service
        w.resize(1440, 900)
        w.show()
        report = dict(success=False, cases=[], sources={k: paper[k]['sha256'] for k in
                      ('original', 'translation', 'analysisPdf')})

        def pause(ms=100):
            loop = QEventLoop()
            QTimer.singleShot(ms, loop.quit)
            loop.exec()

        def js(code):
            loop, values = QEventLoop(), []
            w.page.runJavaScript(code, lambda v: (values.append(v), loop.quit()))
            QTimer.singleShot(10000, loop.quit)
            loop.exec()
            assert values, 'JavaScript timeout'
            return values[0]

        def until(code):
            end = time.monotonic() + 30
            while time.monotonic() < end:
                if js('Boolean(' + code + ')'):
                    return
                pause()
            raise AssertionError(code + '\n' + str(js('document.body.innerText.slice(-1000)')))

        try:
            until("document.querySelectorAll('.paper-card').length")
            js("document.querySelector('.paper-card').click()")
            for language, label in [('original', '原文'), ('translation', '中文译文'), ('analysis', '论文解读')]:
                js("[...document.querySelectorAll('.pdf-controls button')].find(e=>e.textContent===" + json.dumps(label) + ").click()")
                until("document.querySelector('.paper-sheet')?.complete&&document.querySelector('.paper-sheet')?.naturalWidth>0&&document.querySelectorAll('.pdf-text-line').length>0")
                for theme in ('dark', 'light'):
                    for wallpaper in ('on', 'off'):
                        js("(()=>{const r=document.documentElement;r.dataset.theme=" + json.dumps(theme) +
                           ";r.dataset.backgroundImage=" + json.dumps(wallpaper) +
                           ";r.style.setProperty('--panel'," + json.dumps('#182432' if theme == 'dark' else '#fff') +
                           ");r.style.setProperty('--text'," + json.dumps('#e9f0f9' if theme == 'dark' else '#132a49') +
                           ");r.style.setProperty('--app'," + json.dumps('#111b26' if theme == 'dark' else '#f3f6fb') +
                           ");r.style.backgroundImage='linear-gradient(120deg,#254734,#22314b)';})()")
                        pause(400)
                        values = js("(()=>{const p=document.querySelector('.pdf-page'),s=getComputedStyle(p),i=document.querySelector('.paper-sheet'),b=p.getBoundingClientRect();return {background:s.backgroundColor,filter:getComputedStyle(i).filter,textOpacity:getComputedStyle(document.querySelector('.pdf-text-layer')).opacity,x:b.x,y:b.y,width:b.width,height:b.height}})()")
                        assert values['background'] == 'rgb(255, 255, 255)', values
                        assert values['filter'] == 'none', values
                        assert values['textOpacity'] == '0', values
                        image = w.view.grab().toImage()
                        scale = image.devicePixelRatio()
                        pixel = image.pixelColor(round((values['x'] + 8) * scale), round((values['y'] + 8) * scale))
                        assert min(pixel.red(), pixel.green(), pixel.blue()) >= 250, pixel.name()
                        name = f'{language}-{theme}-{wallpaper}'
                        image.save(str(out / (name + '.png')))
                        report['cases'].append(dict(name=name, paperCorner=pixel.name(), **values))
            # The transparent selectable text overlay remains operational.
            js("(()=>{const e=document.querySelector('.pdf-text-line');const r=document.createRange();r.selectNodeContents(e);const s=getSelection();s.removeAllRanges();s.addRange(r);document.dispatchEvent(new Event('selectionchange'));})()")
            until("[...document.querySelectorAll('button')].some(e=>e.textContent==='复制选文'&&!e.disabled)")
            report['selection'] = js('getSelection().toString()')
            assert report['selection'].strip()
            report['success'] = True
        finally:
            w.closing = True
            w.close()
            w.page.deleteLater()
            app.processEvents()
            w.papers_bridge.pool.shutdown(wait=True)
            w.papers_bridge.network_pool.shutdown(wait=True)
            if w.service.process:
                w.service.process.wait(timeout=15)
            w.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            pause(400)
            write_json(out / 'report.json', report)
    print(out, flush=True)


if __name__ == '__main__':
    main()
