"""Owned M18 PDF glyph-paint, pixel density and anchored wheel-zoom checks."""
import argparse, hashlib, json, shutil, time
from pathlib import Path
from workbench_source import configure
from verification_artifacts import verification_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--source', type=Path, required=True, help='Caller-owned authorized extIPA PDF; always read-only')
    args = parser.parse_args()
    configure()
    import numpy as np
    from PyQt6.QtCore import QEventLoop, QTimer, Qt, QRect, QPoint, QPointF
    from PyQt6.QtGui import QImage, QWheelEvent
    from PyQt6.QtWidgets import QApplication
    from ptb_desktop.host import Workbench, register_scheme
    from ptb_desktop.papers import PaperService
    root = Path(__file__).resolve().parents[1]
    source = args.source.resolve()
    register_scheme()
    app = QApplication(['M18-R5-owned-PDF-QA'])
    with verification_run('m18', 'r5-probe' if args.probe else 'r5-qt', recipe='Read-only authorized extIPA 2025; isolated Qt, bitmap/text-layer paint comparison, DPR and Ctrl+wheel') as (out, scratch):
        library = scratch / 'library'; library.mkdir()
        shutil.copyfile(source, library / 'ExtIPA_chart_(2025).pdf')
        shutil.copyfile(root / 'output/paper-reading/2408.05641v1/original.pdf', library / 'Coarticulation.pdf')
        service = PaperService(scratch / 'papers')
        service.set_auto_receive(False); service.set_library_root(library)
        report = dict(success=False, sourceSHA256=hashlib.sha256(source.read_bytes()).hexdigest(), checks=[], renders=[])
        original_render = service.render
        def render(*values):
            start = time.monotonic(); result = original_render(*values)
            report['renders'].append(dict(page=values[2], requestedWidth=values[3], seconds=round(time.monotonic()-start, 4), imageWidth=result.get('imageWidth')))
            return result
        service.render = render
        w = Workbench(root / 'frontend/dist', test=True, vocal_profile=scratch / 'vocal', start_module='M18')
        w.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        w.papers_bridge.service = service
        w.resize(1440, 1000); w.show()
        def pause(ms=100):
            loop=QEventLoop(); QTimer.singleShot(ms,loop.quit); loop.exec()
        def js(code):
            loop=QEventLoop(); values=[]
            w.page.runJavaScript(code,lambda value:(values.append(value),loop.quit()))
            QTimer.singleShot(10000,loop.quit); loop.exec()
            assert values, 'JavaScript callback timed out'
            return values[0]
        def until(code, seconds=40):
            end=time.monotonic()+seconds
            while time.monotonic()<end:
                if js(code): return
                pause()
            raise AssertionError(code+'\n'+str(js('document.body.innerText.slice(-1600)')))
        def capture():
            b=js("(()=>{const e=document.querySelector('.paper-viewport').getBoundingClientRect();return [e.x+20,e.y+20,e.width-40,e.height-40]})()")
            pause(250)
            return w.view.grab(QRect(*map(round,b))).toImage().convertToFormat(QImage.Format.Format_RGBA8888)
        def pixels(im):
            bits=im.constBits(); bits.setsize(im.sizeInBytes())
            return np.frombuffer(bits,dtype=np.uint8).reshape(im.height(),im.bytesPerLine())[:,:im.width()*4].copy()
        try:
            until("!!document.querySelector('.source-tabs button:nth-child(2)')")
            js("document.querySelector('.source-tabs button:nth-child(2)').click();document.documentElement.dataset.theme='light'")
            until("document.querySelectorAll('.paper-card').length===2")
            js("[...document.querySelectorAll('.paper-card')].find(e=>e.textContent.includes('ExtIPA')).click()")
            until("!!document.querySelector('.pdf-text-line')&&document.querySelector('.paper-sheet').complete")
            js("document.querySelector('.paper-viewport').scrollTop=document.querySelector('.pdf-page').offsetHeight*.43")
            pause(500)
            painted=capture(); painted.save(str(out/'chart-with-text-layer.png'))
            if not args.probe:
                js("document.querySelectorAll('.pdf-text-line').forEach(e=>e.style.color='black')")
                assert np.array_equal(pixels(capture()),pixels(painted)), 'Glyph paint must remain isolated even if font color is overridden'
                js("document.querySelectorAll('.pdf-text-line').forEach(e=>e.style.color='')")
            js("document.querySelector('.pdf-text-layer').style.visibility='hidden'")
            hidden=capture(); hidden.save(str(out/'chart-without-text-layer.png'))
            delta=int(np.count_nonzero(np.any(pixels(painted).reshape(-1,4)[:,:3]!=pixels(hidden).reshape(-1,4)[:,:3],axis=1)))
            report.update(textLayerPaintDifferentPixels=delta, dpr=w.devicePixelRatioF(), pageZoom=w.view.zoomFactor())
            js("document.querySelector('.pdf-text-layer').style.visibility=''")
            print(json.dumps(dict(differentPixels=delta, dpr=report['dpr'], renders=report['renders']),ensure_ascii=False),flush=True)
            if not args.probe:
                assert delta==0, 'Invisible selectable text must not repaint PDF glyphs'
                report['checks'].append('Text layer on/off produces identical bitmap pixels, including combining/IPA glyphs')
                def select(value):
                    js("(()=>{const e=document.querySelector('[aria-label=阅读缩放]');e.value="+json.dumps(value)+";e.dispatchEvent(new Event('change',{bubbles:true}));})()")
                select('200')
                until("document.querySelector('.paper-sheet').naturalWidth===2448")
                report['checks'].append('200% reloads at 2448 raster pixels on DPR 1.5 instead of stretching cached fit-width image')
                pause(300)
                point=js("(()=>{const e=document.querySelector('.paper-viewport').getBoundingClientRect();return [e.left+e.width/2,e.top+e.height/2]})()")
                def anchor():
                    return js("(()=>{const p=document.querySelector('.pdf-page').getBoundingClientRect();return [("+str(point[0])+"-p.left)/p.width,("+str(point[1])+"-p.top)/p.height]})()")
                before=anchor();target=w.view.focusProxy()
                pos=QPointF(*point);globalPos=QPointF(w.view.mapToGlobal(QPoint(*map(round,point))))
                event=QWheelEvent(pos,globalPos,QPoint(),QPoint(0,120),Qt.MouseButton.NoButton,Qt.KeyboardModifier.ControlModifier,Qt.ScrollPhase.NoScrollPhase,False)
                QApplication.sendEvent(target,event)
                until("Number(document.querySelector('[aria-label=阅读缩放]').value)>200")
                until("document.querySelector('.paper-sheet').naturalWidth>2448")
                after=anchor();assert max(abs(a-b) for a,b in zip(before,after))<.003,(before,after)
                assert js('document.documentElement.style.zoom')=='1'
                report['wheelAnchor']={'before':before,'after':after}
                report['checks'].append('Native Qt Ctrl+wheel enlarges PDF around pointer, without changing whole application scale')
                # Fully transparent DOM glyphs still support range selection and marks.
                js("(()=>{const e=[...document.querySelectorAll('.pdf-text-line')].find(e=>e.textContent.includes('strong articulation'))??document.querySelector('.pdf-text-line');const r=document.createRange();r.selectNodeContents(e);const s=getSelection();s.removeAllRanges();s.addRange(r);})()")
                until("document.querySelectorAll('.pdf-selection-layer span').length>0")
                assert js("!([...document.querySelectorAll('button')].find(e=>e.textContent==='荧光笔').disabled)")
                js("[...document.querySelectorAll('button')].find(e=>e.textContent==='荧光笔').click()")
                until("document.querySelectorAll('.pdf-mark.highlight').length>0")
                report['checks'].append('Invisible glyph layer remains selectable, live range overlay appears, native highlight persists')
                select('fit');until("document.querySelector('[aria-label=阅读缩放]').value==='fit'")
                until("document.querySelector('.paper-sheet').naturalWidth<2000")
                js("document.documentElement.dataset.theme='dark'");pause(300)
                capture().save(str(out/'chart-dark-fit.png'))
                js("[...document.querySelectorAll('.paper-card')].find(e=>e.textContent.includes('Coarticulation')).click()")
                until("document.querySelectorAll('.pdf-page').length===5&&!!document.querySelector('.paper-sheet')")
                for value in ('300','50','150','fit'):
                    select(value);pause(40)
                until("document.querySelector('.paper-sheet').complete&&document.querySelector('[aria-label=阅读缩放]').value==='fit'")
                assert not js("document.querySelector('.pdf-warning[role=alert]')?.textContent")
                report['checks'].append('Rapid zoom changes and five-page document switching recover, dark theme stays readable')
                from pypdf import PdfWriter
                bounds=scratch/'bounds'; bounds.mkdir()
                for name, dimensions in [('portrait',(595,842)),('tall',(120,600))]:
                    writer=PdfWriter();writer.add_blank_page(width=dimensions[0],height=dimensions[1])
                    with (bounds/(name+'.pdf')).open('wb') as stream:writer.write(stream)
                bounded=PaperService(scratch/'bounds-profile'); bounded.set_library_root(bounds)
                try:
                    for item in bounded.library_status()['documents']:
                        result=bounded.render(item['id'],'original',1,4096)
                        assert result['imageWidth']*result['imageHeight']<=16_777_216
                        assert result['imageHeight']<=6144
                        assert bounded.render(item['id'],'original',1,4096) is result
                        report.setdefault('boundedRasters',[]).append([result['imageWidth'],result['imageHeight']])
                finally:bounded.close_documents()
                report['checks'].append('Native portrait and tall-page maximum requests obey pixel/height budgets and reuse identical cached raster')
                assert hashlib.sha256(source.read_bytes()).hexdigest()==report['sourceSHA256']
            report['success']=True
        finally:
            w.closing=True; w.close(); w.page.deleteLater(); app.processEvents()
            w.papers_bridge.pool.shutdown(wait=True); w.papers_bridge.network_pool.shutdown(wait=True)
            if w.service.process:
                w.service.process.wait(timeout=15)
            (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf8')
    print(str(out),flush=True)


if __name__=='__main__': main()
