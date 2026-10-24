"""Real Qt pointer selection, annotation persistence and PDF export in isolated storage."""
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
    from PyQt6.QtCore import QCoreApplication, QEvent, QEventLoop, QPoint, QPointF, QTimer, Qt
    from PyQt6.QtGui import QMouseEvent
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication
    from ptb_desktop.host import Workbench, register_scheme
    from ptb_desktop.papers import BASE_URL, PaperService
    from pypdf import PdfReader

    register_scheme()
    app = QApplication(['M18-annotations-owned-QA'])
    with verification_run('m18', 'annotations-qt',
                          'Licensed local P2A PDFs; default GPU real Qt pointer clicks; native annotations and export') as (out, scratch):
        print(out, flush=True)
        source = ROOT / 'output/paper-reading/2408.05641v1/layout-20261012/publish'
        catalog = json.loads((source / 'catalog.json').read_text('utf8'))
        paper = next(p for p in catalog['papers'] if p['id'] == 'arxiv-2408-05641-v1')
        items = {'catalog.json': json.dumps(dict(schema='ptb-papers/1', papers=[paper])).encode()}
        for kind in ('original', 'translation', 'analysisPdf'):
            data = (source / paper[kind]['path']).read_bytes()
            assert hashlib.sha256(data).hexdigest() == paper[kind]['sha256']
            items[paper[kind]['path']] = data

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
        w = Workbench(ROOT / 'frontend/dist', test=True, vocal_profile=scratch / 'vocal', start_module='M18')
        w.papers_bridge.service = service
        w.resize(1440, 1000)
        w.show()
        report = dict(success=False, checks=[])

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

        def until(code, seconds=20):
            end = time.monotonic() + seconds
            while time.monotonic() < end:
                if js('Boolean(' + code + ')'):
                    return
                pause()
            raise AssertionError(code + '\n' + str(js('document.body.innerText.slice(-1300)')))

        def point(x, y):
            target = w.view.focusProxy() or w.view
            return target, target.mapFromGlobal(w.view.mapToGlobal(QPoint(round(x), round(y))))

        def click(text):
            expression = '[...document.querySelectorAll("button")].find(e=>e.offsetParent&&e.textContent.trim()===' + json.dumps(text) + ')'
            until('!!(' + expression + ')&&!(' + expression + ').disabled')
            x, y = js('(()=>{const r=(' + expression + ').getBoundingClientRect();return [r.left+r.width/2,r.top+r.height/2]})()')
            target, pos = point(x, y)
            QTest.mouseClick(target, Qt.MouseButton.LeftButton, pos=pos)
            pause(200)

        def shortcut(redo=False):
            target = w.view.focusProxy() or w.view
            modifiers = Qt.KeyboardModifier.ControlModifier
            if redo:
                modifiers |= Qt.KeyboardModifier.ShiftModifier
            QTest.keyClick(target, Qt.Key.Key_Z, modifiers)
            pause(150)

        def note_text():
            return next(item['text'] for item in service.annotations(paper['id'], 'original')['items'] if item['kind'] == 'note')

        def select_line(index=3):
            js('document.querySelector(".paper-viewport").scrollTop=0')
            until('document.querySelectorAll("[data-page=\\"1\\"] .pdf-text-line").length>3')
            # Wait for the fullscreen window/layout/raster transition before taking pointer coordinates.
            previous = None
            for _ in range(20):
                current = js('(()=>{const r=document.querySelectorAll("[data-page=\\"1\\"] .pdf-text-line")[3].getBoundingClientRect();return [r.x,r.y,r.width,r.height,document.querySelector(".paper-sheet").src]})()')
                if current == previous:
                    break
                previous = current
                pause(250)
            else:
                raise AssertionError('PDF geometry/raster did not settle before pointer selection')
            coords = js('(()=>{const r=document.querySelectorAll("[data-page=\\"1\\"] .pdf-text-line")[' + str(index) + '].getBoundingClientRect();return [r.left+3,r.top+r.height/2,r.right-3,r.top+r.height/2]})()')
            js('window.m18Pointer=[];if(!window.m18Track){window.m18Track=true;for(const name of ["pointerdown","pointermove","pointerup"])document.addEventListener(name,e=>window.m18Pointer.push([name,e.clientX,e.clientY,e.buttons,e.target.className]),true);}')
            target, start = point(*coords[:2])
            _, end = point(*coords[2:])
            QTest.mousePress(target, Qt.MouseButton.LeftButton, pos=start)
            # Deliver a deterministic Qt drag with an explicit held-button state.
            QCoreApplication.sendEvent(target, QMouseEvent(QEvent.Type.MouseMove, QPointF(end),
                QPointF(target.mapToGlobal(end)), Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier))
            pause(100)
            QTest.mouseRelease(target, Qt.MouseButton.LeftButton, pos=end)
            try:
                until('document.querySelector(".pdf-mark-tools").textContent.includes("已选择 1 页")', 5)
            except AssertionError:
                report['pointerDiagnostics'] = dict(coords=coords, nativeSize=[w.view.width(), w.view.height()],
                    dom=js('({size:[innerWidth,innerHeight],events:m18Pointer,selection:getSelection().toString(),focus:document.activeElement?.className})'))
                raise

        try:
            until('document.querySelector(".paper-card")')
            js('document.querySelector(".paper-card").click()')
            until('document.querySelector(".pdf-text-line")')
            report['environment'] = js('({secure:isSecureContext,uuid:typeof crypto.randomUUID,userAgent:navigator.userAgent})')
            select_line()
            report['quote'] = js('getSelection().toString()')
            js('getSelection().removeAllRanges()')
            pause(200)
            click('荧光笔')
            until('document.querySelector(".pdf-mark.highlight")', 5)
            assert service.annotations(paper['id'], 'original')['items'][0]['quote'].strip()
            report['checks'].append('Native mouse selection/highlight persists after injected native-range collapse')
            shortcut()
            until('!document.querySelector(".pdf-mark.highlight")')
            assert service.annotations(paper['id'], 'original')['items'] == []
            shortcut(True)
            until('document.querySelector(".pdf-mark.highlight")')
            x, y = js('(()=>{const r=document.querySelector(".pdf-mark.highlight").getBoundingClientRect();return [r.x+r.width/2,r.y+r.height/2]})()')
            target, pos = point(x, y)
            QTest.mouseClick(target, Qt.MouseButton.LeftButton, pos=pos)
            until('document.querySelector(".pdf-mark.is-selected")')
            QTest.keyClick(target, Qt.Key.Key_Delete)
            until('!document.querySelector(".pdf-mark.highlight")')
            shortcut()
            until('document.querySelector(".pdf-mark.highlight")')
            shortcut(True)
            until('!document.querySelector(".pdf-mark.highlight")')
            shortcut()
            until('document.querySelector(".pdf-mark.highlight")')
            report['checks'].append('Native Ctrl+Z/Ctrl+Shift+Z writes actual highlight undo/redo; pointer selection/Delete/restoration works')
            select_line(6)
            js('getSelection().removeAllRanges()')
            pause(200)
            click('添加批注')
            until('document.querySelector("[aria-label=批注内容]")', 5)
            assert not js('document.querySelector("dialog blockquote")')
            js('(()=>{const e=document.querySelector("[aria-label=批注内容]");e.value="选文批注真实Qt回归";e.dispatchEvent(new Event("input",{bubbles:true}));})()')
            click('保存批注')
            until('document.querySelector(".pdf-note-pin")')
            for theme in ('light', 'dark'):
                js('document.documentElement.dataset.theme=' + json.dumps(theme))
                pause(200)
                w.view.grab().save(str(out / (theme + '.png')))
            report['checks'].append('Ordinary reader native mouse note opens/saves after range collapse; light/dark marks and pin')
            click('✎')
            until('document.querySelector(".annotation-note-dialog")')
            for theme in ('light', 'dark'):
                js('document.documentElement.dataset.theme=' + json.dumps(theme))
                pause(200)
                w.view.grab().save(str(out / ('sticky-' + theme + '.png')))
            js('document.querySelector("[aria-label=批注内容]").focus()')
            target = w.view.focusProxy() or w.view
            QTest.keyClicks(target, 'draft')
            shortcut()
            assert len(service.annotations(paper['id'], 'original')['items']) == 2
            assert note_text() == '选文批注真实Qt回归'
            js('(()=>{const e=document.querySelector("[aria-label=批注内容]");e.value="修订后的选文批注真实Qt回归";e.dispatchEvent(new Event("input",{bubbles:true}));})()')
            click('保存批注')
            shortcut()
            assert note_text() == '选文批注真实Qt回归'
            shortcut(True)
            assert note_text() == '修订后的选文批注真实Qt回归'
            report['checks'].append('Native note edit undo/redo persists; textarea undo leaves stored marks intact; sticky editor hides quote and renders light/dark')
            click('全屏阅读')
            until('document.fullscreenElement')
            click('批注 2')
            click('添加本页笔记')
            js('(()=>{const e=document.querySelector("[aria-label=批注内容]");e.value="全屏本页笔记";e.dispatchEvent(new Event("input",{bubbles:true}));})()')
            click('保存批注')
            until('document.querySelectorAll(".pdf-note-pin").length===2')
            report['checks'].append('Fullscreen dialog/native note save; Chrome separately verifies fullscreen pointer selected-text notes')
            click('退出全屏')
            click('中文译文')
            until('document.querySelector(".pdf-controls .primary")?.textContent==="中文译文"&&!document.querySelector(".pdf-mark")')
            click('原文')
            until('document.querySelector(".pdf-note-pin")')
            until('document.querySelector("[aria-label=撤销标注]").disabled')
            report['checks'].append('Document reopen reloads persisted highlight/note; translation remains unmarked')
            destination = scratch / 'annotated.pdf'
            w.papers_bridge.export_picker = lambda _: str(destination)
            click('导出 PDF')
            js('document.querySelector(".export-check input").click()')
            click('选择位置并导出')
            until('!document.querySelector(".export-check")')
            exported = PdfReader(destination)
            annots = [a.get_object() for a in exported.pages[0]['/Annots']]
            assert any(a['/Subtype'] == '/Highlight' for a in annots)
            assert any('选文批注真实Qt回归' in a.get('/Contents', '') for a in annots)
            assert service.annotations(paper['id'], 'translation')['items'] == []
            assert service.annotations(paper['id'], 'analysis')['items'] == []
            report['checks'].append('Standard PDF highlight/text export and document isolation')
            click('帮助')
            until('document.body.textContent.includes("点击颜色或标记工具时会保留本次选文")')
            pause(500)
            for theme in ('light', 'dark'):
                js('document.documentElement.dataset.theme=' + json.dumps(theme))
                for _ in range(3):
                    js('(()=>{const e=[...document.querySelectorAll("h2,h3,h4")].find(e=>e.textContent.includes("荧光笔与文字批注"));e?.scrollIntoView();})()')
                    pause(200)
                until('(()=>{const e=[...document.querySelectorAll("h2,h3,h4")].find(e=>e.textContent.includes("荧光笔与文字批注"));const r=e?.getBoundingClientRect();return r&&r.top>=0&&r.top<innerHeight/2;})()')
                w.view.grab().save(str(out / ('manual-' + theme + '.png')))
            report['checks'].append('Generated M18 help loads updated selection/storage/retry instructions in real Qt, light/dark')
            report['success'] = True
        except Exception as exc:
            report['failure'] = str(exc)
            report['ui'] = js('document.body.innerText.slice(-1800)')
            w.view.grab().save(str(out / 'failure.png'))
            raise
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


if __name__ == '__main__':
    main()
