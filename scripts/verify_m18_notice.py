"""Visible Qt: delete the editorial notice and reacquire it through history."""
import hashlib
from io import BytesIO
import json
from pathlib import Path
import threading
import time

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
    app = QApplication(['M18-notice-owned-QA'])
    with verification_run('m18', 'notice-qt', 'Visible default-GPU source Qt; owned local '
                          'notice/paper copies, real WebChannel, native cache/notes, light/dark') as (out, scratch):
        note = (ROOT / 'output/paper-reading/intro/请看这里~_栏目说明.pdf').read_bytes()
        original = (ROOT / 'output/paper-reading/2408.05641v1/original.pdf').read_bytes()

        def asset(path, data):
            return dict(path=path, size=len(data), sha256=hashlib.sha256(data).hexdigest())

        n = dict(id='reading-introduction', title='请看这里~', authors='井韶子',
                 publishedAt='2026-10-12', updatedAt='2026-10-12', pdf=asset('notice/intro.pdf', note))
        p = dict(id='owned-paper', title='Towards a Quantitative Analysis of Coarticulation',
                 titleZh='协同发音定量分析', authors='Chaofei Fan et al.', version='arXiv:2408.05641v1',
                 publishedAt='2026-10-12', submittedAt='2024-08-10', guide='Owned QA copy', translationNote='QA',
                 sourceUrl='https://arxiv.org/abs/2408.05641v1',
                 license=dict(id='CC-BY-4.0', url='https://creativecommons.org/licenses/by/4.0/'),
                 original=asset('paper/original.pdf', original), translation=asset('paper/original.pdf', original))
        catalog = dict(schema='ptb-papers/1', papers=[p], notice=n)
        items = {'catalog.json': json.dumps(catalog).encode(), 'notice/intro.pdf': note,
                 'paper/original.pdf': original}
        calls = []

        class Response(BytesIO):
            def geturl(self):
                return self.url

        class Opener:
            def open(self, request, timeout):
                calls.append(request.full_url)
                result = Response(items[request.full_url.removeprefix(BASE_URL)])
                result.url = request.full_url
                return result

        service = PaperService(scratch / 'papers', opener=Opener())
        service.set_auto_receive(False)
        service.refresh(threading.Event())
        service.download('0001-01-01', threading.Event(), lambda _: None, [p['id']])
        annotation = dict(id='notice_note', page=1, kind='note', color='yellow',
                          text='栏目说明的保留笔记', quote='', rects=[[.94, .025, .025, .025]])
        service.save_annotation(n['id'], 'original', 0, annotation, False)
        w = Workbench(ROOT / 'frontend/dist', test=True, vocal_profile=scratch / 'vocal', start_module='M18')
        w.papers_bridge.service = service
        w.resize(1440, 900)
        w.show()
        report = dict(success=False, cases=[], sources=dict(notice=n['pdf']['sha256'], paper=p['original']['sha256']))

        def pause(ms=100):
            loop = QEventLoop(); QTimer.singleShot(ms, loop.quit); loop.exec()

        def js(code):
            loop, values = QEventLoop(), []
            w.page.runJavaScript(code, lambda value: (values.append(value), loop.quit()))
            QTimer.singleShot(10000, loop.quit); loop.exec()
            assert values, 'JavaScript timeout'
            return values[0]

        def until(code):
            end = time.monotonic() + 30
            while time.monotonic() < end:
                if js('Boolean(' + code + ')'): return
                pause()
            raise AssertionError(code + '\n' + str(js('document.body.innerText.slice(-1600)')))

        def click(text):
            assert js("(()=>{const e=[...document.querySelectorAll('button')].find(e=>e.textContent.trim()===" +
                      json.dumps(text) + ");if(!e||e.disabled)return false;e.click();return true})()"), text

        def menu(keyboard=False):
            event = "new KeyboardEvent('keydown',{key:'F10',shiftKey:true,bubbles:true})" if keyboard else \
                "new MouseEvent('contextmenu',{bubbles:true,clientX:200,clientY:300})"
            js("document.querySelector('.paper-intro').dispatchEvent(" + event + ")")
            until("document.querySelector('[role=menuitem]')===document.activeElement")
            click('删除本机副本')
            until("document.querySelector('dialog[open]')?.textContent.includes('删除栏目说明')")

        try:
            until("document.querySelector('.paper-intro')&&document.querySelector('.paper-sheet')?.naturalWidth>0")
            for theme in ('light', 'dark'):
                js('document.documentElement.dataset.theme=' + json.dumps(theme))
                menu(keyboard=True)
                click('取消')
                assert js("!!document.querySelector('.paper-intro')") and service.status()['notice']['downloaded']
                menu()
                pause(200); w.view.grab().save(str(out / (theme + '-confirm.png')))
                click('删除本机副本')
                until("!document.querySelector('.paper-intro')&&document.querySelector('.paper-fold strong')?.textContent==='协同发音定量分析'")
                until("document.querySelectorAll('.pdf-page').length===5&&document.querySelector('.paper-sheet')?.naturalWidth>0")
                assert service.status()['notice']['dismissed'] and not service._path(n['pdf']).exists()
                assert js("document.querySelector('.source-tabs button').textContent.trim()==='服务器推荐 1'")

                # Recreate the native service, reload the actual page, and run the startup check.
                w.papers_bridge.pool.submit(service.close_documents).result(timeout=10)
                service = PaperService(scratch / 'papers', opener=Opener())
                w.papers_bridge.service = service
                before = len(calls)
                loaded = []
                loop = QEventLoop()
                def finished(ok):
                    loaded.append(ok); loop.quit()
                w.page.loadFinished.connect(finished)
                w.page.triggerAction(w.page.WebAction.Reload)
                QTimer.singleShot(30000, loop.quit); loop.exec()
                w.page.loadFinished.disconnect(finished)
                assert loaded == [True]
                until("document.querySelectorAll('.paper-card').length===1&&!document.querySelector('.paper-intro')&&[...document.querySelectorAll('button')].some(e=>e.textContent.trim()==='获取往期'&&!e.disabled)")
                w.papers_bridge.automatic_check(startup=True)
                w.papers_bridge.network_pool.submit(lambda: None).result(timeout=10)
                pause(250)
                assert calls[before:] == [BASE_URL + 'catalog.json'] and not service._path(n['pdf']).exists()
                assert not js("document.querySelector('.paper-intro')")
                js('document.documentElement.dataset.theme=' + json.dumps(theme))
                click('获取往期')
                until("document.querySelectorAll('.archive-row:not(.obtained)').length===1&&document.querySelector('.archive-row input')?.disabled===false")
                assert js("document.querySelector('.archive-row').textContent.includes('请看这里~')&&document.querySelector('.archive-row').textContent.includes('栏目说明')")
                assert not js("document.querySelector('.archive-row .abstract-tooltip')")
                # Year/month filtering must include the independent editorial notice.
                click('10 月')
                until("document.querySelectorAll('.archive-row:not(.obtained)').length===1")
                js("document.querySelector('.archive-row input').click()")
                until("[...document.querySelectorAll('button')].some(e=>e.textContent.trim()==='获取已选 1 篇'&&!e.disabled)")
                size = f'{len(note)/1_000_000:.1f} MB' if len(note) >= 1_000_000 else f'{len(note)/1000:.0f} KB'
                assert js("document.querySelector('dialog .hint').textContent").find(size) >= 0
                pause(200); w.view.grab().save(str(out / (theme + '-archive.png')))
                before = len(calls)
                click('获取已选 1 篇')
                until("!!document.querySelector('.paper-intro')&&!document.querySelector('.archive-row:not(.obtained)')")
                assert calls[before:] == [BASE_URL + 'notice/intro.pdf']
                assert service.annotations(n['id'], 'original')['items'] == [annotation]
                js("document.querySelector('.obtained-toggle input').click()")
                until("document.querySelectorAll('.archive-row.obtained input:disabled').length===2")
                click('关闭')
                js("document.querySelector('.paper-intro').click()")
                until("document.querySelectorAll('.pdf-page').length===1&&document.querySelector('.paper-sheet')?.naturalWidth>0")
                assert js("document.querySelector('.source-tabs button').textContent.trim()==='服务器推荐 1'")
                report['cases'].append(dict(theme=theme, keyboardCancel=True, deletion=True, restartHidden=True,
                                           startupNoRefetch=True, historyRestore=True, notesRetained=True, singleAssetBytes=len(note)))
            js("document.querySelector('.paper-card:not(.paper-intro)').dispatchEvent(new MouseEvent('contextmenu',{bubbles:true,clientX:200,clientY:350}))")
            until("document.querySelector('[role=menuitem]')")
            click('删除本机副本')
            until("document.querySelector('dialog[open]')?.textContent.includes('删除论文副本')")
            click('删除本机副本')
            until("document.querySelectorAll('.paper-card').length===1&&document.querySelector('.paper-intro')")
            menu(); click('删除本机副本')
            until("!document.querySelector('.paper-card')&&!document.querySelector('.paper-reader')&&document.querySelector('.paper-empty')")
            pause(200); w.view.grab().save(str(out / 'empty-library.png'))
            click('获取往期')
            until("document.querySelectorAll('.archive-row:not(.obtained)').length===2&&document.querySelector('.archive-row input')?.disabled===false")
            js("[...document.querySelectorAll('.archive-row')].find(e=>e.textContent.includes('请看这里~')).querySelector('input').click()")
            click('获取已选 1 篇')
            until("document.querySelectorAll('.archive-row:not(.obtained)').length===1&&document.querySelector('.paper-intro')")
            click('关闭')
            until("document.querySelector('.paper-sheet')?.naturalWidth>0")
            assert service.status()['papers'][0]['dismissed'] and service.status()['notice']['downloaded']
            assert js("document.querySelector('.source-tabs button').textContent.trim()==='服务器推荐 0'")
            report['emptyLibraryRestore'] = True
            click('帮助')
            until("document.body.textContent.includes('删除后重启或检查更新均不会重新获取')")
            report['updatedHelpReadable'] = True
            report['success'] = True
        finally:
            w.closing = True; w.close(); w.page.deleteLater(); app.processEvents()
            w.papers_bridge.pool.shutdown(wait=True); w.papers_bridge.network_pool.shutdown(wait=True)
            if w.service.process: w.service.process.wait(timeout=15)
            w.deleteLater(); QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete); pause(400)
            write_json(out / 'report.json', report)
    print(out, flush=True)


if __name__ == '__main__':
    main()
