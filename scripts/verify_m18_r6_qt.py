"""Owned Qt tests for the pinned note, annotations, citation and deletion."""
import hashlib
import json
import threading
import time
from pathlib import Path
from workbench_source import configure
from verification_artifacts import verification_run


def main():
    configure()
    from PyQt6.QtCore import QEventLoop, QTimer, Qt
    from PyQt6.QtWidgets import QApplication
    from pypdf import PdfReader
    from ptb_desktop.host import Workbench, register_scheme
    from ptb_desktop.papers import PaperService, BASE_URL
    from io import BytesIO
    root = Path(__file__).resolve().parents[1]
    register_scheme(); app = QApplication(['M18-R6-owned-QA'])
    with verification_run('m18', 'r6-qt', 'Isolated real Qt PDF/annotation/filesystem and light/dark UI; owned copies only') as (out, scratch):
        original = (root/'output/paper-reading/2408.05641v1/original.pdf').read_bytes()
        note = (root/'output/paper-reading/intro/请看这里~_栏目说明.pdf').read_bytes()
        def asset(path, data): return dict(path=path, size=len(data), sha256=hashlib.sha256(data).hexdigest())
        p = dict(id='p2a-qa', title='Towards a Quantitative Analysis of Coarticulation with a Phoneme-to-Articulatory Model',
                 titleZh='协同发音定量分析', authors='Chaofei Fan、Jaimie M. Henderson、Chris Manning、Francis R. Willett',
                 version='arXiv:2408.05641v1', sourceUrl='https://arxiv.org/abs/2408.05641v1',
                 publishedAt='2026-10-12', submittedAt='2024-08-10', guide='Owned QA copy', translationNote='QA',
                 license=dict(id='CC-BY-4.0', url='https://creativecommons.org/licenses/by/4.0/'),
                 original=asset('p2a-qa/original.pdf',original), translation=asset('p2a-qa/original.pdf',original))
        n = dict(id='reading-introduction', title='请看这里~', authors='井韶子', publishedAt='2026-10-12', updatedAt='2026-10-12', pdf=asset('reading-introduction/note.pdf',note))
        catalog = dict(schema='ptb-papers/1', papers=[p], notice=n)
        items = {'catalog.json':json.dumps(catalog).encode(), 'reading-introduction/note.pdf':note, 'p2a-qa/original.pdf':original}
        class Response(BytesIO):
            def geturl(self): return self.url
        class Opener:
            def open(self, request, timeout):
                response=Response(items[request.full_url.removeprefix(BASE_URL)]);response.url=request.full_url;return response
        service=PaperService(scratch/'papers',opener=Opener());service.set_auto_receive(False)
        service.refresh(threading.Event());service.download('0001-01-01',threading.Event(),lambda _:None,['p2a-qa'])
        library=scratch/'library';library.mkdir();(library/'Owned copy.pdf').write_bytes(original);service.set_library_root(library)
        report=dict(success=False,checks=[],paperSHA256=hashlib.sha256(original).hexdigest(),noticeSHA256=hashlib.sha256(note).hexdigest())
        w=Workbench(root/'frontend/dist',test=True,vocal_profile=scratch/'vocal',start_module='M18')
        w.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        w.papers_bridge.service=service;w.papers_bridge.export_picker=lambda _:str(scratch/'annotated.pdf')
        w.resize(1480,980);w.show()
        def pause(ms=100):
            loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
        def js(code):
            loop=QEventLoop();values=[];w.page.runJavaScript(code,lambda value:(values.append(value),loop.quit()))
            QTimer.singleShot(10000,loop.quit);loop.exec();assert values,'JS timed out';return values[0]
        def until(code,seconds=30):
            end=time.monotonic()+seconds
            while time.monotonic()<end:
                if js('Boolean('+code+')'): return
                pause()
            raise AssertionError(code+'\n'+str(js('document.body.innerText.slice(-1400)')))
        def click(text):
            assert js("(()=>{const e=[...document.querySelectorAll('button')].find(e=>e.textContent.trim()==="+json.dumps(text)+");if(!e||e.disabled)return false;e.click();return true})()"),text
        def fill(selector,text):
            assert js("(()=>{const e=document.querySelector("+json.dumps(selector)+");if(!e)return false;e.value="+json.dumps(text)+";e.dispatchEvent(new Event('input',{bubbles:true}));return true})()")
        def select(selector,value):
            js("(()=>{const e=document.querySelector("+json.dumps(selector)+");e.value="+json.dumps(value)+";e.dispatchEvent(new Event('change',{bubbles:true}))})()")
        try:
            until("document.querySelector('.paper-intro')&&document.querySelectorAll('.pdf-page').length===1&&document.querySelector('.paper-sheet')?.naturalWidth>0")
            assert js("document.querySelector('.source-tabs button').textContent.trim()==='服务器推荐 1'")
            assert not js("[...document.querySelectorAll('.pdf-controls button')].some(e=>e.textContent==='中文译文')")
            select('[aria-label="年份筛选"]','1999');assert js("!!document.querySelector('.paper-intro')")
            select('[aria-label="年份筛选"]','');report['checks'].append('Notice pinned outside date filters and counts; one real PDF, no fake translation/citation/license')
            js("[...document.querySelectorAll('.paper-card')].find(e=>e.textContent.includes('协同')).click()")
            until("document.querySelectorAll('.pdf-page').length===5&&document.querySelector('.paper-sheet')?.naturalWidth>0")
            report['toolbar']=js("document.querySelector('.pdf-toolbar')?.outerHTML")
            until("document.querySelector('.pdf-toolbar button[aria-controls]')")
            js("document.querySelector('.pdf-toolbar button[aria-controls]').click()")
            until("document.querySelector('.pdf-notes')&&document.querySelector('.pdf-notes button')?.disabled===false")
            assert js("(()=>{const n=document.querySelector('.pdf-notes').getBoundingClientRect(),v=document.querySelector('.paper-viewport').getBoundingClientRect();return n.width>=180&&n.right<=innerWidth+1&&v.width>400&&document.activeElement===document.querySelector('.pdf-notes')})()")
            click('添加本页笔记');fill('[aria-label="批注内容"]','真实 Qt 本页笔记');click('保存批注')
            until("document.querySelector('.pdf-note-card p')?.textContent==='真实 Qt 本页笔记'")
            assert service.annotations('p2a-qa','original')['items'][0]['quote']==''
            click('编辑');fill('[aria-label="批注内容"]','修改后的本页笔记');click('保存批注')
            until("document.querySelector('.pdf-note-card p')?.textContent==='修改后的本页笔记'")
            click('导出 PDF');js("document.querySelector('.export-check input').click()")
            click('选择位置并导出');until("!document.querySelector('.export-check')")
            a=PdfReader(scratch/'annotated.pdf').pages[0]['/Annots'][0].get_object()
            assert a['/Subtype']=='/Text' and '修改后的本页笔记' in a['/Contents']
            report['checks'].append('Batch note sidebar visible/focused, page note create/edit/native persist and standard PDF export')
            for theme in ('light','dark'):
                js("document.documentElement.dataset.theme="+json.dumps(theme));pause(180)
                w.view.grab().save(str(out/('notes-'+theme+'.png')))
            click('引用格式');until("!!document.querySelector('.paper-citation')")
            assert js("document.querySelector('.paper-citation').value.startsWith('FAN C, HENDERSON J M, MANNING C, et al.')")
            report['citationFonts']=js("[document.querySelector('.paper-citation'),document.querySelector('dialog[open]'),document.querySelector('.paper-citation').parentElement,document.documentElement].map(e=>getComputedStyle(e).fontFamily)")
            assert js("getComputedStyle(document.querySelector('.paper-citation')).fontFamily===getComputedStyle(document.querySelector('dialog[open]')).fontFamily")
            js("document.documentElement.style.setProperty('--font','\"Microsoft YaHei\"')")
            assert js("getComputedStyle(document.querySelector('.paper-citation')).fontFamily.includes('Microsoft YaHei')")
            w.view.grab().save(str(out/'citation-font.png'));click('关闭')
            report['checks'].append('Citation inherits configured app font dynamically and separates all four authors in correct order')
            click('中文译文');until("document.querySelectorAll('.pdf-page').length===5")
            select('[aria-label="页码"]','5');until("document.querySelector('[aria-label=页码]').value==='5'")
            from pypdf import PdfWriter
            writer=PdfWriter();writer.add_blank_page(595,842);writer.add_blank_page(595,842)
            buf=BytesIO();writer.write(buf);writer.close();revision=buf.getvalue()
            p['translation']=asset('p2a-qa/revision.pdf',revision)
            items['p2a-qa/revision.pdf']=revision;items['catalog.json']=json.dumps(catalog).encode()
            click('检查新论文')
            until("document.querySelectorAll('.pdf-page').length===2&&document.querySelector('[aria-label=页码]').value==='2'&&document.querySelector('.paper-sheet')?.naturalWidth>0")
            assert not js("document.querySelector('.pdf-warning[role=alert]')?.textContent")
            report['checks'].append('Refreshed translation hash remounts active PDF and clamps remembered page 5 to revised page count 2')
            js("[...document.querySelectorAll('.paper-card')].find(e=>e.textContent.includes('协同')).dispatchEvent(new MouseEvent('contextmenu',{bubbles:true,clientX:180,clientY:400}))")
            until("!!document.querySelector('[role=menuitem]')");click('删除本机副本')
            until("document.querySelector('dialog[open]')?.textContent.includes('删除本机副本')")
            click('删除本机副本');until("document.querySelectorAll('.paper-card').length===1")
            until("document.querySelectorAll('.pdf-page').length===1&&document.querySelector('.paper-sheet')?.naturalWidth>0")
            assert not js("document.querySelector('.pdf-warning[role=alert]')?.textContent")
            assert service.status()['papers'][0]['dismissed'] and not service._path(p['original']).exists()
            assert (service.root/('annotations-'+p['original']['sha256']+'.json')).exists()
            click('检查新论文');until("document.body.textContent.includes('本次检查无需下载')")
            assert not service._path(p['original']).exists()
            click('获取往期');until("document.querySelector('.archive-row input')?.disabled===false")
            js("document.querySelector('.archive-row input').click()");click('获取已选 1 篇')
            until("document.querySelector('.obtained-toggle')?.textContent.includes('1 篇')");click('关闭')
            js("[...document.querySelectorAll('.paper-card')].find(e=>e.textContent.includes('协同')).click()")
            until("document.querySelectorAll('.pdf-page').length===5");js("document.querySelector('.pdf-toolbar button[aria-controls]').click()")
            until("document.querySelector('.pdf-note-card p')?.textContent==='修改后的本页笔记'")
            click('删除');click('确认删除');until("document.querySelectorAll('.pdf-note-card').length===0")
            report['checks'].append('Context-menu recommendation deletion, automatic/manual check skips removed item, selected archive reacquisition restores note, note deletion works')
            js("document.querySelector('.source-tabs button:nth-child(2)').click()")
            until("document.querySelectorAll('.paper-card').length===1&&document.querySelector('.paper-sheet')?.naturalWidth>0")
            js("document.querySelector('.paper-card').dispatchEvent(new MouseEvent('contextmenu',{bubbles:true,clientX:180,clientY:400}))")
            until("!!document.querySelector('[role=menuitem]')");click('移入回收站');until("!!document.querySelector('dialog[open]')");click('移入回收站')
            until("document.querySelectorAll('.paper-card').length===0")
            assert not (library/'Owned copy.pdf').exists()
            report['checks'].append('Personal library native Windows recycle operation succeeds for owned copy; recommendation remains separate')
            report['success']=True
        finally:
            w.closing=True;w.close();w.page.deleteLater();app.processEvents()
            w.papers_bridge.pool.shutdown(wait=True);w.papers_bridge.network_pool.shutdown(wait=True)
            if w.service.process: w.service.process.wait(timeout=15)
            (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf8')
    print(out,flush=True)


if __name__=='__main__':main()
