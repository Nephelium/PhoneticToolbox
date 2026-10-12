"""Owned native M18 test. User-specified PDFs are copied; originals remain read-only."""
import hashlib
import argparse
import json
from pathlib import Path
import time
from workbench_source import configure
from verification_artifacts import verification_run


def main():
    configure()
    from PyQt6.QtCore import QEventLoop, QTimer, Qt, QPoint
    from PyQt6.QtWidgets import QApplication, QFileDialog
    from PyQt6.QtTest import QTest
    from ptb_desktop.host import Workbench, register_scheme
    from ptb_desktop.papers import PaperService, validate_catalog
    root = Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser();parser.add_argument('--sources',type=Path,required=True)
    options=parser.parse_args();sources=[Path(p) for p in json.loads(options.sources.read_text('utf8'))]
    assert len(sources)==4, 'Provide the four user-authorized PDF paths in a private local JSON file'
    original_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    register_scheme(); app = QApplication(['M18-R3-owned-QA'])
    with verification_run('m18', 'r3-qt', 'Actual Qt/source, live HTTPS catalogue, four authorized PDF copies, physical folder move, annotations and screenshots') as (out, scratch):
        report = {'success': False, 'checks': [], 'captures': [], 'sourceHashes': original_hashes}
        # M18 owns its JSON/cache state and does not require the acoustic job DB.
        w = Workbench(root/'frontend/dist', test=True, vocal_profile=scratch/'vocal', start_module='M18')
        w.papers_bridge.service = PaperService(scratch/'paper-state')
        w.papers_bridge.service.first_launch = '2026-10-07'
        w.papers_bridge.service.catalog = validate_catalog(json.loads((root/'output/paper-reading/m18-r3/publish/catalog.json').read_text('utf8')))
        w.showMaximized()
        def pause(ms=80):
            loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
        def js(code):
            loop=QEventLoop(); values=[];w.page.runJavaScript(code,lambda v:(values.append(v),loop.quit()));QTimer.singleShot(8000,loop.quit);loop.exec();assert values,'JavaScript timeout';return values[0]
        def until(code, seconds=45):
            end=time.monotonic()+seconds
            while time.monotonic()<end:
                if js(code):return
                pause()
            raise AssertionError(code+'\n'+str(js('document.body.innerText.slice(-1400)')))
        def click(text):
            code='[...document.querySelectorAll("button")].find(e=>e.offsetParent&&e.textContent.trim()==='+json.dumps(text)+')'
            until('!!('+code+') && !('+code+').disabled');js(code+'.click()');pause()
        def pointer(selector):
            r=js('(()=>{const r=document.querySelector('+json.dumps(selector)+').getBoundingClientRect();return [r.left+r.width/2,r.top+r.height/2]})()')
            target=w.view.focusProxy() or w.view;position=target.mapFromGlobal(w.view.mapToGlobal(QPoint(round(r[0]),round(r[1]))));QTest.mouseClick(target,Qt.MouseButton.LeftButton,pos=position);pause()
        def input(selector,value,event='input'):
            js('(()=>{const e=document.querySelector('+json.dumps(selector)+');e.value='+json.dumps(value)+';e.dispatchEvent(new Event('+json.dumps(event)+',{bubbles:true}));})()');pause()
        def shot(name):
            pause(220);w.view.grab();pause();pix=w.view.grab();pix.save(str(out/(name+'.png')))
            report['captures'].append({'name':name,'size':[pix.width(),pix.height()],'dpr':pix.devicePixelRatio(),'pageZoom':w.view.zoomFactor(),'maximized':w.isMaximized()})
        def check(text):report['checks'].append(text);print(text,flush=True)
        try:
            until('document.querySelectorAll(".paper-card").length===1')
            js('document.querySelector("[aria-label=收起侧栏]")?.click()');pause()
            click('获取往期');until('document.querySelectorAll(".archive-row:not(.obtained)").length===1 && !document.querySelector(".archive-row input").disabled')
            js('document.querySelector(".archive-row input").click()');shot('reader-archive')
            click('获取已选 1 篇');until('document.querySelectorAll(".archive-row:not(.obtained)").length===0')
            js('document.querySelector(".obtained-toggle input").click()');until('!!document.querySelector(".obtained input:disabled")')
            click('关闭');until('document.querySelectorAll(".pdf-page").length===13&&!!document.querySelector(".pdf-text-line")')
            assert js('[...document.querySelectorAll("button")].every(e=>e.textContent.trim()!=="适宽")')
            assert js('document.querySelector(".paper-card").getBoundingClientRect().height') == 34
            shot('reader-collapsed');check('live catalogue, checkbox-only download, acquired disabled, compact one-line rows and no duplicate fit button')
            pointer('.paper-fold');shot('reader-details');pointer('.paper-fold')
            # Native PDF selection and persisted mark, without writing source PDFs.
            until("""document.querySelectorAll('[data-page="1"] .pdf-text-line').length>3""")
            js("""(()=>{const span=document.querySelectorAll('[data-page="1"] .pdf-text-line')[2];const r=document.createRange();r.selectNodeContents(span);getSelection().removeAllRanges();getSelection().addRange(r);document.querySelector('.paper-viewport').dispatchEvent(new MouseEvent('mouseup',{bubbles:true}));})()""")
            click('荧光笔');until('!!document.querySelector(".pdf-mark.highlight")');click('批注 1');shot('reader-annotations');click('批注 1')
            click('引用格式');shot('reader-citations');click('关闭')
            click('导出 PDF');shot('reader-export');click('取消')
            # Fullscreen is a native pointer gesture so WebEngine grants it.
            js('[...document.querySelectorAll("button")].find(e=>e.textContent.trim()==="全屏阅读").setAttribute("data-fullscreen-test","true")')
            pointer('[data-fullscreen-test]');until('!!document.fullscreenElement');shot('reader-fullscreen');click('退出全屏');until('!document.fullscreenElement')
            click('论文解读');until('document.querySelectorAll(".pdf-page").length===6&&document.querySelector(".paper-sheet")?.alt.includes("论文解读")');shot('reader-analysis')
            js("""document.querySelector('.paper-viewport').scrollTop=document.querySelector('[data-page="5"]').offsetTop""");until("""document.querySelector('input[aria-label=页码]').value==='5'&&!!document.querySelector('[data-page="5"] .paper-sheet')""")
            assert js('document.querySelectorAll(".pdf-text-layer").length') <= 5
            check('actual original/highlight/fullscreen and six-page analysis; scroll page 5 mounts at most five text layers')
            click('6 小时');input('[aria-label="自动检查间隔"]',12);click('保存间隔');until('document.body.innerText.includes("12 小时")');click('12 小时');shot('reader-schedule');click('取消')
            pointer('.auto-switch input');until('document.body.innerText.includes("自动获取已关闭")');assert w.papers_bridge.service.preferences()['autoReceive'] is False
            check('native interval preference 12h and disable persisted, module has no periodic network timer')
            click('我的 PDF 0');until('document.body.innerText.includes("选择论文文件夹")')
            library_root=scratch/'我的论文';library_root.mkdir()
            # Exercise the real directory dialog; it belongs to this test window.
            timer=QTimer()
            def choose():
                for dialog in app.topLevelWidgets():
                    if isinstance(dialog,QFileDialog) and dialog.isVisible():
                        dialog.setDirectory(str(library_root));dialog.accept();timer.stop()
            timer.timeout.connect(choose);timer.start(100)
            click('选择论文文件夹');until('document.body.innerText.includes("更换文件夹")');timer.stop()
            w.papers_bridge.import_picker=lambda:[str(p) for p in sources]
            click('导入 PDF');click('选择 PDF');until('document.querySelectorAll(".paper-card").length===4&&!!document.querySelector(".paper-sheet")')
            assert js('document.querySelectorAll(".source-tabs button")[0].textContent.includes("1")')
            assert js('document.querySelectorAll(".pdf-controls")[0].textContent.trim()') == '本地 PDF'
            js('document.querySelector("[aria-label=新建收藏夹]").click()');input('[aria-label="收藏夹名称"]','方言元音');click('创建')
            js('[...document.querySelectorAll(".paper-card")].find(e=>e.textContent.includes("麦耘")).click()');until('document.querySelector(".paper-fold").textContent.includes("麦耘")&&!!document.querySelector(".paper-sheet")');shot('reader-local')
            old_id=w.papers_bridge.service.library_status()['documents'][0]['id']
            click('移至收藏夹');input('[aria-label="目标收藏夹"]','方言元音','change');click('确定');until('!document.querySelector("dialog[open]")&&!!document.querySelector(".paper-sheet")')
            assert (library_root/'方言元音'/sources[0].name).exists() and not (library_root/sources[0].name).exists()
            input('[aria-label="收藏夹筛选"]','方言元音','change');until('document.querySelectorAll(".paper-card").length===1');shot('reader-local-folder')
            reloaded=PaperService(scratch/'paper-state');assert len(reloaded.library_status()['documents'])==4
            assert reloaded.library_status()['folders']==['方言元音']
            assert all(hashlib.sha256(p.read_bytes()).hexdigest()==original_hashes[p.name] for p in sources)
            check('four authorized real PDFs copied, native root dialog, actual subfolder move, reload and unchanged source hashes; personal/recommended separated')
            # Measure repeated actual worker rendering without replacing its thread.
            p=w.papers_bridge.service.catalog['papers'][0];samples=[]
            for _ in range(3):
                start=time.perf_counter();result=w.papers_bridge.pool.submit(w.papers_bridge.service.render,p['id'],'original',1,1200).result(timeout=20);samples.append(round((time.perf_counter()-start)*1000,2))
            report['renderMs']=samples;assert result['pages']==13
            report['cachedDocuments']=len(w.papers_bridge.service._documents);report['cachedPages']=len(w.papers_bridge.service._pages)
            for size in [(1440,900),(1100,720)]:
                w.showNormal();w.resize(*size);pause(250)
                for theme in ('light','dark'):
                    js('document.documentElement.dataset.theme='+json.dumps(theme));pause(120);assert js('document.documentElement.scrollWidth<=innerWidth+1');shot(str(size[0])+'-'+theme)
            check('Windows Qt light/dark at two sizes, content/document caches and repeated worker render measured')
            click('帮助');until('document.querySelector(".manual-document")?.dataset.manualChapter==="m18"')
            assert js('document.querySelectorAll(".manual-document img").length') == 11
            js('document.querySelectorAll(".manual-document img").forEach(i=>i.loading="eager")')
            until('[...document.querySelectorAll(".manual-document img")].every(i=>i.complete&&i.naturalWidth>0)')
            assert js('document.querySelector(".manual-document").textContent.includes("默认 6 小时")&&document.querySelector(".manual-document").textContent.includes("麦耘")')
            shot('reader-manual');check('M18 help loads current folder/interval instructions and all eleven registered native screenshots')
            report['success']=True
        finally:
            report['sourceHashesUnchanged']=all(hashlib.sha256(p.read_bytes()).hexdigest()==original_hashes[p.name] for p in sources)
            (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n','utf8')
            process=w.service.process;w.closing=True;w.close();w.page.deleteLater();app.processEvents()
            try: process.wait(timeout=15)
            except Exception: pass
            w.papers_bridge.pool.shutdown(wait=True);w.papers_bridge.network_pool.shutdown(wait=True)
            report['serviceExit']=process.poll();(out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n','utf8');print(out,flush=True)


if __name__=='__main__':main()
