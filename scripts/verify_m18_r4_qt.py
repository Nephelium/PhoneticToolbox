"""Owned native paper UI: long local names, switching, safe folder deletion."""
import hashlib,json,shutil,time
from pathlib import Path
from workbench_source import configure
from verification_artifacts import verification_run

def main():
    configure()
    from PyQt6.QtCore import QEventLoop,QTimer,Qt
    from PyQt6.QtWidgets import QApplication
    from pypdf import PdfWriter
    from ptb_desktop.host import Workbench,register_scheme
    from ptb_desktop.papers import PaperService,validate_catalog
    root=Path(__file__).resolve().parents[1]
    content=root/'output/paper-reading/2408.05641v1'
    register_scheme();app=QApplication(['M18-R4-owned-QA'])
    with verification_run('m18','r4-qt',recipe='Qt/native source; 72 generated/authorized public PDF fixtures, long folder names, no user profile or private original touched') as (out,scratch):
        report={'success':False,'checks':[],'captures':[]}
        library=scratch/'个人论文库与协同发音研究资料';library.mkdir()
        long_folder='方言元音与协同发音研究收藏夹及需要核查的实验语音学资料'
        folder=library/long_folder;folder.mkdir()
        shutil.copyfile(content/'original.pdf',folder/'协同发音定量分析.pdf')
        (folder/'说明.txt').write_text('Preserve all files','utf8')
        (folder/'补充材料').mkdir();(folder/'补充材料/info.txt').write_text('Preserve nested folder','utf8')
        (library/'说明.txt').write_text('Collision fixture','utf8')
        blank=PdfWriter();blank.add_blank_page(width=595,height=842)
        for n in range(71):
            with (library/(f'{n:03d}_本地论文测试与特别长的文件题名仅用于检查目录显示.pdf')).open('wb') as stream:blank.write(stream)
        service=PaperService(scratch/'papers');service.first_launch='2026-10-07'
        service.catalog=validate_catalog(json.loads((content/'publish/catalog.json').read_text('utf8')))
        (service.root/'files').mkdir(exist_ok=True)
        for p in service.catalog['papers']:
            for key in ('original','translation','analysisPdf'):
                asset=p[key];base=content/'publish' if p['id']=='arxiv-2408-05641-v1' else root/'output/paper-reading/m18-r3/publish'
                shutil.copyfile(base/asset['path'],service.root/'files'/(asset['sha256']+'.pdf'))
        service.set_library_root(library);service.recommendation_folder('协同发音')
        service.collect_recommendation('arxiv-2408-05641-v1','协同发音')
        w=Workbench(root/'frontend/dist',test=True,vocal_profile=scratch/'vocal',start_module='M18')
        w.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen,True)
        w.papers_bridge.service=service
        w.resize(w.screen().geometry().width(),w.screen().geometry().height());w.showMaximized()
        def pause(ms=100):
            loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
        def js(code):
            loop=QEventLoop();values=[];w.page.runJavaScript(code,lambda value:(values.append(value),loop.quit()));QTimer.singleShot(10000,loop.quit);loop.exec()
            assert values,'JavaScript callback timed out';return values[0]
        def until(code,seconds=50):
            end=time.monotonic()+seconds
            while time.monotonic()<end:
                if js(code):return
                pause()
            raise AssertionError(code+'\n'+str(js('document.body.innerText.slice(-2000)')))
        def click(label):
            code='[...document.querySelectorAll("button")].find(e=>e.offsetParent&&e.textContent.trim()==='+json.dumps(label)+')'
            until('!!('+code+')&&!('+code+').disabled');js(code+'.click()');pause()
        def hit(selector):
            until('!!document.querySelector('+json.dumps(selector)+')');js('document.querySelector('+json.dumps(selector)+').click()');pause()
        def select(selector,value):
            js('(()=>{const e=document.querySelector('+json.dumps(selector)+');e.value='+json.dumps(value)+';e.dispatchEvent(new Event("change",{bubbles:true}));})()');pause()
        def shot(name):
            pause(300);w.grab();pause();pix=w.grab();original=[pix.width(),pix.height()]
            if original==[2561,1601]:pix=pix.copy(0,0,2560,1600)
            assert [pix.width(),pix.height()]==[2560,1600], original
            pix.save(str(out/(name+'.png')))
            report['captures'].append(dict(name=name,size=[pix.width(),pix.height()],originalSize=original,dpr=pix.devicePixelRatio(),pageZoom=w.view.zoomFactor(),maximized=w.isMaximized(),theme='light'))
        def check(text):report['checks'].append(text);print(text,flush=True)
        def layout():
            value=js('''(()=>{const a=document.querySelector('.paper-library').getBoundingClientRect(),r=document.querySelector('.paper-reader').getBoundingClientRect();const controls=[...document.querySelectorAll('.library-tools input,.library-tools select,.library-tools button')];return {asideWidth:a.width,readerLeft:r.left,asideRight:a.right,within:controls.every(e=>{const b=e.getBoundingClientRect();return b.left>=a.left&&b.right<=a.right+1&&b.width>0}),icons:controls.filter(e=>e.tagName==='BUTTON').every(e=>{const b=e.getBoundingClientRect();return document.elementFromPoint(b.left+b.width/2,b.top+b.height/2)?.closest('button')===e}),fileIcons:document.querySelectorAll('.paper-card svg.paper-file-icon').length,rows:document.querySelectorAll('.paper-card').length}})()''')
            assert value['within'] and value['icons'] and value['asideRight']<=value['readerLeft']+1,value
            assert value['fileIcons']==value['rows'],value
            report.setdefault('layouts',[]).append(value)
        try:
            until('document.querySelectorAll(".paper-card").length===2')
            js('document.documentElement.dataset.theme="light";document.querySelector("[aria-label=收起侧栏]")?.click()');pause()
            hit('.paper-card');until('document.querySelectorAll(".pdf-page").length===5&&!!document.querySelector(".pdf-text-line")')
            hit('.paper-fold');shot('reader-new-details');hit('.paper-fold')
            click('中文译文');until('document.querySelectorAll(".pdf-page").length===11&&!!document.querySelector(".pdf-text-line")')
            click('论文解读');until('document.querySelectorAll(".pdf-page").length===6&&!!document.querySelector(".pdf-text-line")')
            click('检查新论文');until('document.body.innerText.includes("服务器目录共 2 篇，本次检查无需下载。")');shot('reader-content-check');click('收起')
            check('Live HTTPS check reports catalogue count; new fixed-version 5/11/6-page PDFs and text layers open')
            select('[aria-label="收藏夹筛选"]','协同发音');hit('[aria-label="删除收藏夹"]');shot('reader-remove-recommendation');click('取消')
            assert service.preferences()['recommendationFolders']==['协同发音']
            hit('[aria-label="删除收藏夹"]');click('删除收藏夹并保留文件');until('!document.querySelector("dialog[open]")&&document.body.innerText.includes("论文及批注保留")')
            assert service.status()['papers'][0]['downloaded'];assert js('document.querySelector("[aria-label=删除收藏夹]").disabled')
            check('Recommendation deletion/cancellation preserves downloaded assets and forbids deleting uncollected/root categories')
            click('我的 PDF 72');until('document.querySelectorAll(".paper-card").length===72');layout();shot('reader-local-sidebar')
            for n in range(3):
                click('服务器推荐 2');layout();click('我的 PDF 72');layout()
            original_render=service.render
            def slow_native_render(*args,**kwargs):
                time.sleep(.08)
                return original_render(*args,**kwargs)
            service.render=slow_native_render
            try:
                for n in range(30):
                    label='服务器推荐 2' if n%2==0 else '我的 PDF 72'
                    js('[...document.querySelectorAll(".source-tabs button")].find(e=>e.textContent.trim()==='+json.dumps(label)+').click()');pause(10)
                click('服务器推荐 2');until('document.querySelectorAll(".pdf-page").length===5&&!!document.querySelector(".pdf-text-line")')
                assert js('!document.body.innerText.includes("论文服务繁忙")')
                click('检查新论文');until('document.body.innerText.includes("服务器目录共 2 篇，本次检查无需下载。")')
                click('收起');click('我的 PDF 72')
                check('Thirty rapid switches while actual native renders incur an 80ms test delay cancel obsolete PDF/annotation work; reader and live catalogue check recover without busy errors')
            finally:service.render=original_render
            for size in [(1440,900),(1100,720)]:
                w.showNormal();w.resize(*size);pause(250)
                for theme in ('light','dark'):
                    js('document.documentElement.dataset.theme='+json.dumps(theme));pause();layout()
                    assert js('document.documentElement.scrollWidth<=innerWidth+1')
            w.resize(w.screen().geometry().width(),w.screen().geometry().height());w.showMaximized();js('document.documentElement.dataset.theme="light"');pause()
            select('[aria-label="收藏夹筛选"]',long_folder);until('document.querySelectorAll(".paper-card").length===1');hit('.paper-card')
            until('document.querySelectorAll(".pdf-page").length===5&&!!document.querySelector(".pdf-text-line")');layout()
            hit('[aria-label="删除收藏夹"]');shot('reader-remove-local');click('删除收藏夹并保留文件')
            until('document.querySelector("dialog[open]")?.textContent.includes("未移动文件")');shot('reader-remove-collision')
            assert (folder/'协同发音定量分析.pdf').exists() and (folder/'说明.txt').exists()
            # Only this owned conflict fixture is removed to exercise a safe retry.
            (library/'说明.txt').unlink();click('删除收藏夹并保留文件')
            until('!document.querySelector("dialog[open]")&&document.body.innerText.includes("已移到上一级")')
            assert not folder.exists() and (library/'补充材料/info.txt').read_text('utf8')=='Preserve nested folder'
            assert hashlib.sha256((library/'协同发音定量分析.pdf').read_bytes()).digest()==hashlib.sha256((content/'original.pdf').read_bytes()).digest()
            until('document.querySelectorAll(".pdf-page").length===5&&!!document.querySelector(".pdf-text-line")')
            assert js('document.querySelector("[aria-label=删除收藏夹]").disabled');shot('reader-remove-local-result')
            check('72-file local library/long folder names and path stay in sidebar across repeated source switches, light/dark and two window sizes; icons remain clickable')
            check('Native local-folder conflict blocks every move; retry preserves PDF/text/nested folder, follows relocated identity and retains a readable PDF')
            report['success']=True
        finally:
            w.closing=True;w.close();w.page.deleteLater();app.processEvents()
            w.papers_bridge.pool.shutdown(wait=True);w.papers_bridge.network_pool.shutdown(wait=True)
            if w.service.process:
                try:w.service.process.wait(timeout=15)
                except Exception:pass
                report['serviceExit']=w.service.process.poll()
            (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf8');print(out,flush=True)

if __name__=='__main__':main()
