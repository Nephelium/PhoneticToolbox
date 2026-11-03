"""Actual M18 Qt/PDF workflows with frozen, hash-verified transport fixtures.

The two recommendation papers use their existing canonical publication bytes.
Four personal PDFs are read only from the explicitly authorized local manifest.
This recipe verifies UI/native algorithms; it does not claim live network QA.
"""
from __future__ import annotations
import argparse,hashlib,json,shutil,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from manual.capture_a_support import Capture
from verification_artifacts import verification_run


def run(args,out,scratch):
    c=Capture(args,out,scratch,'m18','paper-reading')
    from PyQt6.QtCore import QPoint,Qt
    from PyQt6.QtTest import QTest
    from pypdf import PdfWriter
    from ptb_desktop.papers import PaperService,validate_catalog
    source_paths=[Path(p) for p in json.loads(args.sources.read_text(encoding='utf-8'))]
    baseline=json.loads((ROOT/'output/validation/m18/r3-qt-4022315f0b1646bc9e79013db04720c4/report.json').read_text(encoding='utf-8'))['sourceHashes']
    assert len(source_paths)==4 and all(p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==baseline[p.name] for p in source_paths)
    content=ROOT/'output/paper-reading/2408.05641v1'
    catalogue_path=content/'publish/catalog.json';catalogue_bytes=catalogue_path.read_bytes();catalogue=validate_catalog(json.loads(catalogue_bytes))
    assert len(catalogue['papers'])==2
    transport={'catalog.json':catalogue_bytes}
    for paper in catalogue['papers']:
        base=content/'publish' if paper['id']=='arxiv-2408-05641-v1' else ROOT/'output/paper-reading/m18-r3/publish'
        for kind in ('original','translation','analysisPdf'):
            asset=paper[kind];data=(base/asset['path']).read_bytes()
            assert len(data)==asset['size'] and hashlib.sha256(data).hexdigest()==asset['sha256'] and data.startswith(b'%PDF-')
            transport[asset['path']]=data
    service=PaperService(scratch/'papers');service.first_launch='2026-10-07';service.catalog=catalogue
    def read_fixture(path,maximum,cancel):
        if cancel.is_set():raise RuntimeError('Cancelled owned fixture transport')
        data=transport[path];assert len(data)<=maximum
        return data
    service._read=read_fixture
    service.recommendation_folder('协同发音');service.collect_recommendation('arxiv-2408-05641-v1','协同发音')
    c.window.papers_bridge.service=service
    library=scratch/'我的论文';library.mkdir();choice={'root':library}
    c.window.papers_bridge.root_picker=lambda:str(choice['root'])
    c.window.papers_bridge.import_picker=lambda:[str(p) for p in source_paths]
    host_path=ROOT/'desktop/src/ptb_desktop/host.py';c.sources[str(host_path)]=hashlib.sha256(host_path.read_bytes()).hexdigest()
    c.sources[str(catalogue_path)]=hashlib.sha256(catalogue_bytes).hexdigest()
    c.report.update(scope='Actual maximized native Windows Qt; offline frozen catalogue/PDF transport fixtures; actual verification, rendering, annotations, PDF import and filesystem operations; no live-network claim',sourceHashes=baseline,transportFixtureHashes={k:hashlib.sha256(v).hexdigest() for k,v in transport.items()})
    def click(text):c.click(text)
    def hit(selector):
        c.until('!!document.querySelector('+json.dumps(selector,ensure_ascii=False)+')');c.js('document.querySelector('+json.dumps(selector,ensure_ascii=False)+').click()');c.pause()
    def aria(text):return '[aria-label='+json.dumps(c.ui.label(text),ensure_ascii=False)+']'
    def select(text,value):c.fill(aria(text),value,'change')
    def paper(paper_id):
        target=next(p for p in catalogue['papers'] if p['id']==paper_id)
        expr='[...document.querySelectorAll(".paper-card")].find(e=>e.title.startsWith('+json.dumps(target['titleZh']+'\n',ensure_ascii=False)+'))'
        c.until('!!('+expr+')');c.js(expr+'.click()');c.pause()
    def ready(pages):c.until('document.querySelectorAll(".pdf-page").length==='+str(pages)+'&&!!document.querySelector(".pdf-text-line")',60)
    def shot(name):
        if not args.scene or name in args.scene:c.shot(name)
    def pointer(text):
        name=c.ui.label(text);expr='[...document.querySelectorAll("button")].find(e=>e.offsetParent&&e.textContent.trim()==='+json.dumps(name,ensure_ascii=False)+')'
        c.until('!!('+expr+')&&!('+expr+').disabled')
        box=c.js('(()=>{const r='+expr+'.getBoundingClientRect();return [r.x+r.width/2,r.y+r.height/2]})()')
        target=c.window.view.focusProxy() or c.window.view;pos=target.mapFromGlobal(c.window.view.mapToGlobal(QPoint(round(box[0]),round(box[1]))))
        QTest.mouseClick(target,Qt.MouseButton.LeftButton,pos=pos);c.pause()
    try:
        c.until('!!document.querySelector(".paper-reading")');c.prepare()
        click('获取往期');c.until('document.querySelectorAll(".archive-row:not(.obtained)").length===2')
        hit('.archive-row input');shot('reader-archive')
        # The actual selected-download API verifies and writes all six PDFs.
        c.js('document.querySelectorAll(".archive-row input").forEach(e=>{if(!e.checked&&!e.disabled)e.click()})');c.pause()
        hit('dialog[open] .primary');c.until('document.querySelectorAll(".archive-row:not(.obtained)").length===0');click('关闭')
        paper('arxiv-2610-00735-v1');ready(13);shot('reader-collapsed')
        hit('.paper-fold');shot('reader-details');hit('.paper-fold')
        c.until('document.querySelectorAll(".pdf-page:first-child .pdf-text-line").length>3')
        c.js('(()=>{const span=document.querySelectorAll(\'[data-page="1"] .pdf-text-line\')[2],r=document.createRange();r.selectNodeContents(span);getSelection().removeAllRanges();getSelection().addRange(r);document.querySelector(".paper-viewport").dispatchEvent(new MouseEvent("mouseup",{bubbles:true}));})()')
        click('荧光笔');c.until('!!document.querySelector(".pdf-mark.highlight")')
        hit('.pdf-controls button[aria-controls]');shot('reader-annotations');hit('.pdf-controls button[aria-controls]')
        click('引用格式');shot('reader-citations');click('关闭')
        click('导出 PDF');shot('reader-export');click('取消')
        was_maximized=c.window.isMaximized();pointer('全屏阅读');c.until('!!document.fullscreenElement')
        assert c.window.isFullScreen() and was_maximized
        c.pause(600);image=c.window.view.grab().toImage();raw=[image.width(),image.height()];assert raw in ([2560,1600],[2561,1601]) and image.devicePixelRatio()==1.5
        if raw!=[2560,1600]:image=image.copy(0,0,2560,1600)
        file=out/'reader-fullscreen.png';assert image.save(str(file))
        full=dict(name='reader-fullscreen',file=file.name,image=[2560,1600],raw_image=raw,maximized=c.window.isMaximized(),fullscreen=True,documentFullscreen=bool(c.js('!!document.fullscreenElement')),wasMaximized=was_maximized,fullscreenProof={'qt':c.window.isFullScreen(),'document':bool(c.js('!!document.fullscreenElement'))},sha256=hashlib.sha256(file.read_bytes()).hexdigest(),**c.ui.metrics(c.js));c.report['captures'].append(full)
        pointer('退出全屏');c.until('!document.fullscreenElement')
        deadline=time.monotonic()+5;states=[]
        while time.monotonic()<deadline and not c.window.isMaximized():
            states.append(dict(maximized=c.window.isMaximized(),fullscreen=c.window.isFullScreen(),state=c.window.windowState().value));c.pause(200)
        c.report['fullscreenExitStates']=states;assert c.window.isMaximized(),states
        stable=[]
        for _ in range(8):
            c.pause(200);stable.append(dict(maximized=c.window.isMaximized(),fullscreen=c.window.isFullScreen(),state=c.window.windowState().value));assert c.window.isMaximized() and not c.window.isFullScreen()
        c.report['fullscreenExitStableStates']=stable
        full['fullscreenProof']['restoredMaximized']=True;full['fullscreenExitRestoredMaximized']=c.window.isMaximized()
        click('论文解读');ready(6);shot('reader-analysis')
        hit('.interval-button');c.fill(aria('自动检查间隔'),12);click('保存间隔');c.until('document.querySelector(".interval-button").textContent.trim().startsWith("12")');hit('.interval-button');shot('reader-schedule');click('取消')
        paper('arxiv-2408-05641-v1');click('原文');ready(5);hit('.paper-fold');shot('reader-new-details');hit('.paper-fold')
        click('检查新论文');message=c.ui.key('modules.paper-reading.text.536c73779d7f',{'v0':2});c.until('document.body.innerText.includes('+json.dumps(message,ensure_ascii=False)+')');shot('reader-content-check');hit('.paper-notice button')
        select('收藏夹筛选','协同发音');hit(aria('删除收藏夹'));shot('reader-remove-recommendation');click('取消')
        hit('.source-tabs button:nth-child(2)');click('选择论文文件夹');c.until('document.querySelector(".paper-actions").textContent.includes('+json.dumps(c.ui.label('更换文件夹'),ensure_ascii=False)+')')
        click('导入 PDF');click('选择 PDF');c.until('document.querySelectorAll(".paper-card").length===4')
        hit(aria('新建收藏夹'));c.fill(aria('收藏夹名称'),'方言元音');click('创建')
        expr='[...document.querySelectorAll(".paper-card")].find(e=>e.textContent.includes("麦耘"))';c.until('!!('+expr+')');c.js(expr+'.click()');c.until('!!document.querySelector(".paper-sheet")');shot('reader-local')
        click('移至收藏夹');select('目标收藏夹','方言元音');click('确定');c.until('!document.querySelector("dialog[open]")&&!!document.querySelector(".paper-sheet")');select('收藏夹筛选','方言元音');c.until('document.querySelectorAll(".paper-card").length===1');shot('reader-local-folder')
        assert (library/'方言元音'/source_paths[0].name).exists() and not (library/source_paths[0].name).exists()
        # Recreate the original owned 72-file/long-folder demonstration fixture.
        large=scratch/'个人论文库与协同发音研究资料';large.mkdir();long_folder='方言元音与协同发音研究收藏夹及需要核查的实验语音学资料';folder=large/long_folder;folder.mkdir()
        shutil.copyfile(content/'original.pdf',folder/'协同发音定量分析.pdf');(folder/'说明.txt').write_text('Preserve all files',encoding='utf-8');(folder/'补充材料').mkdir();(folder/'补充材料/info.txt').write_text('Preserve nested folder',encoding='utf-8');(large/'说明.txt').write_text('Collision fixture',encoding='utf-8')
        blank=PdfWriter();blank.add_blank_page(width=595,height=842)
        for i in range(71):
            with (large/(f'{i:03d}_本地论文测试与特别长的文件题名仅用于检查目录显示.pdf')).open('wb') as stream:blank.write(stream)
        # Cancel and same-root selection retain the current collection filter.
        c.window.papers_bridge.root_picker=lambda:'';click('更换文件夹');c.until('!document.querySelector(".paper-actions button").disabled');assert c.js('document.querySelector('+json.dumps(aria('收藏夹筛选'),ensure_ascii=False)+').value')=='方言元音'
        c.window.papers_bridge.root_picker=lambda:str(choice['root']);click('更换文件夹');c.until('!document.querySelector(".paper-actions button").disabled');assert c.js('document.querySelector('+json.dumps(aria('收藏夹筛选'),ensure_ascii=False)+').value')=='方言元音'
        choice['root']=large;click('更换文件夹');c.until('document.querySelectorAll(".paper-card").length===72');select('收藏夹筛选',long_folder);c.until('document.querySelectorAll(".paper-card").length===1');hit('.paper-card');ready(5)
        hit(aria('删除收藏夹'));shot('reader-remove-local');click('删除收藏夹并保留文件');c.until('!!document.querySelector("[role=alert]")');shot('reader-remove-collision')
        assert (folder/'协同发音定量分析.pdf').exists() and (folder/'说明.txt').exists()
        (large/'说明.txt').unlink();click('删除收藏夹并保留文件');c.until('!document.querySelector("dialog[open]")');ready(5);shot('reader-remove-local-result')
        assert not folder.exists() and (large/'补充材料/info.txt').read_text(encoding='utf-8')=='Preserve nested folder'
        assert hashlib.sha256((large/'协同发音定量分析.pdf').read_bytes()).hexdigest()==hashlib.sha256((content/'original.pdf').read_bytes()).hexdigest()
        c.report.update(success=True,checks=['Two canonical recommendation PDFs and translations/commentaries obtained through actual hash/length validation and durable writes; offline transport only','Real PDF text selection, highlight persistence, citation/export UI, native fullscreen and restore','Four explicitly authorized originals copied and moved through actual library controls; source hashes unchanged','Cancelled and same-root selections retain collection filter; successful changed root resets stale filters','Owned 72-file long-folder collision refuses every move; retry preserves PDF/text/nested folder'],sourceHashesUnchanged=all(hashlib.sha256(p.read_bytes()).hexdigest()==baseline[p.name] for p in source_paths))
    finally:c.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--locale',choices=('en','ja'),required=True);parser.add_argument('--frontend-dist',type=Path,default=ROOT/'frontend/dist');parser.add_argument('--scene',action='append',default=[]);parser.add_argument('--sources',type=Path,default=ROOT/'local-data/m18-r3-test-sources.json');args=parser.parse_args();args.frontend_dist=args.frontend_dist.resolve(strict=True)
    with verification_run('m18-localized',args.locale,'Actual native Qt/PDF algorithms, frozen verified offline transport fixtures, four specifically authorized read-only PDF originals and owned 72-file library') as (out,scratch):run(args,out,scratch)
