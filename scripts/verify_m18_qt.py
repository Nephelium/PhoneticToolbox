"""M18 real Qt + public HTTPS distribution, using a temporary owned profile."""
import json
import argparse
import time
from pathlib import Path
from uuid import uuid4
from workbench_source import configure


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--history',action='store_true');args=parser.parse_args()
    configure()
    from PyQt6.QtCore import QEventLoop,QTimer,Qt,QPoint
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtTest import QTest
    from ptb_desktop.host import Workbench,register_scheme
    root=Path(__file__).resolve().parents[1]
    out=root/'output/validation/m18'/('qt-'+uuid4().hex);out.mkdir(parents=True)
    register_scheme();app=QApplication(['M18-owned-QA'])
    report={'success':False,'checks':[],'layouts':[]}
    w=Workbench(root/'frontend/dist',test=True,vocal_profile=out/'vocal',start_module='M18')
    if args.history:
        # Simulated later first launch; production date remains local calendar.
        w.papers_bridge.service.first_launch='2026-10-08'
    w.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen);w.resize(1440,900);w.show()
    def pause(ms=100):
        loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
    def js(code):
        loop=QEventLoop();values=[];w.page.runJavaScript(code,lambda v:(values.append(v),loop.quit()));QTimer.singleShot(6000,loop.quit);loop.exec()
        assert values,'JS timeout';return values[0]
    def until(code,seconds=45):
        deadline=time.monotonic()+seconds
        while time.monotonic()<deadline:
            if js(code):return
            pause()
        raise AssertionError(code+'\n'+str(js('document.body.innerText.slice(-1600)')))
    def pointer(selector):
        point=js('(()=>{const r=document.querySelector('+json.dumps(selector)+').getBoundingClientRect();return [r.left+r.width/2,r.top+r.height/2]})()')
        target=w.view.focusProxy() or w.view
        pos=target.mapFromGlobal(w.view.mapToGlobal(QPoint(round(point[0]),round(point[1]))))
        QTest.mouseClick(target,Qt.MouseButton.LeftButton,pos=pos);pause(120)
    def click(text):
        selector='[...document.querySelectorAll(".paper-reading button,dialog button")].find(e=>e.offsetParent&&e.textContent.trim()==='+json.dumps(text)+')'
        until('!!('+selector+')');js(selector+'.setAttribute("data-qa-click","true")');pointer('[data-qa-click=true]');js('document.querySelector("[data-qa-click=true]")?.removeAttribute("data-qa-click")')
    try:
        if args.history:
            until('document.body.innerText.includes("默认获取 2026-10-08")&&!document.querySelector(".paper-notice")')
            assert not list(w.papers_bridge.service.files.iterdir())
            click('获取往期');until('document.querySelector("dialog")?.textContent.includes("共 1 篇，待下载 1 篇")')
            click('获取 1 篇');until('document.querySelector("dialog")?.textContent.includes("待下载 0 篇")');click('关闭')
            assert w.papers_bridge.service.first_launch=='2026-10-08'
            report['checks'].append('simulated next-day first launch downloads no back catalogue; history modal shows 1 pending paper and real retrieval leaves first date unchanged')
        until('document.querySelector(".paper-sheet")?.complete&&document.querySelector(".paper-sheet")?.naturalWidth>0')
        assert w.papers_bridge.service.status()['papers'][0]['downloaded']
        assert js('document.querySelector(".paper-license").textContent.includes("CC-BY-4.0")')
        report['checks'].append('new profile native first-date creation, real HTTPS catalogue and 2 checksum-verified PDFs, automatic current-date retrieval')
        click('收起导读');pointer('button[aria-label="下一页"]')
        until('document.querySelector(".paper-sheet")?.alt.includes("第 2 页")')
        click('中文译文');until('document.querySelector(".paper-sheet")?.alt.includes("中文译文")&&document.querySelector(".paper-sheet")?.complete')
        click('本页文字');until('document.querySelector(".paper-text")?.textContent.includes("未经原作者审校")')
        report['checks'].append('real native pointer: next page, Chinese PDF and native selectable/copyable text including licence and adaptation notice')
        click('本页文字');click('原文');until('document.querySelector(".paper-sheet")?.alt.includes("原文 第 2 页")')
        report['checks'].append('language switch retains separate per-document reading positions')
        click('获取往期');until('document.querySelector("dialog")?.textContent.includes("共 1 篇，待下载 0 篇")');click('关闭')
        for width,height in [(1440,900),(1100,720)]:
            w.resize(width,height)
            for theme in ('light','dark'):
                js('document.documentElement.dataset.theme='+json.dumps(theme));pause(200)
                for language in ('原文','中文译文'):
                    click(language);until('document.querySelector(".paper-sheet")?.complete&&document.querySelector(".paper-sheet")?.naturalWidth>0')
                    metrics=js('(()=>{const r=document.querySelector(".paper-reading").getBoundingClientRect(),v=document.querySelector(".paper-viewport").getBoundingClientRect();return {width:r.width,height:r.height,viewportHeight:v.height,scroll:document.documentElement.scrollWidth,window:innerWidth}})()')
                    assert metrics['viewportHeight']>160 and metrics['scroll']<=metrics['window']+1,metrics
                    w.view.grab();pause(200);w.view.grab().save(str(out/f'{width}-{theme}-{language}.png'))
                    report['layouts'].append({'size':[width,height],'theme':theme,'language':language,**metrics})
        # Offline: block this isolated client's network only, then exercise real
        # cached PDF parsing after a failed refresh.
        class Offline:
            def open(self,*a,**kw):raise OSError('owned offline simulation')
        w.papers_bridge.service.opener=Offline();click('检查新论文')
        until('document.querySelector(".paper-notice")?.textContent.includes("离线阅读")')
        click('原文');until('document.querySelector(".paper-sheet")?.alt.includes("原文")&&document.querySelector(".paper-sheet")?.complete')
        report['checks'].append('failed network refresh retains catalogue and renders actual cached original PDF')
        report['success']=True
    finally:
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf8')
        service=w.service.process
        w.closing=True;w.close();w.page.deleteLater();app.processEvents()
        report['serviceExit']=service.poll()
        assert service.poll()==0
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf8')
        print(json.dumps({'success':report['success'],'output':str(out)},ensure_ascii=True),flush=True)


if __name__=='__main__':main()
