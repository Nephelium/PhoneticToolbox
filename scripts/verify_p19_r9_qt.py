"""Owned hidden Windows host: resize presentation, native fonts and body size.

No physical monitor/DWM or frozen EXE claim. Own database/profile and no audio.
"""
import argparse
import json
import os
import sqlite3
import time
from pathlib import Path
from uuid import uuid4


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--baseline',action='store_true');args=parser.parse_args()
    os.environ['QT_QPA_PLATFORM']='windows'
    os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS','--mute-audio')
    from PyQt6.QtCore import QEventLoop,QTimer,Qt,qVersion
    from PyQt6.QtWidgets import QApplication
    from ptb_desktop.host import Workbench,register_scheme
    from ptb_worker.local_acoustic_files import initialize_local_files
    root=Path(__file__).resolve().parents[1];out=root/'output/validation/p19-r9'/('qt-'+('baseline-' if args.baseline else '')+uuid4().hex)
    out.mkdir(parents=True);db=out/'jobs.sqlite3'
    with sqlite3.connect((root/'output/validation/p06/local-state.sqlite3').as_uri()+'?mode=ro',uri=True) as source,sqlite3.connect(db) as target:
        assert not source.execute("SELECT 1 FROM jobs WHERE state IN ('queued','running','cancel_requested')").fetchone();source.backup(target)
    cache=out/'cache';cache.mkdir();initialize_local_files(cache)
    register_scheme();app=QApplication(['P19-R9-owned-QA'])
    w=Workbench(root/'frontend/dist',test=True,jobs_path=db,local_files_root=cache,vocal_profile=out/'vocal')
    w.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen);w.show()
    report={'success':False,'baseline':args.baseline,'qt':qVersion(),'flags':os.environ.get('QTWEBENGINE_CHROMIUM_FLAGS'),'samples':[],'checks':[],'terminations':[],'scope':'Windows hidden native Qt; physical Windows compositor and EXE not exercised'}
    w.page.renderProcessTerminated.connect(lambda status,code:report['terminations'].append([status.name,code]))
    def pause(ms=100):
        loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
    def js(code):
        loop=QEventLoop();box=[];w.page.runJavaScript(code,lambda value:(box.append(value),loop.quit()));QTimer.singleShot(6000,loop.quit);loop.exec()
        assert box,'JavaScript timeout';return box[0]
    def until(code):
        end=time.monotonic()+40
        while time.monotonic()<end:
            if js(code):return
            pause()
        raise AssertionError(code)
    def click(text):
        assert js('(()=>{const b=[...document.querySelectorAll("button")].find(b=>b.offsetParent&&b.textContent.trim()==='+json.dumps(text)+');if(!b)return false;b.click();return true;})()');pause()
    try:
        until('!!document.querySelector(".home-page")&&document.fonts.status==="loaded"');pause(500)
        renderer=w.page.renderProcessPid();js('window.qaFrames=0;function tick(){qaFrames++;requestAnimationFrame(tick)};tick();window.qaCanvas=document.createElement("canvas");window.qaGL=qaCanvas.getContext("webgl");')
        report['webgl_available']=js('!!qaGL&&!qaGL.isContextLost()')
        for theme,label in [('light','浅色'),('dark','深色')]:
            click('设置');click(label)
            if not args.baseline:
                color=js('getComputedStyle(document.documentElement).getPropertyValue("--app").trim()')
                assert w.page.backgroundColor().name()==color and w.page.backgroundColor().alpha()==255
            for cycle in range(4):
                for state in ('maximized','normal'):
                    start=time.monotonic()
                    if state=='maximized':w.showMaximized();w.resize(w.screen().availableGeometry().size())
                    else:w.showNormal();w.resize(1200+cycle*40,800)
                    for delay in [16,80,250,500]:
                        remain=delay-(time.monotonic()-start)*1000
                        if remain>0:pause(round(remain))
                        shot=w.view.grab().toImage()
                        pixels=[shot.pixelColor(x,y) for x in range(0,shot.width(),max(1,shot.width()//32)) for y in range(0,shot.height(),max(1,shot.height()//24))]
                        colors=len({p.rgb() for p in pixels});black=sum(max(p.red(),p.green(),p.blue())<8 for p in pixels)/len(pixels)
                        sample={'mode':theme,'cycle':cycle,'state':state,'target_ms':delay,'actual_ms':round((time.monotonic()-start)*1000),'black_fraction':black,'colors':colors}
                        report['samples'].append(sample);assert black<.95 and colors>10,sample
                        if cycle==0 and delay==500:shot.save(str(out/(theme+'-'+state+'.png')))
                    assert w.page.renderProcessPid()==renderer and js('qaFrames>0')
            click('首页')
        report['checks'].append('16 maximize/restore transitions, 64 nonblack snapshots, same renderer and responsive JavaScript')
        if not args.baseline:
            click('设置');click('读取本机字体列表');until('document.querySelector(".font-settings [role=status]")?.textContent.includes("已读取")')
            click('浅色');js('document.querySelector(".font-follow input").click()');pause()
            font_controls=js('([...document.querySelectorAll(".font-family-select select")].map(e=>({label:e.getAttribute("aria-label"),value:e.value,options:e.options.length})))')
            assert len(font_controls)==5 and all(e['options']>100 for e in font_controls),font_controls;report['fonts']=font_controls
            js('(()=>{const e=document.querySelector("input[aria-label=正文基础字号]");e.value=18;e.dispatchEvent(new Event("input",{bubbles:true}));})()');click('应用字体')
            until('getComputedStyle(document.documentElement).fontSize==="18px"')
            assert js('getComputedStyle(document.querySelector(".nav-item")).fontSize===getComputedStyle(document.querySelector(".tab-wrap>button")).fontSize')
            click('声道工作台');until('!!document.querySelector("iframe[title=声道工作台]")?.contentDocument?.querySelector(".workspace")')
            until('getComputedStyle(document.querySelector("iframe[title=声道工作台]").contentDocument.documentElement).fontSize==="18px"')
            report['m10']=js('(()=>{const d=document.querySelector("iframe[title=声道工作台]").contentDocument;return {root:getComputedStyle(d.documentElement).fontSize,body:getComputedStyle(d.body).fontSize,parameter:getComputedStyle(d.querySelector(".parameter")).fontSize,webgl:!!d.querySelector("#viewport canvas")};})()')
            assert report['m10']['body']=='18px' and report['m10']['parameter']=='18px';report['checks'].append('actual Qt font database, five native full lists, body size applied and M10 iframe synchronized')
            until('document.querySelector("iframe[title=声道工作台]").contentDocument.querySelector("#loadIndicator").hidden')
            js('document.querySelector("iframe[title=声道工作台]").contentDocument.querySelector("#sagittalButton").click()');pause(300)
            js('(()=>{const win=document.querySelector("iframe[title=声道工作台]").contentWindow,p=win.CanvasRenderingContext2D.prototype,stroke=p.stroke;win.qaStrokes={};p.stroke=function(...args){const id=this.canvas.id;(win.qaStrokes[id]??=[]).push(this.strokeStyle);return stroke.apply(this,args)};})()')
            model_before=js('([...document.querySelector("iframe[title=声道工作台]").contentDocument.querySelectorAll("#viewport svg [data-organ]")].map(e=>({organ:e.dataset.organ,fill:e.getAttribute("fill"),stroke:e.getAttribute("stroke")})))')
            report['model_before']=model_before
            # Existing scene.js already uses a brighter selected-organ outline
            # in dark mode. Preserve that original adaptation as well.
            model_dark=[{**item,'stroke':{'#93595f':'#e8a0a8','#b98587':'#cf8994'}.get(item['stroke'],item['stroke'])} for item in model_before]
            report['m10_palettes']=[]
            for palette in ['codex','absolutely','catppuccin']:
                for theme,label in [('light','浅色'),('dark','深色')]:
                    click('设置');click(label)
                    js('document.querySelector("#palette-choice").click()');pause()
                    js('document.querySelector('+json.dumps('[data-palette-option="'+palette+'"]')+').click()');pause();click('声道工作台');pause(400)
                    metrics=js('(()=>{const d=document.querySelector("iframe[title=声道工作台]").contentDocument,w=d.defaultView,s=getComputedStyle(d.documentElement),c=d.createElement("i");d.body.append(c);c.style.color=s.getPropertyValue("--accent");const accent=getComputedStyle(c).color;c.remove();return {palette:'+json.dumps(palette)+',theme:'+json.dumps(theme)+',accent:s.getPropertyValue("--accent").trim(),accentRgb:accent,green:s.getPropertyValue("--green").trim(),slider:getComputedStyle(d.querySelector(".parameter input")).accentColor,strokes:w.qaStrokes,model:[...d.querySelectorAll("#viewport svg [data-organ]")].map(e=>({organ:e.dataset.organ,fill:e.getAttribute("fill"),stroke:e.getAttribute("stroke")}))};})()')
                    assert metrics['green']==metrics['accent'] and metrics['slider']==metrics['accentRgb'],metrics
                    for chart in ['areaChart','spectrumChart','sectionChart']:assert metrics['accent'] in metrics['strokes'].get(chart,[]),(chart,metrics)
                    report['m10_palettes'].append(metrics)
                    assert metrics['model']==(model_dark if theme=='dark' else model_before) and model_before
                    w.view.grab().save(str(out/(palette+'-'+theme+'-m10.png')))
                    js('document.querySelector("iframe[title=声道工作台]").contentWindow.qaStrokes={}')
            report['checks'].append('three palettes x two modes: real acoustic canvas strokes and slider colours match accent, anatomical model tissue colours unchanged')
        assert not report['terminations'];report['success']=True
    finally:
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf8')
        w.closing=True;w.close();w.page.deleteLater();app.processEvents();print(json.dumps({'out':str(out),'success':report['success'],'checks':report['checks']},ensure_ascii=False))


if __name__=='__main__':main()
