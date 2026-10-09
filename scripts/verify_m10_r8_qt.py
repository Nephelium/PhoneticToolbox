"""M10-R8 owned hidden Qt: real native geometry, metric view and readable controls."""
import json
import os
import sqlite3
import time
from pathlib import Path
from uuid import uuid4


def main():
    os.environ['QT_QPA_PLATFORM']='windows'
    os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS','--mute-audio')
    from PyQt6.QtCore import QEventLoop,QTimer,Qt,QPoint
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtTest import QTest
    from ptb_desktop.host import Workbench,register_scheme
    from ptb_worker.local_acoustic_files import initialize_local_files
    from ptb_desktop.vocal_tract.runtime import Runtime
    root=Path(__file__).resolve().parents[1]
    out=root/'output/validation/m10-r8'/('qt-'+uuid4().hex);out.mkdir(parents=True)
    db=out/'jobs.sqlite3'
    with sqlite3.connect((root/'output/validation/p06/local-state.sqlite3').as_uri()+'?mode=ro',uri=True) as source,sqlite3.connect(db) as target:
        assert not source.execute("SELECT 1 FROM jobs WHERE state IN ('queued','running','cancel_requested')").fetchone()
        source.backup(target)
    cache=out/'cache';cache.mkdir();initialize_local_files(cache)
    profile=out/'vocal'
    native=Runtime(root/'resources/vocal_tract/native',profile,playback_allowed=False)
    try:native.profile.save_frames([{'params':native.engine.presets[p],'name':p,'id':p,'duration':.2,'f0':150} for p in ['a','i','u']])
    finally:native.close()
    register_scheme();app=QApplication(['M10-R8-owned-QA'])
    w=Workbench(root/'frontend/dist',test=True,jobs_path=db,local_files_root=cache,vocal_profile=profile,start_module='M10')
    w.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen);w.resize(1440,900);w.show()
    report={'success':False,'checks':[],'layouts':[],'scope':'Windows hidden native Qt; isolated profile; no physical audio/DPI or EXE'}
    def pause(ms=160):
        loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
    def js(code):
        loop=QEventLoop();box=[];w.page.runJavaScript(code,lambda value:(box.append(value),loop.quit()));QTimer.singleShot(6000,loop.quit);loop.exec()
        assert box,'JavaScript timeout';return box[0]
    def local(code):return js('(()=>{const f=document.querySelector("iframe[title=声道工作台]"),d=f?.contentDocument,win=f?.contentWindow;if(!d)return null;'+code+'})()')
    def until(code,seconds=45):
        deadline=time.monotonic()+seconds
        while time.monotonic()<deadline:
            if local('return '+code):return
            pause(100)
        raise AssertionError(code)
    def settled():
        until('d.body.dataset.engineState==="ready"&&d.body.dataset.posePending==="false"');pause(180)
    def click(selector):
        assert local('const e=d.querySelector('+json.dumps(selector)+');if(!e||e.disabled)return false;e.click();return true;'),selector
        pause(100)
    def input_value(selector,value,event='input'):
        assert local('const e=d.querySelector('+json.dumps(selector)+');if(!e)return false;e.value='+json.dumps(value)+';e.dispatchEvent(new Event('+json.dumps(event)+',{bubbles:true}));return true;');settled()
    def host_click(text):
        assert js('(()=>{const e=[...document.querySelectorAll("button")].find(b=>b.offsetParent&&b.textContent.trim()==='+json.dumps(text)+');if(!e||e.disabled)return false;e.click();return true;})()'),text
        pause()
    def fonts(size):
        host_click('设置')
        assert js('[...document.querySelectorAll(".font-settings")].some(e=>e.offsetParent)')
        for label in ['正文基础字号','图表基础字号']:
            assert js('(()=>{const e=[...document.querySelectorAll('+json.dumps('input[aria-label="'+label+'"]')+')].find(e=>e.offsetParent);if(!e)return false;e.value='+str(size)+';e.dispatchEvent(new Event("input",{bubbles:true}));return true;})()')
        host_click('应用字体')
        until('d.documentElement.style.getPropertyValue("--body-size")==='+json.dumps(str(size)+'px'))
        until('[...document.querySelectorAll(".font-settings [role=status]")].some(e=>e.offsetParent&&e.textContent.includes("字体已应用"))')
        assert js('(()=>{const e=[...document.querySelectorAll('+json.dumps('button[aria-label="关闭 设置"]')+')].find(e=>e.offsetParent);if(!e)return false;e.click();return true;})()')
        pause(250)
        until('d.querySelector("#sectionChart").clientWidth>0&&d.querySelector("#sectionChart").clientHeight>0')
    def metric():
        value=local('const e=d.querySelector("#sectionChart");return {scale:+e.dataset.pixelsPerCm,width:e.clientWidth,height:e.clientHeight};')
        assert value['width']>0 and value['height']>0 and value['scale']>0,value
        assert local('const a=d.querySelector("#sectionChart").getBoundingClientRect(),b=d.querySelector("#sectionHandles").getBoundingClientRect();return Math.abs(a.x-b.x)<.1&&Math.abs(a.y-b.y)<.1&&Math.abs(a.width-b.width)<.1&&Math.abs(a.height-b.height)<.1;'),'canvas/handle viewport mismatch'
        return value
    def text_floor():
        small=local('return [...d.querySelectorAll("body *")].filter(e=>e.namespaceURI==="http://www.w3.org/1999/xhtml"&&e.getClientRects().length&&[...e.childNodes].some(n=>n.nodeType===3&&n.textContent.trim())).filter(e=>parseFloat(win.getComputedStyle(e).fontSize)<11.999).map(e=>({tag:e.tagName,id:e.id,text:e.textContent.slice(0,60),font:win.getComputedStyle(e).fontSize}));')
        assert small==[],small
    try:
        settled()
        local('win.qaErrors=[];win.addEventListener("error",e=>win.qaErrors.push(e.message));win.qaFonts=[];const proto=win.CanvasRenderingContext2D.prototype,fill=proto.fillText;proto.fillText=function(...args){win.qaFonts.push({id:this.canvas.id,font:this.font,text:String(args[0])});return fill.apply(this,args);};')
        fonts(14);input_value('#columnLayout','three','change')
        fixed=metric();assert fixed['scale']>0,fixed
        records=[]
        for name in ['a','i','u','e','o','E','y','2','l','n','t']:
            click('[data-preset="'+name+'"]');settled();assert metric()==fixed,(name,metric(),fixed)
            records.append({'preset':name,**metric()})
        click('#resetPose');settled()
        # All 129 native sections use the same transform, including empty ends.
        samples=local('const e=d.querySelector("#areaChart"),out=[];e.dispatchEvent(new win.KeyboardEvent("keydown",{key:"Home",bubbles:true}));for(let i=0;i<129;i++){const c=d.querySelector("#sectionChart");out.push({section:+d.querySelector("#sliceRange").value,scale:+c.dataset.pixelsPerCm,width:c.clientWidth,height:c.clientHeight,handles:d.querySelectorAll("#sectionHandles g").length});if(i<128)e.dispatchEvent(new win.KeyboardEvent("keydown",{key:"ArrowRight",bubbles:true}));}return out;')
        assert len(samples)==129 and [r['section'] for r in samples]==list(range(129))
        assert all({k:r[k] for k in ['scale','width','height']}==fixed for r in samples)
        report['checks'].append('11 actual native presets and all 129 sections retain exact pixels/cm, viewport size and origin; empty sections included')
        report['native_metric_samples']=records;report['section_samples']=samples
        click('[data-side-region="2"]');settled();until('d.querySelectorAll("#sectionHandles g").length===2')
        baseline=local('return +d.querySelector("#p-TS2").value')
        point=local('const r=d.querySelector("#sectionHandles g").getBoundingClientRect(),fr=f.getBoundingClientRect();return {x:fr.x+r.x+r.width/2,y:fr.y+r.y+r.height/2};')
        target=w.view.focusProxy() or w.view
        start=target.mapFromGlobal(w.view.mapToGlobal(QPoint(round(point['x']),round(point['y']))));end=QPoint(start.x(),start.y()-8)
        QTest.mouseMove(target,start);QTest.mousePress(target,Qt.MouseButton.LeftButton,pos=start);QTest.mouseMove(target,end,delay=60);QTest.mouseRelease(target,Qt.MouseButton.LeftButton,pos=end);settled()
        changed=local('return +d.querySelector("#p-TS2").value');assert changed!=baseline,(baseline,changed)
        assert metric()==fixed
        click('#undoButton');settled();assert local('return +d.querySelector("#p-TS2").value')==baseline
        local('d.querySelector("#sectionHandles g").focus();return true;');QTest.keyClick(target,Qt.Key.Key_Up);settled()
        assert local('return +d.querySelector("#p-TS2").value')!=baseline and metric()==fixed
        click('#undoButton');settled()
        click('#tab-organs');input_value('#lipWidthRange',143);assert metric()==fixed
        report['checks'].append('real Qt side-handle pointer drag and ArrowUp update TS2; undo restores exact value; lip width update keeps metric scale')
        matrix=[(1440,900,'three',s) for s in [10,14,18,24]]+[(1000,700,'two',s) for s in [14,24]]+[(1920,1080,'three',14)]
        for theme in ['light','dark']:
            js('document.documentElement.dataset.theme='+json.dumps(theme));pause(200)
            for width,height,columns,size in matrix:
                w.resize(width,height);fonts(size);input_value('#columnLayout',columns,'change');pause(250)
                for tab in ['organs','sound','motion']:
                    click('#tab-'+tab);text_floor()
                    spacing=local('return [...d.querySelectorAll("#panel-'+tab+' .parameter")].filter(e=>e.getClientRects().length&&e.querySelector("input[type=range]")).map(e=>{const r=e.querySelector("input[type=range]").getBoundingClientRect(),l=e.querySelector("label").getBoundingClientRect(),o=e.querySelector("output")?.getBoundingClientRect();return {gap:r.top-Math.max(l.bottom,o?.bottom||0),height:r.height,top:e.getBoundingClientRect().top};});')
                    assert all(r['gap']<=3.1 and abs(r['height']-18)<.1 for r in spacing),spacing
                    if tab=='organs':
                        panel=local('const e=d.querySelector("#panel-organs");e.scrollTop=0;return {overflow:win.getComputedStyle(e).overflowY,groupGap:win.getComputedStyle(d.querySelector(".parameter-group")).rowGap};')
                        assert panel['overflow']=='auto' and panel['groupGap']=='4px',panel
                        w.view.grab().save(str(out/f'{theme}-{width}-{columns}-{size}px-organs.png'))
                    if width==1440 and size==14 and theme=='light':w.view.grab().save(str(out/f'light-{tab}.png'))
                click('#tab-organs');click('#savePreset');text_floor();click('#presetInputButton');text_floor();click('#presetCancel')
                actual=metric();assert actual['scale']>0,actual
                assert local('return [...d.querySelectorAll(".charts .chart-heading h3")].every(e=>{const a=e.getBoundingClientRect(),b=e.closest("section").getBoundingClientRect();return a.right<=b.right+1;});'),'chart heading overlaps adjacent column'
                # These two drastically different shapes cannot alter scale.
                click('[data-preset=i]');settled();assert metric()==actual
                click('[data-preset=a]');settled();assert metric()==actual
                drawn=local('return win.qaFonts.splice(0);');assert drawn and any(r['text']=='1 cm' for r in drawn)
                assert all(float(r['font'].split('px',1)[0])>=12 for r in drawn),drawn[:5]
                report['layouts'].append({'theme':theme,'width':width,'height':height,'columns':columns,'body_and_figure_size':size,'metric':actual})
        report['checks'].append('14 light/dark/font/layout combinations: body/figure 10,14,18,24 px; all rendered HTML text and canvas labels >=12 px; every slider gap <=3.1 px and height 18 px; fixed scale survives pose changes')
        assert local('return win.qaErrors')==[]
        report['success']=True
    except Exception as error:
        report['failure']=str(error);w.view.grab().save(str(out/'failure.png'));raise
    finally:
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf8')
        w.closing=True;w.close();w.page.deleteLater();app.processEvents()
        print(json.dumps({'out':str(out),'success':report['success'],'checks':report['checks']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
