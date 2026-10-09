"""M10-R10 owned hidden Qt: header removal, shared control sizing and source dialog without a layout picker."""
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
    out=root/'output/validation/m10-r10'/('qt-'+uuid4().hex);out.mkdir(parents=True)
    db=out/'jobs.sqlite3'
    with sqlite3.connect((root/'output/validation/p06/local-state.sqlite3').as_uri()+'?mode=ro',uri=True) as source,sqlite3.connect(db) as target:
        assert not source.execute("SELECT 1 FROM jobs WHERE state IN ('queued','running','cancel_requested')").fetchone()
        source.backup(target)
    cache=out/'cache';cache.mkdir();initialize_local_files(cache)
    profile=out/'vocal'
    native=Runtime(root/'resources/vocal_tract/native',profile,playback_allowed=False)
    try:native.profile.save_frames([{'params':native.engine.presets[p],'name':p,'id':p,'duration':.2,'f0':150} for p in ['a','i','u']])
    finally:native.close()
    register_scheme();app=QApplication(['M10-R10-owned-QA'])
    w=Workbench(root/'frontend/dist',test=True,jobs_path=db,local_files_root=cache,vocal_profile=profile,start_module='M10')
    w.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen);w.resize(1440,900);w.show()
    report={'success':False,'checks':[],'layouts':[],'scope':'Windows hidden native Qt; isolated profile; no physical audio/DPI or EXE'}
    def pause(ms=160):
        loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
    def js(code):
        loop=QEventLoop();box=[];w.page.runJavaScript(code,lambda value:(box.append(value),loop.quit()));QTimer.singleShot(6000,loop.quit);loop.exec()
        assert box,'JavaScript timeout';return box[0]
    def local(code):return js('(()=>{const f=[...document.querySelectorAll("iframe")].find(e=>e.getAttribute("src")?.includes("vocal-tract/")),d=f?.contentDocument,win=f?.contentWindow;if(!d)return null;'+code+'})()')
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
    def host_until(code,seconds=30):
        deadline=time.monotonic()+seconds
        while time.monotonic()<deadline:
            if js(code):return
            pause(100)
        raise AssertionError(code)
    def pointer_click(selector):
        local('d.querySelector('+json.dumps(selector)+').scrollIntoView({block:"nearest",inline:"nearest"});return true;');pause(150)
        point=local('const r=d.querySelector('+json.dumps(selector)+').getBoundingClientRect(),fr=f.getBoundingClientRect();return {x:fr.x+r.x+r.width/2,y:fr.y+r.y+r.height/2};')
        target=w.view.focusProxy() or w.view
        p=target.mapFromGlobal(w.view.mapToGlobal(QPoint(round(point['x']),round(point['y']))))
        QTest.mouseMove(target,p);QTest.mouseClick(target,Qt.MouseButton.LeftButton,pos=p);pause(180)
    def button_style():
        return local('const e=d.querySelector("#aboutButton"),s=win.getComputedStyle(e),r=e.getBoundingClientRect();return {font:s.fontSize,minHeight:s.minHeight,padding:s.padding,border:s.borderTopWidth,radius:s.borderRadius,width:r.width,height:r.height};')
    def reload_columns(value):
        local('localStorage.setItem("m10-columns",'+json.dumps(value)+');d.body.dataset.qaReload="old";f.contentWindow.location.reload();return true;')
        until('d.body?.dataset.qaReload!=="old"&&d.body?.dataset.engineState==="ready"&&d.body?.dataset.posePending==="false"');pause(200)
        assert local('return d.documentElement.dataset.columns')==value
    try:
        settled();fonts(14)
        assert local('return !d.querySelector("#resetPose,#columnLayout,.layout-picker")')
        baseline=button_style();assert baseline['font']=='14px' and baseline['minHeight']=='30px' and baseline['padding']=='4px 9px' and float(baseline['border'].replace('px',''))>0 and baseline['radius']=='6px',baseline
        pointer_click('#aboutButton');until('d.querySelector("#aboutDialog").open')
        assert local('return !d.querySelector("#aboutDialog label,#aboutDialog select")&&!!d.querySelector("#sharedReferences")&&d.querySelectorAll("#aboutDialog a").length===6;')
        w.view.grab().save(str(out/'light-about.png'));click('#closeAbout')
        click('[data-preset=i]');settled();assert local('return d.querySelector("#sectionChart").dataset.pixelsPerCm')
        click('#undoButton');settled();click('#tab-sound');click('#tab-motion');click('#tab-organs')
        click('#savePreset');assert local('return d.querySelector("#presetDialog").open&&d.querySelectorAll(".preset-consonants button").length===186');click('#presetCancel')
        report['checks'].append('real Qt pointer opens model/source dialog; reset and layout label/select absent; full source text, six links and shared-reference entry retained; native preset/undo, three panels and full IPA popup still work')
        for value in ['two','three','auto']:reload_columns(value)
        report['checks'].append('old two/three preferences survive real iframe reload; default/explicit auto still applies without a selector or null listener')
        for theme in ['light','dark']:
            js('document.documentElement.dataset.theme='+json.dumps(theme));pause(200)
            for width,height,size in [(1920,1080,14),(1440,900,14),(1000,700,14),(1440,900,24)]:
                w.resize(width,height);fonts(size);pause(220)
                assert local('return !d.querySelector("#resetPose,#columnLayout,.layout-picker")')
                geometry=local('const e=d.querySelector("#aboutButton"),r=e.getBoundingClientRect(),h=e.closest(".inspector-title").getBoundingClientRect(),t=d.querySelector(".inspector-tabs").getBoundingClientRect();return {left:r.left,right:r.right,top:r.top,bottom:r.bottom,headerLeft:h.left,headerRight:h.right,headerTop:h.top,headerBottom:h.bottom,tabTop:t.top,viewportWidth:win.innerWidth,viewportHeight:win.innerHeight};')
                assert geometry['left']>=geometry['headerLeft']-.5 and geometry['right']<=geometry['headerRight']+.5 and geometry['top']>=geometry['headerTop']-.5 and geometry['bottom']<=geometry['headerBottom']+.5 and geometry['bottom']<=geometry['tabTop']+.5,geometry
                assert local('const a=d.querySelector(".inspector-title .eyebrow").getBoundingClientRect(),b=d.querySelector("#aboutButton").getBoundingClientRect();return a.right+4<=b.left;')
                style=button_style();assert float(style['font'].replace('px',''))==size and style['height']>=30
                w.view.grab().save(str(out/f'{theme}-{width}-{size}px-header.png'))
                pointer_click('#aboutButton');until('d.querySelector("#aboutDialog").open')
                assert local('return !d.querySelector("#aboutDialog label,#aboutDialog select")')
                w.view.grab().save(str(out/f'{theme}-{width}-{size}px-about.png'))
                click('#closeAbout');report['layouts'].append({'theme':theme,'windowWidth':width,'windowHeight':height,'bodySize':size,'button':style,'geometry':geometry})
        report['checks'].append('eight light/dark/14px/24px layouts: enlarged button fits header without clipping or overlap; pointer opens source dialog in all windows, layout controls absent')
        w.resize(1440,900);fonts(14);baseline=button_style()
        host_click('声学参数合成');host_until('[...document.querySelectorAll(".module-toolbar button")].some(e=>e.offsetParent&&e.textContent.trim()==="方法与来源")')
        reference=js('(()=>{const e=[...document.querySelectorAll(".module-toolbar button")].find(e=>e.offsetParent&&e.textContent.trim()==="方法与来源"),s=getComputedStyle(e),r=e.getBoundingClientRect();return {font:s.fontSize,minHeight:s.minHeight,padding:s.padding,border:s.borderTopWidth,radius:s.borderRadius,width:r.width,height:r.height};})()')
        assert all(baseline[k]==reference[k] for k in ['font','minHeight','padding','border','radius']) and all(abs(baseline[k]-reference[k])<.1 for k in ['width','height']),(baseline,reference)
        report['source_button']=baseline;report['reference_button']=reference
        report['checks'].append('actual M06 Method/source button and M10 Model/source button match computed font, padding, border, corner, minimum height and exact five-character button width/height')
        report['success']=True
    except Exception as error:
        report['failure']=str(error);w.view.grab().save(str(out/'failure.png'));raise
    finally:
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf8')
        w.closing=True;w.close();w.page.deleteLater();app.processEvents()
        print(json.dumps({'out':str(out),'success':report['success'],'checks':report['checks']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
