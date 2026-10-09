"""Real Qt, owned profile/window, native pointer input; no physical audio."""
import json
import os
import sys
import time
from pathlib import Path
from uuid import uuid4

ROOT=Path(__file__).resolve().parents[1]
os.environ['PYTHONPATH']=str(ROOT/'packages/phonetic_core/src')+os.pathsep+os.environ.get('PYTHONPATH','')
sys.path.insert(0,str(ROOT/'packages/phonetic_core/src'))
os.environ['QT_QPA_PLATFORM']='windows'
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS','--mute-audio')


def main():
    from PyQt6.QtCore import QEventLoop,QTimer,Qt,QPoint
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtTest import QTest
    from ptb_desktop.host import Workbench,register_scheme
    out=ROOT/'output/validation/m10-r11'/('qt-'+uuid4().hex[:8]);out.mkdir(parents=True)
    register_scheme();app=QApplication(['M10-R11-owned-QA'])
    w=Workbench(ROOT/'frontend/dist',test=True,vocal_profile=out/'profile',vocal_resources=ROOT/'resources/vocal_tract/native',start_module='M10')
    w.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen);w.resize(1680,1000);w.show()
    report={'success':False,'checks':[],'out':str(out)}
    def pause(ms=120):
        loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
    def js(code):
        loop=QEventLoop();box=[];w.page.runJavaScript(code,lambda value:(box.append(value),loop.quit()));QTimer.singleShot(8000,loop.quit);loop.exec()
        assert box,'JS timeout';return box[0]
    def local(code):
        return js('(()=>{const f=[...document.querySelectorAll("iframe")].find(e=>e.getAttribute("src")?.includes("vocal-tract/")),d=f?.contentDocument,win=f?.contentWindow;if(!d)return null;'+code+'})()')
    def until(code,seconds=45):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            if local('return '+code):return
            pause(80)
        raise AssertionError(code)
    def settled():
        until('d.body.dataset.engineState==="ready"&&d.body.dataset.posePending==="false"');pause(80)
        errors=local('return win.qaRenderErrors??[]')
        assert not errors,errors
    def click(selector):
        assert local('const e=d.querySelector('+json.dumps(selector)+');if(!e||e.disabled)return false;e.click();return true;'),selector
        pause(80);settled()
    def value(selector,v):
        assert local('const e=d.querySelector('+json.dumps(selector)+');if(!e)return false;e.value='+json.dumps(v)+';e.dispatchEvent(new Event("input",{bubbles:true}));e.dispatchEvent(new Event("change",{bubbles:true}));return true;'),selector
        settled()
    def shot(name):
        pause(180);w.view.grab().save(str(out/(name+'.png')))
    def state(code):return local('const v=win.qaViewer,s=v.state;'+code)
    def drag(selector,dx,dy):
        point=local('const r=d.querySelector('+json.dumps(selector)+').getBoundingClientRect(),fr=f.getBoundingClientRect();return {x:fr.x+r.x+r.width/2,y:fr.y+r.y+r.height/2};')
        target=w.view.focusProxy() or w.view
        p=target.mapFromGlobal(w.view.mapToGlobal(QPoint(round(point['x']),round(point['y']))))
        QTest.mouseMove(target,p);QTest.mousePress(target,Qt.MouseButton.LeftButton,pos=p)
        for step in range(1,13):
            q=p+QPoint(round(dx*step/12),round(dy*step/12));QTest.mouseMove(target,q,25);pause(30)
        QTest.mouseRelease(target,Qt.MouseButton.LeftButton,pos=q);settled()
    try:
        settled()
        local("const script=d.createElement('script');script.type='module';script.textContent=\"import {viewer} from './app.js';window.qaViewer=viewer;window.qaRenderErrors=[];const update=viewer.update.bind(viewer);viewer.update=s=>{try{return update(s);}catch(e){window.qaRenderErrors.push({error:String(e),stack:e.stack,params:s.params,larynx:s.larynx_height});throw e;}};\";d.body.append(script);return true;")
        until('!!win.qaViewer?.state')
        assert state('return s.geometry_version')=='m10/3'
        js('document.documentElement.dataset.theme="light"');pause(200)
        click('[data-mode=overlay]');click('[data-select-organ=velum]');click('#keepVowel')
        # Small increments straddle the old moving branch's discrete changes.
        positions=[]
        for vo in [0,.01,.03,.10,.25,.50,.70,.72,.74,.76,1.,1.25,1.5]:
            value('#portRange',vo)
            positions.append(state('return {vo:s.nasal.port_area,profile:v.palate.profile,gap:v.palate.gap};'))
            if vo in [0,.25,.5,.74,1.,1.5]:shot('velum-'+str(vo))
        report['velum']=positions;report['checks'].append('native VO incremental sequence renders')
        click('[data-preset=a]');click('[data-select-organ=tongue]')
        value('#p-TTX',7.0);shot('tongue-protrusion')
        assert state('return s.limited[10]')>6.8
        click('[data-preset=a]');before=state('return s.limited.slice(12,14)')
        drag('[data-handle=blade]',25,-18);after=state('return s.limited.slice(12,14)')
        assert after[0]>before[0] and after[1]>before[1],(before,after)
        report['checks'].append({'native_blade_drag':[before,after]});shot('blade-drag')
        click('[data-select-organ=velum]');old=state('return {hyoid:s.contours.lower_cover[4],glottis:s.contours.lower_cover[0]}')
        value('#larynxHeight',.7);new=state('return {hyoid:s.contours.lower_cover[4],glottis:s.contours.lower_cover[0]}')
        assert old['hyoid']==new['hyoid'] and abs(new['glottis'][1]-old['glottis'][1]-.7)<1e-4
        drag('[data-handle=larynx]',0,15)
        shifted=state('return s.larynx_height');assert shifted<.7
        click('#undoButton');assert abs(state('return s.larynx_height')-.7)<1e-6
        click('#redoButton');assert abs(state('return s.larynx_height')-shifted)<1e-6
        value('#p-TTX',6.7)
        saved_pose=state('return {params:s.params,larynx_height:s.larynx_height}')
        click('#savePreset');click('[data-ipa="a"]');until('!d.querySelector("#presetDialog").open')
        disk=json.loads((out/'profile/presets.json').read_text('utf-8'))['presets'][0]
        assert disk['larynx_height']==saved_pose['larynx_height'] and disk['params']==saved_pose['params']
        click('[data-preset=u]');assert state('return s.larynx_height')==0
        click('#customPreset');click('[data-ipa="a"]');settled()
        assert state('return {params:s.params,larynx_height:s.larynx_height}')==saved_pose
        click('#tab-motion');click('#captureFrame')
        until('d.querySelectorAll(".pose-card").length===1')
        end=time.monotonic()+5
        while not (out/'profile/keyframes.json').exists() and time.monotonic()<end:pause(80)
        frame=json.loads((out/'profile/keyframes.json').read_text('utf-8'))['frames'][0]
        assert frame['larynx_height']==saved_pose['larynx_height'] and frame['params']==saved_pose['params']
        click('#tab-organs')
        report['checks'].append('larynx native drag, undo/redo and saved extended tongue/larynx preset roundtrip')
        shot('independent-larynx');report['checks'].append({'independent_larynx':[old,new]})
        click('[data-preset=a]')
        click('#threeButton')
        assert not state('return v.fullModel')
        for mode in ['organs','airway','overlay']:
            click('[data-mode='+mode+']')
            assert state('return [...v.meshes.values(),v.softPalate,v.head,v.airway].every(m=>(Array.isArray(m.material)?m.material:[m.material]).every(t=>t.clippingPlanes?.length===1))')
            shot('half-'+mode)
        click('#fullModelToggle')
        assert state('return v.fullModel&&!v.palateSection.visible&&!v.tongueSection.visible&&v.airway.material.every(m=>m.clippingPlanes.length===0)')
        shot('full-overlay');click('#fullModelToggle')
        js('document.documentElement.dataset.theme="dark"');pause(200);shot('half-dark')
        report['checks'].append('three modes clip all native tissue, head and oral/nasal air; explicit full option removes clipping and cut caps')
        click('#sagittalButton');click('[data-preset=a]');value('#portRange',1.5)
        assert state('return s.uvula_contact_lift')>.6
        shot('velum-contact-area')
        for vo in [.01,.02,.25,.4,.75,1,1.5]:
            value('#portRange',vo)
            assert state('return v.airPaths.some(p=>Math.min(...p.map(x=>x[1]))<-6&&Math.max(...p.map(x=>x[1]))>4)'),f'open oral/nasal section disconnected at {vo}'
            state('v.zoom=2.8;v.pan=[-2.9,.4];v.draw2D();return true;')
            assert local('return d.querySelector("[data-organ=velum]").dataset.style==="head-background"&&d.querySelector("[data-organ=velum]").getAttribute("fill")==="none"')
            if vo in [.02,.25,.4,1,1.5]:shot('velum-closeup-'+str(vo))
            state('v.zoom=1;v.pan=[0,0];v.draw2D();return true;')
        report['checks'].append('native uvula contact and 2.8x continuous oral-to-nasal section at seven openings; background-colored fixed uvula')
        click('[data-preset=a]');value('#p-LD',-1)
        closed=state('return s.airway_sections.findIndex((x,i)=>i>20&&x.area<1e-7)')
        assert closed>=0;value('#sliceRange',closed)
        assert local('return Number(d.querySelector("#areaChart").dataset.geometricArea)')<1e-7
        shot('closure-area');report['checks'].append('closed native geometric section reaches zero on area plot at the same selected station')
        click('[data-preset=l]');value('#sliceRange',109)
        assert local('return d.querySelector("#sectionStatus").textContent.includes("侧面仍有通路")')
        assert local('return Number(d.querySelector("#areaChart").dataset.geometricArea)')>.01
        shot('lateral-contact-area')
        for width,height,size,theme in [(1280,800,14,'light'),(1280,800,24,'dark'),(1920,1080,18,'light')]:
            w.resize(width,height);pause(150)
            js('document.documentElement.dataset.theme='+json.dumps(theme))
            local('d.documentElement.style.fontSize='+json.dumps(str(size)+'px')+';return true;')
            pause(250)
            assert local('return d.documentElement.scrollWidth<=d.documentElement.clientWidth+1')
            assert local('return d.querySelector("#larynxControl").scrollWidth<=d.querySelector("#larynxControl").clientWidth+1')
            assert local('return [...d.querySelectorAll("#presets button")].every(e=>e.clientHeight>=parseFloat(win.getComputedStyle(e).fontSize)*1.5)')
            shot(f'layout-{width}-{size}-{theme}')
        report['checks'].append('lateral-contact positive area and narrow/large-font/wide layouts')
        report['success']=True
    except Exception as exc:
        report['failure']=repr(exc);shot('failure');raise
    finally:
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf-8')
        w.closing=True;w.close();w.page.deleteLater();app.processEvents()
        print(json.dumps({'out':str(out),'success':report['success'],'checks':report['checks']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
