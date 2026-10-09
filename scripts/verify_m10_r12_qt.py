"""Owned hidden Qt and native M10 worker, real pointer/keyboard regression."""
import json
import os
import sys
import time
from pathlib import Path
from uuid import uuid4

ROOT=Path(__file__).resolve().parents[1]
from workbench_source import bind_sources
bind_sources(ROOT)
os.environ['QT_QPA_PLATFORM']='windows'
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS','--mute-audio')


def main():
    from PyQt6.QtCore import QEventLoop,QTimer,Qt,QPoint,QRect
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtTest import QTest
    from ptb_desktop.host import Workbench,register_scheme
    out=Path(os.environ.get('PTB_M10_QA_OUTPUT',str(ROOT/'output/validation/m10-r12')))/('qt-'+uuid4().hex[:8]);out.mkdir(parents=True)
    register_scheme();app=QApplication(['M10-R12-owned-QA'])
    w=Workbench(ROOT/'frontend/dist',test=True,vocal_profile=out/'profile',vocal_resources=ROOT/'resources/vocal_tract/native',start_module='M10')
    w.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen);w.resize(1680,1000);w.show()
    report={'success':False,'checks':[],'out':str(out)}
    def pause(ms=120):
        loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
    def js(code):
        loop=QEventLoop();box=[];w.page.runJavaScript(code,lambda v:(box.append(v),loop.quit()));QTimer.singleShot(8000,loop.quit);loop.exec()
        assert box,'JS timeout';return box[0]
    def local(code):return js('(()=>{const f=[...document.querySelectorAll("iframe")].find(e=>e.getAttribute("src")?.includes("vocal-tract/")),d=f?.contentDocument,win=f?.contentWindow;if(!d)return null;'+code+'})()')
    def until(code):
        end=time.monotonic()+45
        while time.monotonic()<end:
            if local('return '+code):return
            pause(80)
        raise AssertionError(code)
    def settled():
        until('d.body.dataset.engineState==="ready"&&d.body.dataset.posePending==="false"');pause(120)
        assert not local('return win.qaRenderErrors??[]'),local('return win.qaRenderErrors')
    def click(selector):
        assert local('const e=d.querySelector('+json.dumps(selector)+');if(!e||e.disabled)return false;e.click();return true;'),selector
        pause();settled()
    def value(selector,v):
        assert local('const e=d.querySelector('+json.dumps(selector)+');if(!e)return false;e.value='+json.dumps(v)+';e.dispatchEvent(new Event("input",{bubbles:true}));e.dispatchEvent(new Event("change",{bubbles:true}));return true;'),selector
        settled()
    def state(code):return local('const v=win.qaViewer,s=v.state;'+code)
    def shot(name):pause(200);w.view.grab().save(str(out/(name+'.png')))
    def detail(name,handle,offset,size):
        xy=state('const h=d.querySelector(\'[data-handle="'+handle+'"]\'),p=v.svg.createSVGPoint(),q=p.matrixTransform(h.getScreenCTM()),r=f.getBoundingClientRect();return [q.x+r.x,q.y+r.y];')
        w.view.grab(QRect(round(xy[0]+offset[0]),round(xy[1]+offset[1]),*size)).save(str(out/(name+'.png')))
    def drag(handle,dx,dy):
        point=state('const h=d.querySelector(\'[data-handle="'+handle+'"]\'),p=v.svg.createSVGPoint();p.x=0;p.y=0;const q=p.matrixTransform(h.getScreenCTM()),r=f.getBoundingClientRect();return [q.x+r.x,q.y+r.y];')
        target=w.view.focusProxy() or w.view
        start=target.mapFromGlobal(w.view.mapToGlobal(QPoint(round(point[0]),round(point[1]))))
        QTest.mouseMove(target,start);QTest.mousePress(target,Qt.MouseButton.LeftButton,pos=start)
        for i in range(1,13):
            end=start+QPoint(round(dx*i/12),round(dy*i/12));QTest.mouseMove(target,end,25);pause(30)
        QTest.mouseRelease(target,Qt.MouseButton.LeftButton,pos=end);settled()
    try:
        settled()
        local("const script=d.createElement('script');script.type='module';script.textContent=\"import {viewer} from './app.js';window.qaViewer=viewer;window.qaRenderErrors=[];const update=viewer.update.bind(viewer);viewer.update=s=>{try{return update(s);}catch(e){window.qaRenderErrors.push({error:String(e),stack:e.stack});throw e;}};\";d.body.append(script);return true;")
        until('!!win.qaViewer?.state')
        assert state('return v.selected')=='tongue'
        assert local('return !!d.querySelector("[data-handle=larynx]")')
        shot('initial-larynx')
        drag('larynx',0,-14);assert state('return s.larynx_height')>0
        click('#undoButton');assert abs(state('return s.larynx_height'))<1e-8
        for organ in ['lips','velum','tongue']:
            click('[data-select-organ='+organ+']');assert local('return d.querySelectorAll("[data-handle=larynx]").length')==1
        report['checks'].append('initial larynx visible; native drag/undo; remains visible across organ selections')
        click('#teethToggle');click('[data-mode=airway]')
        before=state('return s.meshes.filter(m=>m.name.endsWith("_teeth"))')
        old=state('return s.limited[13]');drag('blade',0,35);new=state('return s.limited[13]')
        assert new<old-.15,(old,new)
        assert state('return s.meshes.filter(m=>m.name.endsWith("_teeth"))')==before
        report['checks'].append({'blade_native_drag':[old,new]});shot('blade-down')
        click('[data-preset=a]')
        for selector,v in [('#p-TCY',-2.1),('#p-TTY',.25),('#p-TBX',2.6),('#p-TBY',-1.3),('#p-TTX',2.1)]:value(selector,v)
        if os.environ.get('PTB_M10_QA_TISSUE'):
            assert state('return s.centerline.length===257&&s.airway_sections.length===257')
            value('#p-TBX',2.3)
            state('v.zoom=2;v.draw2D();return true;')
            movements=[]
            for dx,dy in [(-10,0),(10,0),(0,-10),(0,10)]:
                before_point=state('return v.handles().find(h=>h.name==="blade").p')
                before_pose=state('return s.params')
                drag('blade',dx,dy)
                after_point=state('return v.handles().find(h=>h.name==="blade").p')
                movement=[after_point[k]-before_point[k] for k in range(2)]
                component=movement[0] if dx else -movement[1]
                assert component*(dx or dy)>0,(dx,dy,movement)
                # Real surface attachment, including the folded/depressed pose.
                assert state('const r=s.tongue_blade_rib,a=Math.floor(r),t=r-a,h=v.handles().find(h=>h.name==="blade").p;return [0,1].every(k=>Math.abs(h[k]-(s.contours.tongue[a][k]*(1-t)+s.contours.tongue[a+1][k]*t))<1e-8)')
                movements.append({'pointer_px':[dx,dy],'surface_cm':movement,'compute_ms':state('return s.compute_ms')})
                click('#undoButton');assert state('return s.params')==before_pose
            report['checks'].append({'attached_blade_four_directions':movements})
            value('#p-TBX',2.6)
            state('v.zoom=1;v.draw2D();return true;')
        old=state('return s.limited[10]');drag('tip',-18,0);new=state('return s.limited[10]')
        assert new<old-.1,(old,new)
        report['checks'].append({'tip_native_retract':[old,new]});shot('retroflex')
        if os.environ.get('PTB_M10_QA_SMOOTH'):
            state('v.zoom=2.8;v.pan=[.2,0];v.draw2D();return true;');shot('retroflex-closeup')
            detail('tip-detail','tip',[-160,-50],[300,230])
            state('v.zoom=1;v.pan=[0,0];v.draw2D();return true;')
        pose=state('return {params:s.params,larynx_height:s.larynx_height}')
        click('#savePreset');click('[data-ipa="a"]');until('!d.querySelector("#presetDialog").open')
        saved=json.loads((out/'profile/presets.json').read_text('utf-8'))['presets'][0]
        assert saved['params']==pose['params']
        click('[data-preset=u]');click('#customPreset');click('[data-ipa="a"]')
        assert state('return {params:s.params,larynx_height:s.larynx_height}')==pose
        click('#tab-motion');click('#captureFrame');until('d.querySelectorAll(".pose-card").length===1')
        until('d.querySelectorAll(".pose-card").length===1');pause(500)
        saved_frame=json.loads((out/'profile/keyframes.json').read_text('utf-8'))['frames'][0]
        assert saved_frame['params']==pose['params'];click('#tab-organs')
        report['checks'].append('depressed/retracted pose saves, loads and persists as a keyframe with unchanged requested parameters')
        click('#threeButton')
        if os.environ.get('PTB_M10_QA_TISSUE'):
            assert state('const p=v.handles().find(h=>h.name==="blade").p,q=v.controls.get("blade").mesh.position;return Math.hypot(p[0]-q.x,p[1]-q.y,p[2]-q.z)<1e-8')
        for organ in ['tongue','lips','velum']:
            click('[data-select-organ='+organ+']');assert state('return v.controls.get("larynx").mesh.visible')
        click('[data-mode=organs]');shot('retroflex-3d')
        if os.environ.get('PTB_M10_QA_SMOOTH'):
            state('v.camera.position.set(6,2,14);v.orbit.target.set(1,-.3,0);v.orbit.update();v.needsRender=true;return true;');shot('retroflex-3d-closeup')
        click('[data-preset=a]');value('#p-TBY',-1.5);shot('teeth-3d')
        if os.environ.get('PTB_M10_QA_TISSUE'):
            click('#sagittalButton');click('[data-mode=airway]');click('[data-select-organ=tongue]')
            for selector,v in [('#p-TCY',-2.1),('#p-TTY',.6),('#p-TTX',3.2),('#p-TBX',2.6),('#p-TBY',-.8)]:value(selector,v)
            state('v.zoom=2.8;v.pan=[1,0];v.draw2D();return true;')
            for theme in ['light','dark']:
                js('document.documentElement.dataset.theme='+json.dumps(theme));pause(200)
                shot('raised-blade-'+theme);detail('raised-blade-detail-'+theme,'tip',[-230,-90],[360,340])
            assert local('return d.querySelector("[data-role=midsagittal-airway]").getAttribute("mask")==="url(#sectionOutsideTissue)"')
            state('v.zoom=1;v.pan=[0,0];v.draw2D();return true;')
        click('#sagittalButton');click('[data-preset=a]');click('[data-select-organ=velum]');value('#portRange',.4)
        click('[data-mode=airway]')
        assert local('return d.querySelector("[data-organ=velum]").getAttribute("stroke")==="none"')
        assert local('return !d.querySelector("[data-role=velum-exposed-boundary]").getAttribute("d").endsWith("Z")')
        state('v.zoom=2.8;v.pan=[-2.9,.4];v.draw2D();return true;');shot('velum-closeup')
        if os.environ.get('PTB_M10_QA_SMOOTH'):detail('velum-detail','velum',[-100,-270],[340,315])
        state('v.zoom=1;v.pan=[0,0];v.draw2D();return true;')
        for theme in ['light','dark']:
            js('document.documentElement.dataset.theme='+json.dumps(theme));pause(200);shot('all-'+theme)
        for width,height,size in [(1280,800,24),(1920,1080,14)]:
            w.resize(width,height);local('d.documentElement.style.fontSize='+json.dumps(str(size)+'px')+';return true;');pause(250)
            assert local('return d.documentElement.scrollWidth<=d.documentElement.clientWidth+1')
            assert local('return !!d.querySelector("[data-handle=larynx]")')
            shot(f'layout-{width}-{size}')
        report['checks'].append('three-dimensional constant larynx and dynamic dental clipping; exposed velum border; light/dark rendering')
        report['success']=True
    except Exception as exc:
        report['failure']=repr(exc);shot('failure');raise
    finally:
        processes=[w.service.process,w.vocal.process]
        w.closing=True;w.close();w.page.deleteLater();app.processEvents()
        report['owned_exit_codes']=[p.poll() for p in processes if p is not None]
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf-8')
        assert all(code==0 for code in report['owned_exit_codes']),report['owned_exit_codes']
        print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
