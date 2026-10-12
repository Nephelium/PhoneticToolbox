"""Owned hidden Qt exercise of tongue edits, air display and persistence."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from workbench_source import configure
from verification_artifacts import verification_run,write_json

ROOT=Path(__file__).resolve().parents[1]
R18='--r18' in sys.argv
METHOD='m10/r18' if R18 else 'm10/r17'
configure(ROOT)
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS','--mute-audio')

def main():
    if '--child' not in sys.argv:
        from ptb_desktop.vocal_tract.process_guard import OwnedJob
        with verification_run('m10-r18' if R18 else 'm10-r17','qt','hidden Qt, synthetic s/l/t and tip sweep, actual controls/undo/redo, v4 export and legacy import; zero monitor volume') as (out,scratch):
            with (out/'qt.log').open('w',encoding='utf8') as log:
                p=subprocess.Popen([sys.executable,'-B','-X','utf8',__file__,'--child',*(['--r18'] if R18 else []),*(['--r18-only'] if '--r18-only' in sys.argv else []),str(out),str(scratch)],cwd=ROOT,stdout=log,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW)
                job=OwnedJob(p)
                try:
                    assert p.wait(timeout=240)==0,str(out/'qt.log')
                finally:
                    if p.poll() is None:p.terminate();p.wait(timeout=10)
                    job.close()
            print(out/'report.json')
        return
    from PyQt6.QtCore import QEventLoop,QTimer,Qt,QPoint,QPointF
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtTest import QTest
    from PyQt6.QtGui import QWheelEvent
    from ptb_desktop.host import Workbench,register_scheme
    out,scratch=map(Path,sys.argv[-2:]);report={'passed':False,'checks':[],'timing':[]}
    register_scheme();app=QApplication(['M10-R17-owned-QA'])
    w=Workbench(ROOT/'frontend/dist',test=True,vocal_profile=scratch/'profile',vocal_resources=ROOT/'resources/vocal_tract/native',start_module='M10')
    w.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen);w.resize(1650,1050);w.show()
    target=scratch/'frames.ptb-vocal.json'
    w.bridge.test_vocal_picker=lambda op:str(target)
    def pause(ms=100):
        loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
    def js(code):
        loop=QEventLoop();box=[];w.page.runJavaScript(code,lambda v:(box.append(v),loop.quit()));QTimer.singleShot(10000,loop.quit);loop.exec();assert box,'JS timeout';return box[0]
    def local(code):return js('(()=>{const f=document.querySelector("iframe"),d=f?.contentDocument,win=f?.contentWindow;if(!d)return null;'+code+'})()')
    def until(code,seconds=45):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            if local('return '+code):return
            pause(80)
        raise AssertionError(code)
    def settled():until('d.body.dataset.engineState==="ready"&&d.body.dataset.posePending==="false"')
    def click(selector):
        assert local('const e=d.querySelector('+json.dumps(selector)+');if(!e||e.disabled)return false;e.click();return true;'),selector
        pause();settled()
    def edit(name,value):
        start=time.perf_counter()
        assert local('const e=d.querySelector('+json.dumps('#p-'+name)+');if(!e)return false;e.value='+json.dumps(value)+';e.dispatchEvent(new Event("input"));e.dispatchEvent(new Event("change"));return true;')
        pause(20);settled();report['timing'].append(round((time.perf_counter()-start)*1000,1))
    def capture(name):
        pause(250);w.view.grab().save(str(out/(name+'.png')))
    def section(index):
        local('const e=d.querySelector("#sliceRange");e.value='+str(index)+';e.dispatchEvent(new Event("input"));return true;');pause();settled()
    def check_pose(name):
        info=local('return {method:win.qaViewer.state.geometry_method,sections:win.qaViewer.state.centerline.length,triangles:win.qaViewer.airData.triangles.length/3,anchor:win.qaViewer.state.contours.lower_cover[win.qaViewer.state.tongue_floor_rib]};')
        assert info['method']==METHOD;report['checks'].append({'pose':name,**info})
    try:
        settled();w.vocal.invoke('audio/settings',{'volume':0})
        local("const script=d.createElement('script');script.type='module';script.textContent=\"import {viewer} from './app.js';window.qaViewer=viewer;\";d.body.append(script);return true;")
        until('!!win.qaViewer?.state')
        if '--r18-only' in sys.argv:
            from verify_m10_r18_ui import exercise
            def drag_handle(handle,dx,dy):
                pt=js('(()=>{const f=document.querySelector("iframe"),e=f.contentDocument.querySelector("[data-handle='+handle+'] circle:last-of-type"),r=e.getBoundingClientRect(),fr=f.getBoundingClientRect();return [fr.x+r.x+r.width/2,fr.y+r.y+r.height/2];})()')
                point=QPoint(round(pt[0]),round(pt[1]));receiver=w.view.focusProxy() or w.view
                QTest.mousePress(receiver,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,point)
                for fraction in [.25,.5,.75,1.]:
                    QTest.mouseMove(receiver,point+QPoint(round(dx*fraction),round(dy*fraction)));pause(100)
                QTest.mouseRelease(receiver,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,point+QPoint(dx,dy));pause(100);settled()
            exercise(local,click,until,settled,capture,w,target,report,drag_handle)
            report['passed']=True
            return

        click('[data-preset=l]');check_pose('l');capture('l-sagittal')
        click('#threeButton');click('#teethToggle');click('#fullModelToggle');click('[data-mode=airway]')
        local('win.qaViewer.camera.position.set(14,8,19);win.qaViewer.orbit.target.set(1,-1,0);win.qaViewer.orbit.update();win.qaViewer.needsRender=true;return true;')
        section(109);capture('l-airway-teeth')
        indices=local('return JSON.stringify(win.qaViewer.airData.triangles)');click('#teethToggle')
        assert local('return JSON.stringify(win.qaViewer.airData.triangles)')==indices
        click('#teethToggle');report['checks'].append('hiding teeth preserves exact air cavity clipping')
        # The s preset remains a test input; no removed UI preset is restored.
        meta=w.vocal.invoke('meta',{})
        doc=w.vocal.invoke('document/create',{'frames':[{'params':meta['presets']['s'],'f0':150,'lip_width':1,'duration':.2}],'pitch_curve':[]})
        target.write_text(json.dumps(doc),encoding='utf8');click('#tab-motion');click('#importFrames')
        until('d.querySelector("#frameStatus").textContent.includes("已导入")')
        click('#keyframeList .pose-card');click('#tab-organs')
        check_pose('s');section(108);capture('s-airway-teeth')
        edit('TS3',.8);after=local('return win.qaViewer.state.params[18]');click('#undoButton');assert local('return win.qaViewer.state.params[18]')!=after
        click('#redoButton');assert local('return win.qaViewer.state.params[18]')==after
        report['checks'].append('actual anterior edge control plus undo/redo')
        click('[data-preset=a]');click('#sagittalButton');click('[data-mode=organs]')
        anchor=local('return win.qaViewer.state.contours.lower_cover[10]')
        for x,y in [(5,1),(2,2.5),(7,0)]:
            edit('TTX',x);edit('TTY',y)
            assert local('return win.qaViewer.state.contours.lower_cover[10]')==anchor
            assert local('return !!d.querySelector("[data-role=tongue-floor-anchor]")')
            if x==2:capture('curled-floor-anchor')
        report['checks'].append('jaw-fixed floor marker remains stable during tip extension and curl')
        click('[data-preset=t]');check_pose('t');click('#tab-motion');click('#captureFrame');click('#exportFrames')
        until('d.querySelector("#frameStatus").textContent.includes("已导出")')
        doc=json.loads(target.read_text('utf8'));assert doc['version']==(5 if R18 else 4) and doc['geometry_method']==METHOD
        click('#importFrames');until('d.querySelector("#frameStatus").textContent.includes("已导入")')
        doc.update(version=2,model='VTL-2.4-JD2-M10-3');doc.pop('geometry_method');target.write_text(json.dumps(doc),encoding='utf8')
        click('#importFrames');until('d.querySelector("#frameStatus").textContent.includes("旧文件参数已保留")')
        report['checks'].append('real host export/import uses current exchange version; v2 retains params and shows recomputation notice')
        click('#playFrames');until('!d.body.classList.contains("motion-busy")')
        assert '失败' not in local('return d.querySelector("#frameStatus").textContent')
        report['checks'].append('keyframe playback completes at zero monitor volume')
        for mode,label in [('dark','深色'),('light','浅色')]:
            assert js('(()=>{const e=document.querySelector("button[title=设置]");if(!e)return false;e.click();return true;})()');pause()
            assert js('(()=>{const e=[...document.querySelectorAll(".mode-choices button")].find(e=>e.textContent.trim()=='+json.dumps(label)+');if(!e)return false;e.click();return true;})()');pause()
            js('document.querySelector("#tab-M10").click()');pause(250)
            assert local('return d.documentElement.dataset.theme')==mode
            capture(mode)
        click('[data-preset=a]');click('#sagittalButton');click('#tab-organs')
        edit('TTY',1.5);edit('TBY',-3)
        assert local('return win.qaViewer.state.limited[13]')>-2
        capture('bounded-blade-posterior-floor')
        click('[data-preset=a]');click('[data-side-region="2"]')
        def position(selector):
            return js('(()=>{const f=document.querySelector("iframe"),e=f.contentDocument.querySelector('+json.dumps(selector)+'),r=e.getBoundingClientRect(),fr=f.getBoundingClientRect();return [fr.x+r.x+r.width/2,fr.y+r.y+r.height/2];})()')
        pt=position('#sectionHandles [data-side] circle:last-child');point=QPoint(round(pt[0]),round(pt[1]))
        receiver=w.view.focusProxy() or w.view
        before=local('return win.qaViewer.state.params[17]')
        QTest.mousePress(receiver,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,point)
        for dy in range(2,25,2):
            QTest.mouseMove(receiver,point+QPoint(0,-dy));pause(45)
        QTest.mouseRelease(receiver,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,point+QPoint(0,-24));settled()
        after=local('return win.qaViewer.state.params[17]')
        assert after>before,(pt,before,after)
        report['checks'].append({'actual_side_pointer':True,'before':before,'after':after,'render_ms':local('return +win.qaViewer.el.dataset.renderMs')})
        # Keep synthesized output arriving at zero monitor volume during zoom.
        click('#tab-sound');click('#liveButton')
        click('[data-analysis="monitor"]');pause(400)
        def wheel(selector,delta,ctrl=True):
            pt=position(selector);p=QPointF(*pt)
            event=QWheelEvent(p,QPointF(receiver.mapToGlobal(QPoint(round(pt[0]),round(pt[1])))),QPoint(0,0),QPoint(0,delta),Qt.MouseButton.NoButton,Qt.KeyboardModifier.ControlModifier if ctrl else Qt.KeyboardModifier.NoModifier,Qt.ScrollPhase.NoScrollPhase,False)
            QApplication.sendEvent(receiver,event);pause(120)
        def span():return local('return +d.querySelector("#waveCanvas").dataset.viewSeconds')
        initial=span();wheel('#waveCanvas',120);small=span();assert small<initial,(initial,small)
        assert local('return d.querySelector("#waveCanvas").dataset.viewStart===d.querySelector("#spectrogramCanvas").dataset.viewStart')
        wheel('#spectrogramCanvas',-120);assert span()>small
        before=span();wheel('#waveCanvas',-120,False);assert span()==before
        click('#monitorExpand');wheel('#spectrogramCanvas',120);assert span()<before
        capture('monitor-ctrl-wheel-expanded')
        # QTest mouseDClick does not deliver a DOM double click in this hidden
        # WebEngine instance. Exercise the registered DOM handler explicitly;
        # the wheel and side-pointer checks above remain actual Qt events.
        local('d.querySelector("#waveCanvas").dispatchEvent(new win.MouseEvent("dblclick",{bubbles:true}));return true;');pause(150)
        assert abs(span()-3)<1e-8,span()
        click('#monitorClose')
        report['checks'].append({'actual_ctrl_wheel':True,'before':initial,'zoomed':small,'reset':span(),'shared_axes':True,'expanded':True,'ordinary_wheel_unchanged':True,'reset_event':'DOM dblclick'})
        capture('monitor-two-columns')
        w.resize(2050,1150);pause(600)
        layout=local('const r=s=>{const a=d.querySelector(s).getBoundingClientRect();return {x:a.x,y:a.y,width:a.width,height:a.height};};return {width:win.innerWidth,studio:r(".studio"),analysis:r(".analysis"),inspector:r(".inspector"),wave:r("#waveCanvas")};')
        assert layout['width']>=1600,layout
        assert layout['studio']['x']<layout['analysis']['x']<layout['inspector']['x'],layout
        assert abs(layout['studio']['y']-layout['analysis']['y'])<1,layout
        assert layout['wave']['height']>40,layout
        wheel('#waveCanvas',120);assert span()<3
        capture('monitor-three-columns')
        click('#liveButton');click('#tab-organs');click('[data-analysis="acoustics"]')
        click('[data-side-region="3"]')
        pt=position('#sectionHandles [data-side] circle:last-child');point=QPoint(round(pt[0]),round(pt[1]))
        before=local('return win.qaViewer.state.params[18]')
        QTest.mousePress(receiver,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,point)
        for dy in range(2,19,2):
            QTest.mouseMove(receiver,point+QPoint(0,-dy));pause(45)
        QTest.mouseRelease(receiver,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,point+QPoint(0,-18));settled()
        after=local('return win.qaViewer.state.params[18]')
        assert after>before,(before,after)
        report['checks'].append({'three_column_layout':layout,'actual_ctrl_wheel':True,'anterior_pointer_before':before,'anterior_pointer_after':after})
        capture('organs-three-columns')
        # Regress the reported anterior underside crossing the jaw/incisors.
        click('[data-preset=a]');edit('JA',-3.95);edit('TTX',7);edit('TTY',-2)
        capture('jaw-lower-incisor-contact')
        outline=local('return JSON.stringify(win.qaViewer.state.tongueOutline)')
        click('#teethToggle')
        assert local('return JSON.stringify(win.qaViewer.state.tongueOutline)')==outline
        capture('jaw-contact-hidden-teeth');click('#teethToggle')
        for jaw in [-7,0]:
            edit('JA',jaw);capture('jaw-contact-'+str(jaw))
        outline=local('return JSON.stringify(win.qaViewer.state.tongueOutline)')
        section_faces=local('return win.qaViewer.state.meshes.find(m=>m.name==="tongue").triangles.length')
        click('#threeButton')
        assert local('return JSON.stringify(win.qaViewer.state.tongueOutline)')==outline
        assert local('return win.qaViewer.state.meshes.find(m=>m.name==="tongue").triangles.length')>section_faces
        local('win.qaViewer.camera.position.set(14,8,19);win.qaViewer.orbit.target.set(1,-1,0);win.qaViewer.orbit.update();win.qaViewer.needsRender=true;return true;')
        capture('jaw-lower-teeth-three-dimensions')
        report['checks'].append('jaw and lower-incisor contact in three jaw positions, 2-D/3-D and hidden teeth')
        click('[data-preset=a]');click('#sagittalButton')
        report['render_timing']=local('const a=[];for(let i=0;i<6;i++){win.qaViewer.update(win.qaViewer.rawState);a.push(win.qaViewer.renderTiming);}return a;')
        # Analytical weak PCM fixture through the real monitor analyzer and
        # parent/iframe port. It never reaches an audio device. The preceding
        # tests separately exercised live native output at zero volume.
        import numpy as np
        from phonetic_core.vocal_tract.monitor import analyze_output
        t=np.arange(3*48000)/48000
        signal=(.004*np.sin(2*np.pi*150*t)+.001*np.sin(2*np.pi*300*t)).astype(np.float32)
        invoke=w.vocal.invoke
        def monitor_fixture(op,body=None,**kw):
            if op!='audio/monitor':return invoke(op,body,**kw)
            body=body or {}
            return analyze_output(signal,48000,end=3,generation=999,**body)
        w.vocal.invoke=monitor_fixture
        click('[data-analysis="monitor"]')
        local('d.querySelector("#waveCanvas").dispatchEvent(new win.MouseEvent("dblclick",{bubbles:true}));return true;');pause(250)
        for _ in range(60):
            if span()<.1:break
            wheel('#waveCanvas',120)
        until('d.querySelector("#waveCanvas").dataset.waveMode==="samples"&&+d.querySelector("#waveCanvas").dataset.sampleCount>1000')
        assert local('return +d.querySelector("#waveCanvas").dataset.amplitudeLimit')<.02
        capture('monitor-real-samples-100ms')
        for _ in range(60):
            if span()<.012:break
            wheel('#waveCanvas',120)
        pause(300)
        assert .001<span()<.02
        assert local('return +d.querySelector("#waveCanvas").dataset.sampleCount')>100
        capture('monitor-real-samples-cycles')
        local('const e=d.querySelector("#monitorAmplitude");e.value="fixed";e.dispatchEvent(new Event("change"));return true;')
        assert local('return +d.querySelector("#waveCanvas").dataset.amplitudeLimit')==1
        local('const e=d.querySelector("#monitorAmplitude");e.value="auto";e.dispatchEvent(new Event("change"));return true;')
        assert js('(()=>{const e=document.querySelector("button[title=设置]");e.click();return true;})()');pause()
        assert js('(()=>{const e=[...document.querySelectorAll(".mode-choices button")].find(e=>e.textContent.trim()==="深色");e.click();return true;})()');pause()
        js('document.querySelector("#tab-M10").click()');pause(250)
        capture('monitor-real-samples-dark')
        click('#monitorExpand');capture('monitor-real-samples-expanded');click('#monitorClose')
        report['checks'].append({'weak_pcm_fixture':True,'frequency_hz':[150,300],'amplitudes':[.004,.001],'span':span(),'wave':local('return {...d.querySelector("#waveCanvas").dataset}'),'fixed_axis':True,'expanded':True})
        for _ in range(60):
            if span()<=.001:break
            wheel('#waveCanvas',120)
        pause(300)
        assert abs(span()-.001)<1e-9 and local('return +d.querySelector("#waveCanvas").dataset.sampleCount')==48
        report['checks'].append('1 ms waveform contains exactly 48 original samples')
        local('d.querySelector("#waveCanvas").dispatchEvent(new win.MouseEvent("dblclick",{bubbles:true}));return true;');pause(250)
        assert span()==3
        w.vocal.invoke=invoke
        if R18:
            from verify_m10_r18_ui import exercise
            def drag_handle(handle,dx,dy):
                pt=position('[data-handle='+handle+'] circle:last-of-type');point=QPoint(round(pt[0]),round(pt[1]))
                QTest.mousePress(receiver,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,point)
                for fraction in [.25,.5,.75,1.]:
                    QTest.mouseMove(receiver,point+QPoint(round(dx*fraction),round(dy*fraction)));pause(100)
                QTest.mouseRelease(receiver,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,point+QPoint(dx,dy));pause(100);settled()
            exercise(local,click,until,settled,capture,w,target,report,drag_handle)
        report['passed']=True
    except Exception:
        capture('failure');raise
    finally:
        write_json(out/'report.json',report);w.vocal.close();w.close();w.deleteLater();pause(500)

if __name__=='__main__':main()
