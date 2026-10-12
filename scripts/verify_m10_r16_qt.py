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
configure(ROOT)
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS','--mute-audio')

def main():
    if '--child' not in sys.argv:
        from ptb_desktop.vocal_tract.process_guard import OwnedJob
        with verification_run('m10-r16','qt','hidden Qt, synthetic s/l/t and tip sweep, actual controls/undo/redo, v3 export and legacy import; zero monitor volume') as (out,scratch):
            with (out/'qt.log').open('w',encoding='utf8') as log:
                p=subprocess.Popen([sys.executable,'-B','-X','utf8',__file__,'--child',str(out),str(scratch)],cwd=ROOT,stdout=log,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW)
                job=OwnedJob(p)
                try:
                    assert p.wait(timeout=240)==0,str(out/'qt.log')
                finally:
                    if p.poll() is None:p.terminate();p.wait(timeout=10)
                    job.close()
            print(out/'report.json')
        return
    from PyQt6.QtCore import QEventLoop,QTimer,Qt
    from PyQt6.QtWidgets import QApplication
    from ptb_desktop.host import Workbench,register_scheme
    out,scratch=map(Path,sys.argv[-2:]);report={'passed':False,'checks':[],'timing':[]}
    register_scheme();app=QApplication(['M10-R16-owned-QA'])
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
        pause();settled();report['timing'].append(round((time.perf_counter()-start)*1000,1))
    def capture(name):
        pause(250);w.view.grab().save(str(out/(name+'.png')))
    def section(index):
        local('const e=d.querySelector("#sliceRange");e.value='+str(index)+';e.dispatchEvent(new Event("input"));return true;');pause();settled()
    def check_pose(name):
        info=local('return {method:win.qaViewer.state.geometry_method,sections:win.qaViewer.state.centerline.length,triangles:win.qaViewer.airData.triangles.length/3,anchor:win.qaViewer.state.contours.lower_cover[win.qaViewer.state.tongue_floor_rib]};')
        assert info['method']=='m10/r16';report['checks'].append({'pose':name,**info})
    try:
        settled();w.vocal.invoke('audio/settings',{'volume':0})
        local("const script=d.createElement('script');script.type='module';script.textContent=\"import {viewer} from './app.js';window.qaViewer=viewer;\";d.body.append(script);return true;")
        until('!!win.qaViewer?.state')
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
        anchor=local('return win.qaViewer.state.contours.lower_cover[12]')
        for x,y in [(5,1),(2,2.5),(7,0)]:
            edit('TTX',x);edit('TTY',y)
            assert local('return win.qaViewer.state.contours.lower_cover[12]')==anchor
            assert local('return !!d.querySelector("[data-role=tongue-floor-anchor]")')
            if x==2:capture('curled-floor-anchor')
        report['checks'].append('jaw-fixed floor marker remains stable during tip extension and curl')
        click('[data-preset=t]');check_pose('t');click('#tab-motion');click('#captureFrame');click('#exportFrames')
        until('d.querySelector("#frameStatus").textContent.includes("已导出")')
        doc=json.loads(target.read_text('utf8'));assert doc['version']==3 and doc['geometry_method']=='m10/r16'
        click('#importFrames');until('d.querySelector("#frameStatus").textContent.includes("已导入")')
        doc.update(version=2,model='VTL-2.4-JD2-M10-3');doc.pop('geometry_method');target.write_text(json.dumps(doc),encoding='utf8')
        click('#importFrames');until('d.querySelector("#frameStatus").textContent.includes("旧文件参数已保留")')
        report['checks'].append('real host export/import uses v3; v2 retains params and shows recomputation notice')
        click('#playFrames');until('!d.body.classList.contains("motion-busy")')
        assert '失败' not in local('return d.querySelector("#frameStatus").textContent')
        report['checks'].append('keyframe playback completes on R16 at zero monitor volume')
        for mode,label in [('dark','深色'),('light','浅色')]:
            assert js('(()=>{const e=document.querySelector("button[title=设置]");if(!e)return false;e.click();return true;})()');pause()
            assert js('(()=>{const e=[...document.querySelectorAll(".mode-choices button")].find(e=>e.textContent.trim()=='+json.dumps(label)+');if(!e)return false;e.click();return true;})()');pause()
            js('document.querySelector("#tab-M10").click()');pause(250)
            assert local('return d.documentElement.dataset.theme')==mode
            capture(mode)
        report['passed']=True
    except Exception:
        capture('failure');raise
    finally:
        write_json(out/'report.json',report);w.vocal.close();w.close();w.deleteLater();pause(500)

if __name__=='__main__':main()
