"""Owned Qt dental raster sweep and preset save/load validation."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from workbench_source import configure
from verification_artifacts import verification_run, write_json

ROOT = Path(__file__).resolve().parents[1]
configure(ROOT)
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS', '--mute-audio')


def main():
    if '--child' not in sys.argv:
        from ptb_desktop.vocal_tract.process_guard import OwnedJob
        with verification_run('m10-r19', 'qt', 'actual Qt SVG raster comparison at rigid teeth during tongue sweeps; preset states and complete defaults') as (out, scratch):
            with (out / 'qt.log').open('w', encoding='utf-8') as log:
                p = subprocess.Popen([sys.executable, '-B', '-X', 'utf8', __file__, '--child', str(out), str(scratch)], cwd=ROOT, stdout=log, stderr=log, creationflags=subprocess.CREATE_NO_WINDOW)
                job = OwnedJob(p)
                try:
                    assert p.wait(timeout=240) == 0, str(out / 'qt.log')
                finally:
                    if p.poll() is None:
                        p.terminate()
                        p.wait(timeout=10)
                    job.close()
            print(out / 'report.json')
        return
    from PyQt6.QtCore import QEventLoop, QTimer, Qt
    from PyQt6.QtWidgets import QApplication
    from ptb_desktop.host import Workbench, register_scheme
    from phonetic_core.vocal_tract.engine import Engine
    out, scratch = map(Path, sys.argv[-2:])
    report = {'passed': False, 'raster': [], 'checks': []}
    register_scheme()
    app = QApplication(['M10-R19-owned-QA'])
    w = Workbench(ROOT / 'frontend/dist', test=True, vocal_profile=scratch / 'profile', vocal_resources=ROOT / 'resources/vocal_tract/native', start_module='M10')
    w.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
    w.resize(1650, 1050)
    w.show()
    engine = Engine(resource_dir=ROOT / 'resources/vocal_tract/native')

    def pause(ms=80):
        loop = QEventLoop()
        QTimer.singleShot(ms, loop.quit)
        loop.exec()

    def js(code):
        loop = QEventLoop()
        box = []
        w.page.runJavaScript(code, lambda v: (box.append(v), loop.quit()))
        QTimer.singleShot(10000, loop.quit)
        loop.exec()
        assert box, 'JS timeout'
        return box[0]

    def local(code):
        return js('(()=>{const f=document.querySelector("iframe"),d=f?.contentDocument,win=f?.contentWindow;if(!d)return null;' + code + '})()')

    def until(code, seconds=45):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if local('return ' + code):
                return
            pause()
        raise AssertionError(code)

    def settled():
        until('d.body.dataset.engineState==="ready"&&d.body.dataset.posePending==="false"')

    def click(selector):
        assert local('const e=d.querySelector(' + json.dumps(selector) + ');if(!e||e.disabled)return false;e.click();return true;'), selector
        pause()
        settled()

    def capture(name):
        pause(150)
        w.view.grab().save(str(out / (name + '.png')))

    try:
        settled()
        local('const e=d.createElement("script");e.type="module";e.textContent="import {viewer} from \'./app.js\';window.qaViewer=viewer;";d.body.append(e);return true;')
        until('!!win.qaViewer?.state')
        assert local('return +d.querySelector("#sectionCount").value===65&&win.qaViewer.rawState.section_count===65')
        for count in [129,65]:
            local('const e=d.querySelector("#sectionCount");e.value='+json.dumps(str(count))+';e.dispatchEvent(new win.Event("change"));return true;');settled()
            assert local('return win.qaViewer.rawState.section_count')==count
        report['checks'].append({'default_sections':65,'intervals':64,'sampling_switch_129_65':True})
        engine.set_section_count(65)
        # Rasterize the actual SVG, suppressing annotations and handles only.
        # The expected pixels come independently from the head outline over
        # the panel, before any airway or tongue is drawn.
        raster = r'''
          win.qaRaster=null;win.qaSamples=[];win.qaSpill=[];
          const v=win.qaViewer,svg=v.svg.cloneNode(true),box=v.svg.viewBox.baseVal;
          svg.setAttribute('width',1000);svg.setAttribute('height',700);
          svg.querySelectorAll('[data-handle],text,[data-role$="guide"],[data-role="tongue-floor-anchor"]').forEach(e=>e.remove());
          svg.querySelectorAll('path:not([data-role="head-outline"])').forEach(e=>e.setAttribute('stroke','none'));
          const reference=svg.cloneNode(true);
          reference.querySelector('g').replaceChildren(reference.querySelector('[data-role="head-outline"]').cloneNode(true));
          const panel=win.getComputedStyle(d.documentElement).getPropertyValue('--panel').trim();
          const render=source=>new Promise((resolve,reject)=>{const img=new win.Image();img.onload=()=>{const canvas=d.createElement('canvas');canvas.width=1000;canvas.height=700;const ctx=canvas.getContext('2d');ctx.fillStyle=panel;ctx.fillRect(0,0,1000,700);ctx.drawImage(img,0,0);resolve(ctx);};img.onerror=reject;img.src='data:image/svg+xml;charset=utf-8,'+encodeURIComponent(new win.XMLSerializer().serializeToString(source));});
          Promise.all([render(svg),render(reference)]).then(([actual,expected])=>{
            const inside=(p,pts)=>{let sign=0;for(let i=0;i<pts.length-1;i++){const a=pts[i],b=pts[i+1],cross=(b[0]-a[0])*(p[1]-a[1])-(b[1]-a[1])*(p[0]-a[0]);if(Math.abs(cross)<.04)return false;if(sign&&sign*Math.sign(cross)<0)return false;sign=Math.sign(cross);}return true;};
            const result=[],matrix=v.svg.getScreenCTM();
            for(const name of ['upper_teeth','lower_teeth']){const pts=v.state.contours[name],xs=pts.map(p=>p[0]),ys=pts.map(p=>p[1]);let count=0,bad=0,max=0;
              // preserveAspectRatio uses the smaller scale and centres the SVG.
              const scale=Math.min(1000/box.width,700/box.height),ox=(1000-box.width*scale)/2,oy=(700-box.height*scale)/2;
              for(let x=Math.min(...xs);x<Math.max(...xs);x+=.025)for(let y=Math.min(...ys);y<Math.max(...ys);y+=.025){if(!inside([x,y],pts))continue;const px=Math.floor(ox+(x-box.x)*scale),py=Math.floor(oy+(-y-box.y)*scale);const a=actual.getImageData(px,py,1,1).data,b=expected.getImageData(px,py,1,1).data,delta=Math.max(...[0,1,2].map(i=>Math.abs(a[i]-b[i])));count++;if(delta>3)bad++;max=Math.max(max,delta);const point=new win.DOMPoint(x,-y).matrixTransform(matrix);win.qaSamples.push({name,x:point.x,y:point.y,rgb:[...b].slice(0,3)});}
              result.push({name,count,bad,max});
            }
            const contains=([x,y],pts)=>{let result=false;for(let i=0,j=pts.length-1;i<pts.length;j=i++){const a=pts[i],b=pts[j];if((a[1]>y)!==(b[1]>y)&&x<(b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0])result=!result;}return result;};
            const tissue=[v.state.tongueOutline,...['upper_teeth','lower_teeth','upper_lip','lower_lip'].map(n=>v.state.contours[n])];
            const near=(p,pts)=>pts.some((a,i)=>{const b=pts[(i+1)%pts.length],dx=b[0]-a[0],dy=b[1]-a[1],t=Math.max(0,Math.min(1,((p[0]-a[0])*dx+(p[1]-a[1])*dy)/(dx*dx+dy*dy||1)));return Math.hypot(p[0]-a[0]-t*dx,p[1]-a[1]-t*dy)<.065;});
            for(let x=4.15;x<6.2;x+=.025)for(let y=-2.5;y<.4;y+=.025){const xy=[x,y],p=new win.DOMPoint(x,-y);if(!(v.airPaths.filter(pts=>contains(xy,pts)).length%2)||!contains(xy,v.headContour)||contains(xy,v.state.oralBoundary)||tissue.some(t=>contains(xy,t))||[v.headContour,v.state.oralBoundary,...tissue].some(t=>near(xy,t)))continue;win.qaSpill.push([x,y]);const scale=Math.min(1000/box.width,700/box.height),ox=(1000-box.width*scale)/2,oy=(700-box.height*scale)/2,px=Math.floor(ox+(x-box.x)*scale),py=Math.floor(oy+(-y-box.y)*scale),b=expected.getImageData(px,py,1,1).data,point=p.matrixTransform(matrix);win.qaSamples.push({name:'dental_surround',x:point.x,y:point.y,rgb:[...b].slice(0,3)});}
            win.qaRaster=result;
          }).catch(e=>{win.qaRaster={error:String(e)};});return true;
        '''
        presets=json.loads((ROOT / 'frontend/public/vocal-tract/default-presets.json').read_text('utf-8'))
        cases=[]
        for name in ['a','t','l']:
            pose=presets['presets'].get(name, {'params':engine.presets[name]})
            for dx,dy in [(0,0),(-.4,0),(.4,0),(0,-.3),(0,.3),(-2,0),(-1,-1),(1,1)]:
                p=list(pose['params']);p[10]+=dx;p[11]+=dy
                engine.manual_root=pose.get('manual_root',False)
                cases.append((f'{name}-{dx}-{dy}', engine.snapshot(p, tongue_shape=pose.get('tongue_shape'), lip_width=pose.get('lip_width',1))))
        p=list(engine.presets['a']);p[3]=-3.95;p[10]=7;p[11]=0
        protruded=engine.snapshot(p,tongue_shape=[-.5,-.3,-.5,-.4])
        cases.append(('protruded',protruded))
        for theme,label in [('dark','深色'),('light','浅色')]:
            js('document.querySelector("button[title=设置]").click()');pause()
            assert js('(()=>{const e=document.querySelector("[data-preview-mode='+theme+']");if(!e)return false;e.click();return true;})()')
            js('document.querySelector("#tab-M10").click()');pause(200)
            for name,state in cases:
                local('win.qaViewer.setOptions({head:true,teeth:false,labels:true,nose:true});win.qaViewer.update('+json.dumps(state)+');return true;')
                local(raster);until('win.qaRaster!==null')
                result=local('return win.qaRaster')
                assert isinstance(result,list),result
                local('win.qaViewer.svg.querySelectorAll(\'[data-handle],text,[data-role$="guide"],[data-role="tongue-floor-anchor"]\').forEach(e=>e.remove());return true;');pause(40)
                local('win.qaViewer.svg.querySelectorAll(\'path:not([data-role="head-outline"])\').forEach(e=>e.setAttribute("stroke","none"));return true;');pause(40)
                frame=js('(()=>{const r=document.querySelector("iframe").getBoundingClientRect();return {x:r.x,y:r.y};})()')
                picture=w.view.grab().toImage();ratio=picture.width()/w.view.width()
                counts={n:{'count':0,'bad':0,'max':0} for n in ['upper_teeth','lower_teeth','dental_surround']}
                for sample in local('return win.qaSamples'):
                    color=picture.pixelColor(round((frame['x']+sample['x'])*ratio),round((frame['y']+sample['y'])*ratio))
                    delta=max(abs(a-b) for a,b in zip([color.red(),color.green(),color.blue()],sample['rgb']))
                    count=counts[sample['name']];count['count']+=1;count['bad']+=int(delta>3);count['max']=max(count['max'],delta)
                report['raster'].append({'theme':theme,'case':name,'teeth':result,'live_pixels':counts,'raw_spill':local('return win.qaSpill.length')})
                if theme=='dark' and name=='a--0.4-0':
                    # Reproduce the old overpaint with the same geometry and
                    # sample set; the test must detect removal of the fix.
                    local('d.querySelector(\'[clip-path="url(#sampledAirwayClip)"]\').removeAttribute("clip-path");return true;');pause(40)
                    broken=w.view.grab().toImage();bad=0
                    for sample in local('return win.qaSamples.filter(p=>p.name==="dental_surround")'):
                        color=broken.pixelColor(round((frame['x']+sample['x'])*ratio),round((frame['y']+sample['y'])*ratio))
                        bad+=int(max(abs(a-b) for a,b in zip([color.red(),color.green(),color.blue()],sample['rgb']))>3)
                    assert bad>0 and counts['dental_surround']['bad']==0
                    report['checks'].append({'unclipped_air_bad_pixels':bad,'bounded_air_bad_pixels':0,'samples':counts['dental_surround']['count']})
                    local('win.qaViewer.draw2D();return true;')
            local('win.qaViewer.draw2D();return true;')
            assert local('return d.querySelector("#noseToggle").checked&&win.qaViewer.showNose&&!!d.querySelector("[data-role=nasal-reference]")')
            capture('teeth-'+theme)
            local('win.qaViewer.setOptions({teeth:true,mode:"overlay"});win.qaViewer.zoom=2;win.qaViewer.pan=[2.8,-.5];win.qaViewer.draw2D();return true;')
            capture('protruded-'+theme)
            local('win.qaViewer.setOptions({teeth:false,mode:"organs"});win.qaViewer.reset();return true;')
        assert all(tooth['count']>0 and tooth['bad']==0 for row in report['raster'] for tooth in row['teeth']), report['raster']
        assert all(tooth['bad']==0 for row in report['raster'] for tooth in row['live_pixels'].values()), report['raster']
        assert sum(row['raw_spill'] for row in report['raster'])>0
        click('[data-preset=a]')
        assert local('return d.querySelector("#customPreset").textContent')=='已存构型'
        assert not local('return !!d.querySelector("#presets [data-preset=s]")')
        states={}
        for mode in ['load','save']:
            click('#customPreset' if mode=='load' else '#savePreset')
            states[mode]=local('return ["t","l","s","p"].map(s=>{const e=d.querySelector("[data-ipa="+s+"]");return {symbol:s,stored:e.classList.contains("stored"),opacity:win.getComputedStyle(e).opacity,disabled:e.disabled};})')
            capture('presets-'+mode)
            click('#presetCancel')
        for loaded,saved in zip(states['load'],states['save']):
            assert loaded['stored']==saved['stored'] and loaded['opacity']==saved['opacity'],states
            assert not saved['disabled'],saved
        report['checks'].append({'matching_save_load_state':states})
        for name,key in [('t','t'),('n','t'),('l','l')]:
            click('[data-preset='+name+']')
            actual=local('return {params:win.qaViewer.rawState.params,tongue_shape:win.qaViewer.rawState.tongue_shape,lip_width:win.qaViewer.rawState.lip_width,manual_root:win.qaViewer.rawState.manual_root,f0:+d.querySelector("#f0Range").value,sourceMode:d.querySelector("#sourceMode").value}')
            expected=presets['presets'][key]
            for field in ['params','tongue_shape','lip_width','manual_root','f0']:
                assert actual[field]==expected[field],(name,field,actual[field],expected[field])
            assert actual['sourceMode']==expected['source']['mode']
            report['checks'].append({'complete_default':name,'matches':key})
        # Exercise the real checkbox and compare the nasal image region, so
        # a checked control with a hidden or accidentally overpainted nose fails.
        click('[data-preset=t]')
        assert local('return d.querySelector("#noseToggle").checked&&win.qaViewer.showNose')
        bounds=local('const v=win.qaViewer,r=f.getBoundingClientRect(),m=v.svg.getScreenCTM(),points=v.nosePaths.flat().map(p=>new win.DOMPoint(p[0],-p[1]).matrixTransform(m));return [Math.min(...points.map(p=>p.x))+r.x,Math.min(...points.map(p=>p.y))+r.y,Math.max(...points.map(p=>p.x))+r.x,Math.max(...points.map(p=>p.y))+r.y];')
        pause(100);nose_on=w.view.grab().toImage();ratio=nose_on.width()/w.view.width()
        click('#noseToggle')
        assert local('return !d.querySelector("#noseToggle").checked&&!win.qaViewer.showNose&&!d.querySelector("[data-role=nasal-reference]")')
        pause(100);nose_off=w.view.grab().toImage()
        x0,y0,x1,y1=[round(x*ratio) for x in bounds];changed=0;tested=0
        for x in range(max(0,x0),min(nose_on.width(),x1),3):
            for y in range(max(0,y0),min(nose_on.height(),y1),3):
                tested+=1;a=nose_on.pixelColor(x,y);b=nose_off.pixelColor(x,y)
                changed+=int(max(abs(a.red()-b.red()),abs(a.green()-b.green()),abs(a.blue()-b.blue()))>8)
        assert changed>100 and changed/tested>.05,(changed,tested)
        click('#noseToggle')
        assert local('return d.querySelector("#noseToggle").checked&&win.qaViewer.showNose&&!!d.querySelector("[data-role=nasal-reference]")')
        click('#threeButton')
        assert local('return win.qaViewer.showNose&&win.qaViewer.airData.nasalReference.triangles.length>0')
        for organ in ['tongue','lips','velum']:
            assert local('win.qaViewer.select('+json.dumps(organ)+');return [...win.qaViewer.controls.values()].every(c=>c.mesh.visible)')
        click('#sagittalButton')
        for organ in ['tongue','lips','velum']:
            assert local('win.qaViewer.select('+json.dumps(organ)+');return win.qaViewer.handles().every(h=>!!d.querySelector("[data-handle="+h.name+"]")===!h.side)')
        local('win.qaViewer.select("tongue");return true;')
        report['checks'].append({'handles_2d':11,'handles_3d':14,'side_handles_absent_in_sagittal':True,'independent_of_selected_organ':True})
        for handle in ['upper_lip','lower_lip','velum']:
            before=local('return win.qaViewer.rawState.params')
            key='ArrowRight' if handle=='velum' else 'ArrowUp'
            local('d.querySelector("[data-handle='+handle+']").dispatchEvent(new win.KeyboardEvent("keydown",{key:'+json.dumps(key)+',bubbles:true}));return true;');settled()
            assert local('return win.qaViewer.rawState.params')!=before,handle
            click('[data-preset=t]')
        report['checks'].append('lip and velum handles respond directly to keyboard edits')
        teeth=local('return [win.qaViewer.rawState.contours.upper_teeth,win.qaViewer.rawState.contours.lower_teeth]')
        for _ in range(4):
            local('d.querySelector("[data-handle=tip]").dispatchEvent(new win.KeyboardEvent("keydown",{key:"ArrowDown",bubbles:true}));return true;');settled()
            assert local('return [win.qaViewer.rawState.contours.upper_teeth,win.qaViewer.rawState.contours.lower_teeth]')==teeth
            assert local('return !!d.querySelector("[data-role=nasal-reference]")')
        report['checks'].append({'nasal_toggle_pixels_changed':changed,'nasal_pixels_tested':tested,'nasal_2d_3d':True,'tip_key_edits_keep_teeth_and_nose':True})
        click('[data-preset=l]')
        # Save a dim, enabled symbol into the isolated profile, then reopen.
        click('#savePreset');click('[data-ipa=p]')
        until('!d.querySelector("#presetDialog").open')
        click('#customPreset')
        assert local('return d.querySelector("[data-ipa=p]").classList.contains("stored")&&!d.querySelector("[data-ipa=p]").disabled')
        click('[data-ipa=p]')
        saved=w.vocal.invoke('presets/load',{})
        assert len(saved)==1 and saved[0]['name']=='p' and saved[0]['tongue_shape']==presets['presets']['l']['tongue_shape']
        report['checks'].append('dim unsaved symbol is clickable; newly saved p is lit and reloads the complete pose')
        click('[data-preset=t]')
        for theme in ['light','dark']:
            js('document.querySelector("button[title=设置]").click()');pause()
            assert js('(()=>{const e=document.querySelector("[data-preview-mode='+theme+']");if(!e)return false;e.click();return true;})()')
            js('document.querySelector("#tab-M10").click()');pause(200)
            capture('normal-'+theme)
            click('#savePreset');capture('save-'+theme);click('#presetCancel')
        report['passed']=True
    except Exception:
        capture('failure')
        raise
    finally:
        write_json(out/'report.json',report)
        engine.close();w.vocal.close();w.close();w.deleteLater();pause(500)


if __name__=='__main__':
    main()
