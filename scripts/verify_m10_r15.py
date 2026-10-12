"""Owned Qt/native M10-R15 and independent MP4 decoder verification.

Synthetic poses only. Playback uses zero monitor volume; no audible test tone.
All profiles and video/audio exports live in lifecycle-managed scratch.
"""
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from contextlib import nullcontext
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from workbench_source import configure
from verification_artifacts import verification_run, write_json

ROOT=Path(__file__).resolve().parents[1]
configure(ROOT)
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS','--mute-audio')


def native(report):
    from phonetic_core.vocal_tract.engine import Engine
    baseline=json.loads((ROOT/'tests/fixtures/m10-r14-257-digests.json').read_text('utf-8'))['fields']
    e=Engine(resource_dir=ROOT/'resources/vocal_tract/native')
    try:
        assert e.section_count==129
        e.set_section_count(257)
        for name, fields in baseline.items():
            s=e.snapshot(e.presets[name])
            for key, expected in fields.items():
                actual=hashlib.sha256(json.dumps(s[key],sort_keys=True,separators=(',',':')).encode()).hexdigest()
                assert actual==expected,(name,key)
        report['r14_257_exact_digest_regression']=True
        rows=[]
        for count in [65,129,257]:
            e.set_section_count(count)
            poses=[e.presets[name] for name in ['a','i','u','n','l','s']]
            for p in poses:e.snapshot(p)
            times=[]
            for _ in range(5):
                for p in poses:
                    start=time.perf_counter();s=e.snapshot(p);times.append((time.perf_counter()-start)*1000)
            assert len(s['centerline'])==count
            rows.append(dict(sections=count,median_ms=float(np.median(times)),p95_ms=float(np.percentile(times,95)),samples=len(times)))
        report['native_timing']=rows
    finally:e.close()


def main():
    recipe='verify_m10_r15.py; six native presets x 5, 65/129/257; single 50 ms voiced/silent and 0.6 s two-pose MP4; hidden Qt, monitor volume zero'
    if '--qt-child' not in sys.argv:
        from ptb_desktop.vocal_tract.process_guard import OwnedJob
        with verification_run('m10-r15','qt',recipe) as (out,scratch):
            process=None;job=None;started=time.monotonic()
            try:
                with (out/'qt.log').open('w',encoding='utf-8') as log:
                    process=subprocess.Popen([sys.executable,'-B','-X','utf8',__file__,'--qt-child',str(out),str(scratch)],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW)
                    job=OwnedJob(process)
                    while process.poll() is None:
                        if time.monotonic()-started>300:raise RuntimeError('Owned Qt verification timeout')
                        time.sleep(.1)
                    if process.poll()!=0:raise RuntimeError(f'Qt verification failed ({process.returncode}); see qt.log and failure.png')
            finally:
                if process and process.poll() is None:process.terminate();process.wait(timeout=10)
                if job:job.close()
            print(json.dumps({'out':str(out),**json.loads((out/'report.json').read_text('utf-8'))},ensure_ascii=False))
        return
    from PyQt6.QtCore import QEventLoop,QTimer,Qt
    from PyQt6.QtWidgets import QApplication,QFileDialog
    from ptb_desktop.host import Workbench,register_scheme
    report={'passed':False,'checks':[]}
    # The parent finalizer runs after this Qt process and all Chromium children
    # have exited, so no live browser process can block scratch cleanup.
    with nullcontext((Path(sys.argv[2]),Path(sys.argv[3]))) as (out,scratch):
        native(report)
        register_scheme();app=QApplication(['M10-R15-owned-QA'])
        w=Workbench(ROOT/'frontend/dist',test=True,vocal_profile=scratch/'profile',vocal_resources=ROOT/'resources/vocal_tract/native',start_module='M10')
        w.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen);w.resize(1650,1000);w.show()
        target=scratch/'single-current.mp4'
        w.bridge.test_vocal_picker=lambda op:str(target) if op=='video/begin' else ''
        def pause(ms=100):
            loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
        def js(code):
            loop=QEventLoop();box=[];w.page.runJavaScript(code,lambda v:(box.append(v),loop.quit()));QTimer.singleShot(8000,loop.quit);loop.exec()
            assert box,'JS timeout';return box[0]
        def local(code):
            return js('(()=>{const f=document.querySelector("iframe"),d=f?.contentDocument,win=f?.contentWindow;if(!d)return null;'+code+'})()')
        def until(code,seconds=60):
            end=time.monotonic()+seconds
            while time.monotonic()<end:
                if local('return '+code):return
                pause(80)
            raise AssertionError(code+'; '+str(local('return [d.body.dataset.engineState,d.querySelector("#frameStatus")?.textContent,d.querySelector("#videoStatus")?.textContent]')))
        def settled():until('d.body.dataset.engineState==="ready"&&d.body.dataset.posePending==="false"')
        def click(selector):
            assert local('const e=d.querySelector('+json.dumps(selector)+');if(!e||e.disabled)return false;e.click();return true;'),selector
            pause();settled()
        def change(selector,value):
            assert local('const e=d.querySelector('+json.dumps(selector)+');if(!e||e.disabled)return false;e.value='+json.dumps(value)+';e.dispatchEvent(new Event("change"));return true;'),selector
            pause();settled()
        def sequence():return json.loads((scratch/'profile/keyframes.json').read_text('utf-8'))
        def wait_saved(count):
            end=time.monotonic()+15
            while time.monotonic()<end:
                if (scratch/'profile/keyframes.json').is_file() and len(sequence()['frames'])==count:return
                pause()
            raise AssertionError('sequence persistence timeout')
        def export(name,views):
            nonlocal target
            target=scratch/(name+'.mp4')
            click('#exportVideo');change('#videoViews',views);click('#videoStart')
            until('d.querySelector("#videoStatus").textContent.startsWith("已保存")||d.querySelector("#videoStatus").textContent.includes("失败")',150)
            status=local('return d.querySelector("#videoStatus").textContent')
            assert target.is_file(),status
            probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(target)]))
            v=next(s for s in probe['streams'] if s['codec_type']=='video');a=next(s for s in probe['streams'] if s['codec_type']=='audio')
            assert v['codec_name']=='h264' and a['codec_name']=='aac'
            expected=round(sum(f['duration'] for f in sequence()['frames'])*48000)/48000
            assert abs(float(v['duration'])-expected)<1/48000 and abs(float(a['duration'])-expected)<1/48000
            packets=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v','-show_packets','-of','json',str(target)]))['packets']
            assert len(packets)==int(np.ceil(expected*30))
            assert all(abs(float(p['pts_time'])-i/30)<1e-6 for i,p in enumerate(packets))
            assert abs(float(packets[-1]['pts_time'])+float(packets[-1]['duration_time'])-expected)<1e-6
            decoded=np.frombuffer(subprocess.check_output(['ffmpeg','-v','error','-i',str(target),'-map','0:a:0','-f','f32le','-acodec','pcm_f32le','-']),dtype='<f4')
            prepared=w.vocal.invoke('animation/prepare',dict(**sequence(),keep_vowel=True))
            assert prepared['cached']
            payload=w.vocal.invoke('animation/audio',{'id':prepared['id']})
            reference=np.frombuffer(base64.b64decode(payload['base64']),dtype='<f4')
            decoded=decoded[:len(reference)]
            assert len(decoded)==len(reference)
            if np.max(np.abs(reference))>0:
                correlation=float(np.corrcoef(decoded,reference)[0,1]);assert correlation>.95,correlation
            else:correlation=None;assert np.max(np.abs(decoded))<1e-8
            subprocess.run(['ffmpeg','-v','error','-i',str(target),'-frames:v','1','-y',str(out/(name+'.png'))],check=True)
            report['checks'].append(dict(video=name,dimensions=[v['width'],v['height']],frames=len(packets),duration=expected,correlation=correlation,bytes=target.stat().st_size))
            click('#videoClose')
        try:
            settled()
            local("const script=d.createElement('script');script.type='module';script.textContent=\"import {viewer} from './app.js';window.qaViewer=viewer;\";d.body.append(script);return true;")
            until('!!win.qaViewer?.state')
            assert local('return d.querySelector("#sectionCount").value')=='129'
            for count in [65,257,129]:
                change('#sectionCount',str(count))
                assert local('return win.qaViewer.state.centerline.length')==count
                assert local('return Number(win.localStorage.getItem("m10-section-count"))')==count
            report['checks'].append('Qt selection changes real centerline arrays and persists 65/257/129')
            w.vocal.invoke('audio/settings',{'volume':0})
            click('#tab-motion');click('#captureFrame');wait_saved(1)
            change('#keyframeList [data-time-unit=s]',.05)
            assert not local('return d.querySelector("#playFrames").disabled||d.querySelector("#exportVideo").disabled')
            local('d.querySelector("#pitchCanvas").dispatchEvent(new KeyboardEvent("keydown",{key:"ArrowUp",bubbles:true}));return true;')
            until('d.querySelector("#pitchStatus").textContent.includes("手绘")')
            click('#pitchReset')
            click('#playFrames');until('!d.body.classList.contains("motion-busy")')
            assert '失败' not in local('return d.querySelector("#frameStatus").textContent')
            report['checks'].append('one 50 ms frame allows F0 editing and actual zero-volume native playback')
            # Video tests restore monitor gain only while exporting offline.
            w.vocal.invoke('audio/settings',{'volume':.8})
            export('single-current','current');export('single-six','six')
            click('#clearFrames');click('#captureSilence');wait_saved(1)
            change('#keyframeList [data-time-unit=s]',.05)
            export('silent-current','current')
            click('#clearFrames');click('#captureFrame');click('[data-preset=i]');click('#captureFrame');wait_saved(2)
            change('#keyframeList .pose-card:first-child [data-time-unit=s]',.3)
            change('#keyframeList .pose-card:last-child [data-time-unit=s]',.3)
            export('two-pose-current','current')
            click('#clearFrames');click('[data-preset=a]');click('#captureSilence');click('[data-preset=i]');click('#captureFrame');wait_saved(2)
            change('#keyframeList .pose-card:first-child [data-time-unit=s]',.4)
            change('#keyframeList .pose-card:last-child [data-time-unit=s]',.2)
            export('silent-transition-current','current')
            saved=sequence();prepared=w.vocal.invoke('animation/prepare',{'frames':saved['frames'],'pitch_curve':saved['pitch_curve']})
            states=[w.vocal.invoke('animation/picture',{'id':prepared['id'],'index':i})['state'] for i in [0,6,12]]
            assert states[0]['silent'] and states[1]['silent'] and not states[2].get('silent')
            assert states[0]['contours']['tongue']!=states[1]['contours']['tongue']!=states[2]['contours']['tongue']
            np.testing.assert_allclose(states[1]['params'],(np.array(states[0]['params'])+states[2]['params'])/2,atol=1e-9)
            w.vocal.invoke('audio/settings',{'volume':0})
            click('#playFrames');until('!d.body.classList.contains("motion-busy")')
            assert '失败' not in local('return d.querySelector("#frameStatus").textContent')
            report['checks'].append('silent frame reaches the next pose continuously in cached geometry, native playback and MP4; its audio interval remains gated')
            click('#clearFrames');assert local('return d.querySelector("#playFrames").disabled&&d.querySelector("#exportVideo").disabled')
            # Intercept only the picker to verify the actual host default args.
            captured=[];original=QFileDialog.getSaveFileName
            QFileDialog.getSaveFileName=lambda *args,**kw:(captured.append(args) or ('',''))
            try:
                w.bridge.test_dialog=False;w.bridge.vocal('qa-picker',json.dumps({'op':'video/begin','body':dict(width=1280,height=720,fps=30,duration=.05)}));pause()
            finally:QFileDialog.getSaveFileName=original;w.bridge.test_dialog=True
            assert captured and captured[0][2]=='声道动作.mp4' and captured[0][3]=='MP4 视频 (*.mp4)'
            report['checks'].append('host picker defaults to .mp4; empty sequence stays disabled')
            for mode,label in [('light','浅色'),('dark','深色')]:
                assert js('(()=>{const e=document.querySelector("button[title=设置]");if(!e)return false;e.click();return true;})()')
                pause()
                assert js('(()=>{const e=[...document.querySelectorAll(".mode-choices button")].find(e=>e.textContent.trim()=='+json.dumps(label)+');if(!e)return false;e.click();return true;})()')
                pause();js('document.querySelector("#tab-M10").click()');pause(300)
                assert local('return d.documentElement.dataset.theme')==mode
                w.view.grab().save(str(out/(mode+'.png')))
            report['checks'].append('actual settings controls propagate light/dark theme to the embedded M10 view')
            report['passed']=True
        except Exception:
            w.view.grab().save(str(out/'failure.png'));raise
        finally:
            report['dll_sha256']=hashlib.sha256((ROOT/'resources/vocal_tract/native/geometry_p2.dll').read_bytes()).hexdigest()
            write_json(out/'report.json',report)
            w.vocal.close();w.close();w.deleteLater();pause(500)
        print(json.dumps({'out':str(out),**report},ensure_ascii=False))


if __name__=='__main__':
    try:main()
    except Exception:
        if '--qt-child' in sys.argv:
            import traceback
            (Path(sys.argv[2])/'child-error.log').write_text(traceback.format_exc(),encoding='utf-8')
        raise
