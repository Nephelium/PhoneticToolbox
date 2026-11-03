"""Real localized M10 manual scenes; no audio playback or video export."""
from pathlib import Path
import argparse,hashlib,json,sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from verification_artifacts import verification_run
from manual.capture_a_support import Capture

def run(args):
    with verification_run('m10-manual','localized-'+args.locale,'Actual maximized Qt/native engine; real model controls and own saved drafts; light/DPR1.5/page100; no playback or export')as(out,scratch):
        c=Capture(args,out,scratch,'m10','vocal-tract');print(out,flush=True)
        for p in (ROOT/'frontend/public/vocal-tract').iterdir():
            if p.is_file()and p.suffix in ('.js','.mjs','.html','.css'):c.sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        for code in ('zh-CN',args.locale):
            p=ROOT/f'frontend/src/i18n/locales/{code}/vocal-tract-frame.json';c.sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        try:
            c.until('document.querySelector(".vocal-page iframe")?.contentDocument?.body?.dataset.engineState==="ready"');c.frame_ready();c.prepare()
            c.until('document.querySelector(".vocal-page iframe").contentDocument.documentElement.dataset.fontLocale==='+json.dumps(args.locale))
            c.frame_click('[data-preset="a"]');c.frame_ready();c.shot('m10-overview-light')
            # Drag the real sagittal control with Qt pointer input.
            from PyQt6.QtCore import QPoint,Qt
            from PyQt6.QtTest import QTest
            point=c.frame('const r=d.querySelector("[data-handle=blade] circle").getBoundingClientRect(),p=f.getBoundingClientRect();return [p.x+r.x+r.width/2,p.y+r.y+r.height/2];')
            before=c.frame('return d.body.dataset.poseRevision;')
            a=QPoint(round(point[0]),round(point[1]));b=QPoint(a.x()+12,a.y()-10)
            target=c.window.view.focusProxy()or c.window.view
            QTest.mousePress(target,Qt.MouseButton.LeftButton,pos=a);QTest.mouseMove(target,b,150);QTest.mouseRelease(target,Qt.MouseButton.LeftButton,pos=b);c.pause(350);c.frame_ready()
            assert c.frame('return d.body.dataset.poseRevision;')!=before,'Real drag did not update the posture'
            c.shot('m10-tongue-drag-light',{'model':'frame:#viewport','tongueControls':'frame:#parameterGroups'})
            c.frame_click('[data-select-organ="lips"]');c.frame_fill('#lipWidthRange',130);c.frame_ready()
            c.shot('m10-lips-controls-light',{'lips':'frame:.lip-width-panel','parameters':'frame:#parameterGroups'})
            c.frame_click('[data-select-organ="velum"]');c.frame_fill('#portRange',.35);c.frame_ready()
            c.frame('d.querySelector("#velumControls").scrollIntoView({block:"center"});');c.pause(200)
            c.shot('m10-velum-port-light',{'port':'frame:#velumControls','crossSection':'frame:.section-stage'})
            c.frame_click('[data-select-organ="tongue"]');c.frame_fill('#p-TS3',-.6);c.frame_ready()
            c.frame_click('[data-side-region="3"]');c.pause(150)
            c.shot('m10-lateral-section-light',{'crossSection':'frame:.section-stage'})
            c.frame_click('#threeButton');c.pause(450);c.shot('m10-three-half-light')
            c.frame_click('#fullModelToggle');c.pause(450);c.shot('m10-three-full-light')
            c.frame_click('#fullModelToggle');c.frame_click('#sagittalButton');c.frame_click('[data-preset="a"]');c.frame_ready()
            c.frame_click('#tab-sound');c.frame_fill('#f0Range',150);c.frame_ready()
            c.frame('d.querySelector(".inspector").scrollTop=0;');c.shot('m10-sound-source-light',{'source':'frame:#sourceControls','perturbation':'frame:#perturbationControls','phonation':'frame:.phonation-buttons','frequency':'frame:.audition-controls'})
            c.frame_click('#savePreset');c.frame_click('#presetInputButton');c.frame_fill('#presetInput','ã');c.shot('m10-save-ipa-light',{'input':'frame:#presetInputRow','dialog':'frame:#presetDialog'})
            c.frame_click('#presetInputSave');c.until('document.querySelector(".vocal-page iframe").contentDocument.querySelector("#presetStatus").textContent.includes("ã")')
            c.frame_click('#presetCancel');c.frame_click('#customPreset');c.shot('m10-load-ipa-light');c.frame_click('#presetCancel')
            c.frame_click('[data-preset="a"]');c.frame_ready();c.frame_click('#tab-sound');c.frame_fill('#f0Range',150);c.frame_ready();c.frame_click('#tab-motion');c.frame_click('#captureFrame');c.frame_fill('.pose-card input[data-time-unit=s]',.8,'change')
            c.frame_click('#captureSilence');c.frame_fill('.pose-card:nth-child(2) input[data-time-unit=s]',.3,'change')
            c.frame_click('[data-preset="u"]');c.frame_ready();c.frame_click('#tab-sound');c.frame_fill('#f0Range',150);c.frame_ready();c.frame_click('#tab-motion');c.frame_click('#captureFrame');c.frame_fill('.pose-card:nth-child(3) input[data-time-unit=s]',.5,'change')
            c.frame('d.querySelector(".ipa-sequence").open=false;d.querySelector("#panel-motion").scrollTop=0;d.querySelector("#keyframeList").scrollTop=0;')
            c.until('!document.querySelector(".vocal-page iframe").contentDocument.querySelector("#toast").classList.contains("show")')
            c.shot('m10-keyframes-silence-light',{'cards':'frame:#keyframeList','transport':'frame:.transport','pitch':'frame:#pitchEditor'})
            c.frame_click('.pose-card .pose-tools button');c.frame_ready();c.frame_click('#tab-motion')
            c.shot('m10-edit-keyframe-light',{'cards':'frame:#keyframeList'})
            c.frame_click('.pose-card .pose-tools button');c.pause(300)
            c.frame_click('#pitchExpand');c.pause(400)
            c.frame('const canvas=d.querySelector("#pitchCanvas");canvas.focus();for(let i=0;i<15;i++)canvas.dispatchEvent(new KeyboardEvent("keydown",{key:"ArrowRight",bubbles:true}));canvas.dispatchEvent(new KeyboardEvent("keydown",{key:"ArrowUp",bubbles:true}));')
            c.shot('m10-f0-expanded-light',{'pitch':'frame:#pitchDialog'})
            c.frame_click('#pitchClose');c.frame_click('#exportVideo');c.shot('m10-video-dialog-light',{'video':'frame:#videoDialog'});c.frame_click('#videoClose')
            assert c.frame('return d.querySelectorAll(".pose-card").length;')==3
            c.report['checks'].append('Real blade drag changed revision; custom combined IPA saved; three frames total1.6s; pitch edited by keyboard; video options opened without playback/export')
            c.report['success']=True
        finally:c.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--locale',choices=('zh-CN','en','ja'),required=True);parser.add_argument('--frontend-dist',type=Path,required=True);run(parser.parse_args())
