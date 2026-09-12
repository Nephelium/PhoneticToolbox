"""Real, owned Qt verification of portable frames, replay, and encoded videos."""
import hashlib
import json
import sys
import time
import uuid
from pathlib import Path
from PyQt6.QtCore import QEventLoop,QTimer,Qt
from PyQt6.QtWidgets import QApplication
from ptb_desktop.host import Workbench,register_scheme


def main(root=None,out=None):
    root=Path(root or Path(__file__).resolve().parents[1]);out=Path(out or root/'output/validation/m10/r4-qt');out.mkdir(parents=True,exist_ok=True)
    register_scheme();app=QApplication(['m10-recording-qa']);app.setApplicationName('M10-R4-owned-QA')
    profile=out/('profile-'+uuid.uuid4().hex[:8]);w=Workbench(root/'frontend/dist',test=True,vocal_profile=profile,vocal_resources=root/'resources/vocal_tract/native',start_module='M10')
    w.setFixedSize(1650,1000);w.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating);w.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents);w.show()
    def js(code):
        loop=QEventLoop();box=[];w.page.runJavaScript(code,lambda value:(box.append(value),loop.quit()));QTimer.singleShot(5000,loop.quit);loop.exec();return box[0] if box else None
    def wait(ms=150):loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
    def local(code):return js("(()=>{const d=document.querySelector('iframe')?.contentDocument;if(!d)return null;"+code+'})()')
    def until(code,seconds=45):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            value=local('return '+code+';')
            if value:return value
            wait()
        raise RuntimeError('UI timeout: '+code+' '+str(local("return d.querySelector('#frameStatus')?.textContent+' '+d.querySelector('#videoStatus')?.textContent;")))
    def click(selector):local('d.querySelector('+json.dumps(selector)+').click();');wait()
    def input(selector,value,event='change'):local('const e=d.querySelector('+json.dumps(selector)+');e.value='+json.dumps(value)+';e.dispatchEvent(new Event('+json.dumps(event)+'));');wait()
    def ready():until("d.body.dataset.posePending==='false'")
    def shot(name):wait(250);w.view.grab().save(str(out/(name+'.png')))
    def stable_pixels(selector):
        rect=json.loads(local("const r=d.querySelector("+json.dumps(selector)+").getBoundingClientRect(),f=document.querySelector('iframe').getBoundingClientRect();return JSON.stringify([f.x+r.x,f.y+r.y,r.width,r.height]);"))
        hashes=[];pixels=[];geometry=[]
        for _ in range(8):
            wait(125);pix=w.view.grab();ratio=pix.devicePixelRatio();x,y,width,height=[round(v*ratio) for v in rect];image=pix.copy(x,y,width,height).toImage()
            hashes.append(hashlib.sha256(image.bits().asstring(image.sizeInBytes())).hexdigest())
            pixels.append(pix);geometry.append((w.width(),w.height(),ratio,local("return d.querySelector('#viewport').dataset.renderCount;")))
        if len(set(hashes))!=1:
            for i,pix in enumerate(pixels):pix.save(str(out/f'stability-{i}.png'))
        assert len(set(hashes))==1,(selector,hashes,geometry)
        return hashes[0]
    selections={'document/save':out/'演示.ptb-vocal.json','document/open':out/'演示.ptb-vocal.json','video/begin':out/'current.webm'}
    w.bridge.test_vocal_picker=lambda op:str(selections[op])
    result={}
    try:
        until("d.querySelector('#engineStatus')?.textContent.includes('已连接')",55);ready()
        assert local("return d.querySelector('#f0Range').value;")=='150'
        assert not local("return !!d.querySelector('[data-preset=s]');")
        input('#sourceMode','whisper');ready();click('[data-preset=i]');ready();assert local("return d.querySelector('#sourceMode').value;")=='whisper'
        click('#savePreset');input('#presetName','演示 /i/');click('#presetConfirm');until("!d.querySelector('#presetDialog').open")
        assert not local("return d.querySelector('#renamePreset').disabled;")
        click('#renamePreset');until("d.querySelector('#presetDialog').open");input('#presetName','视频 /i/');click('#presetConfirm');until("!d.querySelector('#presetDialog').open")
        custom=json.loads((profile/'presets.json').read_text('utf-8'))['presets'][0]
        assert custom['name']=='视频 /i/'
        click('[data-preset=a]');ready();input('#customPreset',custom['id']);ready()
        assert float(local("return d.querySelector('#p-TCX').value;"))==round(custom['params'][8],2)
        input('#sourceMode','voiced');ready();click('#tab-motion');click('#captureFrame');click('[data-preset=a]');ready();click('#captureFrame')
        until("d.querySelectorAll('.pose-card').length===2")
        assert json.loads(local("return JSON.stringify([...d.querySelectorAll('.pose-card input[type=number]')].map(e=>e.value));"))==['0.2','0.2']
        click('#captureSilence');until("d.querySelectorAll('.pose-card').length===3");click('[aria-label="前移姿势 3"]');input('[aria-label="姿势 2 静音秒数"]',.05)
        click('#threeButton');click('#pitchExpand');until("d.querySelector('#pitchDialog').open")
        local("const c=d.querySelector('#pitchCanvas');c.focus();c.dispatchEvent(new KeyboardEvent('keydown',{key:'ArrowUp',bubbles:true}));");wait(350)
        before_pitch=(profile/'keyframes.json').read_bytes()
        local("const c=d.querySelector('#pitchCanvas');for(let i=0;i<50;i++)c.dispatchEvent(new KeyboardEvent('keydown',{key:'ArrowRight',bubbles:true}));c.dispatchEvent(new KeyboardEvent('keydown',{key:'ArrowUp',bubbles:true}));const r=c.getBoundingClientRect();c.dispatchEvent(new PointerEvent('pointerdown',{clientX:r.left+42+(r.width-54)*.5,clientY:r.top+70,pointerId:1,bubbles:true}));c.dispatchEvent(new PointerEvent('pointerup',{pointerId:1,bubbles:true}));");wait(350)
        assert '静音' in local("return d.querySelector('#pitchCanvas').getAttribute('aria-valuetext');")
        assert (profile/'keyframes.json').read_bytes()==before_pitch
        result['silence_f0_blocked']=True
        shot('pitch-expanded');result['pitch_stable_sha256']=stable_pixels('#pitchCanvas');click('#pitchClose');click('#exportFrames');until("d.querySelector('#frameStatus').textContent.includes('已导出')")
        document=json.loads(selections['document/save'].read_text('utf-8'));assert document['version']==1 and len(document['pitch_curve'])==201 and document['frames'][1]['silent'] and document['frames'][1]['duration']==.05
        click('[aria-label="移除姿势 2"]');click('#importFrames');until("d.querySelectorAll('.pose-card').length===3")
        bad=out/'invalid.json';bad.write_text('{"version":99}',encoding='utf-8');selections['document/open']=bad
        before=(profile/'keyframes.json').read_bytes();click('#importFrames');until("d.querySelector('#frameStatus').textContent.includes('导入失败')");assert before==(profile/'keyframes.json').read_bytes()
        ready();w.vocal.invoke('audio/settings',{'volume':0});click('#playFrames');until("d.querySelector('#frameStatus').textContent.includes('播放结束')",80);ready()
        click('#playFrames');until("d.querySelector('#frameStatus').textContent.includes('已复用')",80);ready();result['replay_cache']=local("return d.querySelector('#playFrames').dataset.cache;")
        click('#threeButton');ready();wait(600)
        count=local("return d.querySelector('#viewport').dataset.renderCount;");hashes=[]
        rect=json.loads(local("const r=d.querySelector('#viewport').getBoundingClientRect();return JSON.stringify({x:r.x,y:r.y,w:r.width,h:r.height});"))
        # Idle render counter proves no timer continuously clears/redraws WebGL.
        for _ in range(8):wait(125);assert local("return d.querySelector('#viewport').dataset.renderCount;")==count
        result['viewport_stable_sha256']=stable_pixels('#viewport')
        result['idle_render_count']=count;shot('idle-3d')
        (out/'current-sequence.json').write_bytes((profile/'keyframes.json').read_bytes())
        w.vocal.invoke('audio/settings',{'volume':.8});click('#exportVideo');click('#videoStart')
        until("d.querySelector('#videoStatus').textContent.startsWith('已保存')||d.querySelector('#videoStatus').textContent.includes('失败')",120)
        result['current']=local("return d.querySelector('#videoStatus').textContent;");print(result['current'],flush=True)
        assert selections['video/begin'].is_file(),result['current'];shot('video-current-done');click('#videoClose')
        input('[aria-label="姿势 2 静音秒数"]',.2);ready();wait(350)
        (out/'six-sequence.json').write_bytes((profile/'keyframes.json').read_bytes())
        assert json.loads((out/'six-sequence.json').read_text('utf-8'))['frames'][1]['duration']==.2
        selections['video/begin']=out/'six.webm';click('#exportVideo');input('#videoViews','six');click('#videoStart')
        until("d.querySelector('#videoStatus').textContent.startsWith('已保存')||d.querySelector('#videoStatus').textContent.includes('失败')",180)
        result['six']=local("return d.querySelector('#videoStatus').textContent;");print(result['six'],flush=True)
        assert selections['video/begin'].is_file(),result['six'];shot('video-six-done');click('#videoClose')
        selections['video/begin']=out/'cancelled.webm';selections['video/begin'].write_bytes(b'preserved-target');click('#exportVideo');click('#videoStart');wait(400);click('#videoClose')
        until("d.querySelector('#videoStatus').textContent.includes('已取消')",60)
        assert selections['video/begin'].read_bytes()==b'preserved-target'
        result.update(passed=True,document_frames=len(document['frames']),curve_points=len(document['pitch_curve']),frozen=bool(getattr(sys,'frozen',False)),profile=str(profile))
        click('#videoClose');click('#tab-organs');click('[data-preset=n]');ready()
        local("d.querySelector('#keepVowel').checked=false;d.querySelector('#keepVowel').dispatchEvent(new Event('change'));");ready()
        for area in [0,.5,1.5]:
            input('#portRange',area,'input');ready();click('[data-mode=airway]');click('#sagittalButton');shot('nasal-'+str(area)+'-2d');click('#threeButton');shot('nasal-'+str(area)+'-3d')
        click('[data-mode=organs]');click('[data-preset=a]');ready();input('#p-TCX',-3,'input');ready()
        assert float(local("return d.querySelector('#p-TCX').value;"))>-1.1
        shot('posterior-limit');click('#tab-sound');w.vocal.invoke('audio/settings',{'volume':0});click('#oneSecondButton')
        until("d.querySelector('#liveButton').classList.contains('active')");until("!d.querySelector('#liveButton').classList.contains('active')")
        result['one_second_samples']=w.vocal.invoke('status')['audio']['generated_samples'];assert result['one_second_samples']==48000
        shot('sound-controls')
        (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False),flush=True)
    except Exception:
        shot('failure');raise
    finally:w.closing=True;w.close();w.page.deleteLater();app.processEvents()


if __name__=='__main__':main()
