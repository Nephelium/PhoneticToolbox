"""Owned Qt window, real native worker and saved screenshots for M10."""
import json
import sys
import time
import uuid
from pathlib import Path
from PyQt6.QtCore import QEventLoop,QTimer
from PyQt6.QtWidgets import QApplication
from ptb_desktop.host import Workbench,register_scheme

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/validation/m10/qt'


def main(*,root=None,out=None):
    global ROOT,OUT
    if root is not None:ROOT=root
    if out is not None:OUT=out
    OUT.mkdir(parents=True,exist_ok=True)
    register_scheme();app=QApplication(['m10-qa']);app.setApplicationName('M10-owned-QA')
    profile=OUT/('profile-'+uuid.uuid4().hex[:8])
    window=Workbench(ROOT/'frontend/dist',test=True,vocal_profile=profile,vocal_resources=ROOT/'resources/vocal_tract/native',start_module='M10')
    window.resize(1650,1000);window.show()
    def js(code):
        loop=QEventLoop();box=[]
        window.page.runJavaScript(code,lambda value:(box.append(value),loop.quit()))
        QTimer.singleShot(5000,loop.quit);loop.exec()
        return box[0] if box else None
    def until(code,seconds=35):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            value=js(code)
            if value:return value
            loop=QEventLoop();QTimer.singleShot(120,loop.quit);loop.exec()
        raise RuntimeError('UI condition timed out: '+code)
    def local(code):return js("(()=>{const d=document.querySelector('iframe')?.contentDocument;if(!d)return null;"+code+'})()')
    def ready():
        settle(180)
        until("document.querySelector('iframe').contentDocument.body.dataset.posePending==='false'")
    def settle(ms=350):
        loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
    def shot(name):
        settle();window.view.grab().save(str(OUT/(name+'.png')))
    try:
        until("!!document.querySelector('button[title=\"声道工作台\"]')")
        print('shell ready',flush=True)
        js("document.querySelector('button[title=\"声道工作台\"]').click()")
        js("document.documentElement.dataset.theme='light'")
        until("document.querySelector('iframe')?.contentDocument?.querySelector('#engineStatus')?.textContent.includes('已连接')",55)
        print('native connected',flush=True)
        assert local("return d.querySelector('[data-role=nasal-reference]')?.dataset.registration==='nasal/3';")
        print(local("return JSON.stringify({status:d.querySelector('#statusText').textContent,controls:d.querySelectorAll('#parameterGroups input').length,handles:[...d.querySelectorAll('[data-handle]')].map(e=>e.dataset.handle)});"),flush=True)
        shot('01-organs-light')
        local("d.querySelector('#columnLayout').value='three';d.querySelector('#columnLayout').dispatchEvent(new Event('change'));return true;")
        until("document.querySelector('iframe').contentDocument.documentElement.dataset.columns==='three'")
        settle()
        columns=json.loads(local("return JSON.stringify(['.studio','.analysis','.inspector'].map(s=>{const r=d.querySelector(s).getBoundingClientRect();return {s,x:r.x,y:r.y,w:r.width,h:r.height}}));"))
        assert columns[0]['x']<columns[1]['x']<columns[2]['x'] and len({r['y'] for r in columns})==1
        print('three independent workspace columns',columns,flush=True)
        shot('02-three-columns')
        local("d.querySelector('[data-preset=i]').click();return true;");ready()
        changed=local("return Number(d.querySelector('#p-TCX').value);")
        local("d.querySelector('#undoButton').click();return true;");ready()
        assert local("return Number(d.querySelector('#p-TCX').value);")!=changed
        local("d.querySelector('#redoButton').click();return true;");ready()
        assert local("return Number(d.querySelector('#p-TCX').value);")==changed
        local("d.querySelector('#resetPose').click();return true;");ready()
        print('preset / undo / redo / reset passed',flush=True)
        local("d.querySelector('[data-preset=n]').click();return true;");ready()
        local("d.querySelector('#tab-motion').click();return true;")
        local("d.querySelector('#captureFrame').click();return true;")
        until("document.querySelector('iframe').contentDocument.querySelectorAll('.pose-card').length>0")
        local("d.querySelector('.pose-name').value='/n/';d.querySelector('.pose-name').dispatchEvent(new Event('change'));return true;")
        local("d.querySelector('[data-preset=a]').click();return true;");ready()
        local("d.querySelector('#captureFrame').click();return true;");settle()
        local("d.querySelector('.pose-card').click();return true;");ready()
        assert local("return Number(d.querySelector('#portRange').value)>.1;")
        local("d.querySelector('[aria-label=\"编辑姿势 1\"]').click();d.querySelector('#tab-organs').click();return true;")
        local("const input=d.querySelector('#p-HY');input.value=-4.4;input.dispatchEvent(new Event('input'));return true;");ready()
        local("d.querySelector('#tab-motion').click();return true;")
        assert local("return !!d.querySelector('[aria-label=\"保存姿势 1\"]');")
        local("d.querySelector('[aria-label=\"保存姿势 1\"]').click();return true;");settle()
        until("!!document.querySelector('iframe').contentDocument.querySelector('[aria-label=\"编辑姿势 1\"]')")
        sequence=json.loads((profile/'keyframes.json').read_text('utf-8'))
        assert sequence['frames'][0]['params'][1]==-4.4
        print('whole-card navigation / cross-tab edit / save / rename persisted',flush=True)
        shot('03-frames')
        local("d.querySelector('#pitchExpand').click();return true;")
        until("document.querySelector('iframe').contentDocument.querySelector('#pitchDialog').open")
        local("const c=d.querySelector('#pitchCanvas');c.focus();c.dispatchEvent(new KeyboardEvent('keydown',{key:'ArrowUp',bubbles:true}));return true;");settle()
        sequence=json.loads((profile/'keyframes.json').read_text('utf-8'));assert len(sequence['pitch_curve'])==201
        shot('04-pitch-expanded')
        local("d.querySelector('#pitchClose').click();d.querySelector('#threeButton').click();return true;")
        window.vocal.invoke('audio/settings',{'volume':0})
        local("d.querySelector('#playFrames').click();return true;")
        until("document.querySelector('iframe').contentDocument.querySelector('#frameStatus').textContent.includes('播放结束')",45)
        print('real generated motion and F0 playback completed',flush=True)
        loop=QEventLoop();QTimer.singleShot(1000,loop.quit);loop.exec();shot('05-three-dimensional')
        js("document.documentElement.dataset.theme='dark'")
        loop=QEventLoop();QTimer.singleShot(300,loop.quit);loop.exec();shot('06-dark')
        local("d.querySelector('#sagittalButton').click();d.querySelector('[data-preset=l]').click();d.querySelector('#tab-organs').click();return true;");ready()
        # Select the observed central-contact section for the fitted lateral.
        local("const s=d.querySelector('#sliceRange');s.value=109;s.dispatchEvent(new Event('input'));return true;");ready();shot('07-lateral-dark')
        local("d.querySelector('[data-preset=t]').click();return true;");ready();shot('08-closure-dark')
        local("d.querySelector('#columnLayout').value='two';d.querySelector('#columnLayout').dispatchEvent(new Event('change'));return true;");settle()
        assert local("return d.querySelector('.analysis').getBoundingClientRect().y>d.querySelector('.studio').getBoundingClientRect().y;")
        shot('09-two-columns-dark')
        local("d.querySelector('#tab-motion').click();return true;")
        for _ in range(20):local("d.querySelector('#captureFrame').click();return true;")
        settle(800)
        assert local("return d.querySelectorAll('.pose-card').length===22&&d.querySelector('#keyframeList').scrollHeight>d.querySelector('#keyframeList').clientHeight;")
        shot('10-scroll-list')
        # Real PortAudio stream, deliberately zero monitoring volume in QA.
        window.vocal.invoke('audio/settings',{'volume':0})
        local("d.querySelector('[data-preset=a]').click();return true;");ready()
        window.vocal.invoke('live',{'active':True});settle(900)
        live=window.vocal.invoke('status');assert live['active'] and not live['audio_error']
        local("d.querySelector('[data-analysis=monitor]').click();return true;");settle(600)
        monitor=window.vocal.invoke('audio/monitor',{'seconds':3,'window_ms':20,'hop_ms':5})
        assert monitor['waveform'] and monitor['spectrogram']
        local("d.querySelector('#monitorExpand').click();return true;");settle()
        assert local("return d.querySelector('#monitorDialog').open;")
        shot('11-monitor-dark')
        local("d.querySelector('#monitorStop').click();return true;");settle(400)
        assert not window.vocal.invoke('status')['active']
        local("d.querySelector('#monitorClose').click();return true;")
        window.vocal.invoke('deactivate')
        print('22 cards with scroll / F0 persistence / real muted output / two-column layout passed',flush=True)
        (OUT/'result.json').write_text(json.dumps({'status':'passed','columns':columns,'profile':str(profile),'audio':live,'frozen':bool(getattr(sys,'frozen',False))},ensure_ascii=False,indent=2),encoding='utf-8')
        print('screenshots complete',flush=True)
    except Exception:
        shot('failure');print(local("return d.body.innerText.slice(-2500);"),flush=True);raise
    finally:
        window.closing=True;window.close();window.page.deleteLater();app.processEvents()


if __name__=='__main__':main()
