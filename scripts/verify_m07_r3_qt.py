"""M07-R3 actual Windows Qt: hidden test window, native gestures, muted audio."""
import hashlib
import json
import os
import time
os.environ.setdefault('QT_QPA_PLATFORM','windows')
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS','--disable-gpu --disable-gpu-compositing --mute-audio')
from PyQt6.QtCore import QEventLoop,QTimer,Qt,QPoint
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication,QFileDialog
from ptb_desktop.host import Workbench,register_scheme
from verify_m07_wiring import setup,ROOT


def main():
    out,db,cache=setup();inputs=out/'inputs';inputs.mkdir();saved=out/'saved';saved.mkdir()
    for i in (0,1):(inputs/f'input{i}.wav').write_bytes((ROOT/f'output/validation/m07/baseline/round1/input{i}.wav').read_bytes())
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.iterdir()}
    register_scheme();app=QApplication(['M07-R3-owned-QA'])
    w=Workbench(ROOT/'frontend/dist',test=True,jobs_path=db,local_files_root=cache,vocal_profile=out/'vocal-profile')
    w.resize(1920,1080);w.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen);w.show();w.page.setAudioMuted(True)
    QFileDialog.getExistingDirectory=lambda *a,**k:str(saved if '结果' in str(a) else inputs)
    print(out,flush=True)
    report=dict(success=False,checks=[],layouts=[],scope='Windows actual Qt hidden; production built frontend; native QTest pointer/keyboard; synthetic audio; muted WebAudio',schema_applied=[])
    def pause(ms=100):
        loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
    def js(code):
        loop=QEventLoop();box=[];w.page.runJavaScript(code,lambda result:(box.append(result),loop.quit()));QTimer.singleShot(6000,loop.quit);loop.exec()
        if not box:raise RuntimeError('JS timeout')
        return box[0]
    def until(code,seconds=70):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            if js(code):return
            pause()
        raise AssertionError(code+' '+str(js('document.body.innerText.slice(-5000)')))
    def click(text):
        assert js('(()=>{const e=[...document.querySelectorAll("button")].find(b=>b.offsetParent&&b.textContent.trim()==='+json.dumps(text)+');if(!e||e.disabled)return false;e.click();return true})()'),text
        pause(100)
    def fill(label,value):
        js('(()=>{const e=document.querySelector('+json.dumps('[aria-label="'+label+'"]')+');e.value='+json.dumps(str(value))+';e.dispatchEvent(new Event("input",{bubbles:true}));e.dispatchEvent(new Event("change",{bubbles:true}))})()')
    def native_click(selector):
        print('native '+selector,flush=True)
        rect=js('(()=>{const e=document.querySelector('+json.dumps(selector)+');e.scrollIntoView({block:"nearest"});const r=e.getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2}})()')
        target=w.view.focusProxy() or w.view;QTest.mouseClick(target,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,QPoint(round(rect['x']),round(rect['y'])));pause(200)
    def idle():until('document.querySelector(".m07-page").getAttribute("aria-busy")==="false"')
    def curves(count):until('document.querySelectorAll(".f0-plot path[data-curve=synthesis]").length==='+str(count))

    downloads=[]
    def accept_download(request):
        if request.downloadFileName().endswith('.png'):
            downloads.append(request)
    QFileDialog.getSaveFileName=lambda *a,**k:(str(out/'r3-native-export-1.png'),'PNG')
    w.page.profile().downloadRequested.connect(accept_download)
    def native_fill(selector,value):
        native_click(selector);receiver=w.view.focusProxy() or w.view
        QTest.keyClick(receiver,Qt.Key.Key_A,Qt.KeyboardModifier.ControlModifier);QTest.keyClicks(receiver,str(value));pause(100)
    try:
        until('!!document.querySelector("nav")');click('发声类型合成');until('!!document.querySelector(".m07-page")')
        w.view.grab().save(str(out/'r3-qt-empty.png'))
        click('打开音频目录');until('document.querySelector("select[aria-label=源音频]").options.length===3')
        for label,index in [('源音频',1),('目标音频',2)]:fill(label,js('document.querySelector('+json.dumps('select[aria-label="'+label+'"]')+').options['+str(index)+'].value'))
        until('[...document.querySelectorAll("button")].some(b=>b.textContent.trim()==="提取 F0"&&!b.disabled)');click('提取 F0');idle();until('document.body.innerText.includes("分析完成，控制点尚未修改")')
        click('生成当前');idle();curves(9);w.view.repaint();pause(1200);print('generated nine steps',flush=True)
        assert js('document.querySelectorAll(".result-audios button").length')==10
        assert not js('document.body.innerText.includes("已载入本组完整音频")')
        native_click('.result-audios button:nth-child(3)');curves(1);until('document.querySelector(".f0-plot").dataset.activeStep==="step02"')
        native_click('.synthesized-plot .audio-transport button:nth-child(2)');until('document.querySelector(".f0-plot").dataset.activeStep===""')
        native_click('.result-audios button:first-child');curves(9);until('document.querySelector(".f0-plot").dataset.activeStep==="step01"')
        print('whole playing',flush=True)
        until('document.querySelector(".f0-plot").dataset.activeStep==="step02"');native_click('.synthesized-plot .audio-transport button:first-child')
        until('document.querySelector(".f0-plot").dataset.activeStep===""')
        assert js('document.querySelector(".synthesized-plot .audio-transport button").textContent.trim()')=='播放选区'
        native_click('.synthesized-plot .audio-transport button:first-child');until('document.querySelector(".f0-plot").dataset.activeStep!==""')
        native_click('.source-plot .audio-transport button:first-child');until('document.querySelector(".f0-plot").dataset.activeStep===""')
        native_click('.source-plot .audio-transport button:nth-child(2)')
        report['checks'].append('native single step/whole starts corresponding F0 highlight; natural first-to-second transition, pause/resume and source playback takeover')
        native_fill('[aria-label="F0 图窗纵轴下限"]',70);native_fill('[aria-label="F0 图窗纵轴上限"]',120)
        until('document.querySelector(".f0-plot").dataset.yMin==="70"&&document.querySelector(".f0-plot").dataset.yMax==="120"')
        assert js('[...document.querySelectorAll("button")].some(b=>b.textContent.trim()==="生成当前"&&!b.disabled)')
        js('window.exportScenes=[];const serializer=XMLSerializer.prototype.serializeToString;XMLSerializer.prototype.serializeToString=function(node){const raw=serializer.call(this,node);if(node.querySelector?.("[data-y-min]")&&node.querySelector("text"))window.exportScenes.push(raw);return raw;}')
        native_click('.main-f0 .module-section-actions button');until('window.exportScenes.length===1')
        deadline=time.monotonic()+20
        while time.monotonic()<deadline and (not downloads or not downloads[0].isFinished()):pause(100)
        assert downloads and downloads[0].isFinished() and downloads[0].state().name=='DownloadCompleted'
        from xml.etree import ElementTree as ET
        import struct
        raw=js('window.exportScenes[0]');(out/'r3-native-export.svg').write_text(raw,encoding='utf8');tree=ET.fromstring(raw)
        plot=next(e for e in tree.iter() if 'data-y-min' in e.attrib)
        assert plot.attrib['data-y-min']=='70' and plot.attrib['data-y-max']=='120'
        text=[e.text for e in tree.iter() if e.tag.endswith('text')]
        assert all(f'step{i:02d}' in text for i in range(1,10)) and '70' in text and '120' in text
        png=(out/'r3-native-export-1.png').read_bytes();assert png[:8]==b'\x89PNG\r\n\x1a\n'
        pixel_width,pixel_height=struct.unpack_from('>II',png,16);offset=8;dpi=None
        while offset<len(png):
            size=struct.unpack_from('>I',png,offset)[0]
            if png[offset+4:offset+8]==b'pHYs':
                x,y,unit=struct.unpack_from('>IIB',png,offset+8);assert unit==1;dpi=[x*.0254,y*.0254]
            offset+=size+12
        assert pixel_width>500 and pixel_height>300 and dpi and abs(dpi[0]-300)<.1
        report['export']=dict(width=pixel_width,height=pixel_height,dpi=dpi,axis=[70,120],legends=9,sha256=hashlib.sha256(png).hexdigest())
        native_fill('[aria-label="F0 图窗纵轴上限"]',60);assert js('!!document.querySelector(".f0-axis-controls [role=alert]")');assert js('document.querySelector(".f0-plot").dataset.yMax')=='120'
        click('自动范围');until('document.querySelector(".f0-plot").dataset.yMin==="0"');report['checks'].append('native typed display range and real Qt PNG download preserve current axis and all nine legends at 300 dpi; invalid range and automatic reset')
        for width,height,right in [(1920,1080,300),(1920,1080,460),(1440,900,300),(1280,800,300)]:
            w.resize(width,height);pause(200)
            if width>=1440:
                for attempt in range(60):
                    current=js("Number(document.querySelector('[aria-label=\"连续统与任务宽度\"]').getAttribute('aria-valuenow'))")
                    if current==right:break
                    js("document.querySelector('[aria-label=\"连续统与任务宽度\"]').focus()")
                    QTest.keyClick(w.view.focusProxy() or w.view,Qt.Key.Key_Left if current<right else Qt.Key.Key_Right);pause(40)
                assert current==right

            for theme in ('light','dark'):
                js('document.documentElement.dataset.theme='+json.dumps(theme));w.view.repaint();pause(1000)
                geo=js('(()=>{const b=e=>{const r=e.getBoundingClientRect();return {x:r.x,y:r.y,width:r.width,height:r.height,bottom:r.bottom}},q=s=>document.querySelector(s);return {body:b(q(".workbench-right-body")),group:b(q(".result-audios")),save:b(q(".result-row>.primary")),footer:b(q(".m07-attribution")),buttons:[...document.querySelectorAll(".generate-actions>.primary")].map(b),screen:innerHeight}})()')
                if width>=1440:assert geo['group']['bottom']<=geo['body']['bottom']+1 and geo['save']['bottom']<=geo['body']['bottom']+1,geo
                if right==460:assert abs(geo['buttons'][0]['y']-geo['buttons'][1]['y'])<1,geo
                assert geo['footer']['bottom']<=geo['screen']+1
                w.view.grab().save(str(out/f'r3-qt-{width}-{right}-{theme}.png'));report['layouts'].append(dict(width=width,height=height,right=right,theme=theme,**geo))
        w.resize(1920,1080);w.view.setZoomFactor(1.5);pause(1200);assert js('document.querySelector(".m07-attribution").getBoundingClientRect().bottom<=innerHeight+1');w.view.grab().save(str(out/'r3-qt-150percent.png'))
        assert hashes=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.iterdir()}
        report['checks'].append('eight native light/dark layouts and Qt 150 percent zoom, compact desktop group visibility and wide two-column actions; original synthetic inputs unchanged');report['success']=True
    finally:
        (out/'r3-qt-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
        js('window.onbeforeunload=null');w.closing=True;w.close();w.page.deleteLater();app.processEvents();app.quit();print(out,flush=True)


if __name__=='__main__':main()
