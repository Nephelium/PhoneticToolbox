"""M13-R2: owned hidden native Qt, bundled production frontend, private test state."""
import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path
from uuid import uuid4

os.environ.setdefault('QT_QPA_PLATFORM','windows')
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS','--disable-gpu --disable-gpu-compositing --mute-audio')
from PyQt6.QtCore import QEventLoop,QTimer,Qt,QPoint
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication,QFileDialog
from ptb_desktop.host import Workbench,register_scheme
from ptb_worker.local_acoustic_files import initialize_local_files


def main():
    root=Path(__file__).resolve().parents[1];out=root/'output/validation/m13-r2'/('qt-'+uuid4().hex);out.mkdir(parents=True)
    db=out/'jobs.sqlite3'
    with sqlite3.connect((root/'output/validation/p06/local-state.sqlite3').as_uri()+'?mode=ro',uri=True) as source,sqlite3.connect(db) as target:
        assert not source.execute("SELECT 1 FROM jobs WHERE state IN ('queued','running','cancel_requested')").fetchone()
        source.backup(target)
    cache=out/'cache';cache.mkdir();initialize_local_files(cache)
    register_scheme();app=QApplication(['M13-R2-owned-QA']);w=Workbench(root/'frontend/dist',test=True,jobs_path=db,local_files_root=cache,vocal_profile=out/'vocal-profile')
    w.resize(1440,900);w.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen);w.show()
    report=dict(success=False,checks=[],layouts=[],scope='actual hidden Windows Qt; native QTest pointer/keys; private copied SQLite; no existing schema/data changes, physical DPI or EXE')
    def pause(ms=100):
        loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
    def js(code):
        loop=QEventLoop();box=[];w.page.runJavaScript(code,lambda result:(box.append(result),loop.quit()));QTimer.singleShot(6000,loop.quit);loop.exec()
        assert box,'JS timeout';return box[0]
    def until(code,seconds=40):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            if js(code):return
            pause()
        raise AssertionError(code+' '+str(js('document.body.innerText.slice(-4000)')))
    def click(text):
        assert js('(()=>{const name='+json.dumps(text)+';const e=[...document.querySelectorAll("button")].find(e=>e.offsetParent&&(e.textContent.trim()===name||e.getAttribute("aria-label")===name||(e.matches(".nav-item")&&e.textContent.includes(name))));if(!e||e.disabled)return false;e.click();return true})()'),text
        pause()
    def fill(label,value):
        js('(()=>{const e=document.querySelector('+json.dumps('[aria-label="'+label+'"]')+');e.value='+json.dumps(value)+';e.dispatchEvent(new Event("input",{bubbles:true}));e.dispatchEvent(new Event("change",{bubbles:true}))})()');pause()
    def native_click(selector):
        point=js('(()=>{const e=document.querySelector('+json.dumps(selector)+');e.scrollIntoView({block:"nearest"});const r=e.getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2}})()')
        pause();QTest.mouseClick(w.view.focusProxy() or w.view,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,QPoint(round(point['x']),round(point['y'])));pause(200)
    def snapshot(name):
        pause(500);w.view.grab().save(str(out/name))
    target=out/'native-formatting.png'
    QFileDialog.getSaveFileName=lambda *a,**k:(str(target),'PNG')
    try:
        until('!!document.querySelector("nav")');click('汉字转国际音标');until('!!document.querySelector(".mandarin-ipa-page")')
        assert js('document.querySelector("[aria-label=待转换汉字文本]").value===""&&document.querySelector("[aria-label=转换标准]").value==="Standard Chinese (Beijing)"')
        until('document.querySelector("[aria-label=汉字字体]").options.length>100')
        report['font_count']=js('document.querySelector("[aria-label=汉字字体]").options.length')
        js('localStorage.setItem("ptb.v3.mandarin-ipa.v1.M13",JSON.stringify({version:1,text:"旧银行",standard:"UntPhesoca严",selectedVariants:{"行_2":1}}));localStorage.setItem("ptb.v3.m13-settings-open:M13","false")')
        click('关闭 汉字转国际音标');click('汉字转国际音标');until('!!document.querySelector(".mandarin-ipa-page")')
        assert js('document.querySelector("[aria-label=待转换汉字文本]").value===""&&!!document.querySelector(".m13-settings-section").offsetParent')
        click('恢复本机草稿');until('document.querySelector("[aria-label=待转换汉字文本]").value==="旧银行"');assert js('document.querySelectorAll(".m13-mapped")[2].dataset.value==="x̞ɑ̟ŋ̚˧˥"')
        report['checks'].append('native complete font list, empty Beijing startup, ignored old collapse flag, explicit old draft restoration')
        fill('待转换汉字文本','女略');fill('转换标准','Standard Chinese (Beijing)严');native_click('.m13-tone-toggle')
        assert js('[...document.querySelectorAll(".m13-mapped")].map(e=>e.dataset.value).join("|")==="ny|lye̞"')
        fill('音标颜色','#2345ab');fill('汉字颜色','#ab4523');fill('汉字字体','KaiTi')
        style=js('(()=>{const a=document.querySelector(".m13-ipa"),h=document.querySelector(".m13-hanzi");return {ipa:getComputedStyle(a).fontFamily,hanzi:getComputedStyle(h).fontFamily,ic:getComputedStyle(a).color,hc:getComputedStyle(h).color}})()')
        assert 'PTB-Doulos' in style['ipa'] and 'KaiTi' not in style['ipa'] and 'KaiTi' in style['hanzi'];assert style['ic']=='rgb(35, 69, 171)' and style['hc']=='rgb(171, 69, 35)'
        report['style']=style
        js('window.paint=[];const original=CanvasRenderingContext2D.prototype.fillText;CanvasRenderingContext2D.prototype.fillText=function(text,...args){paint.push({text,font:this.font,color:this.fillStyle});return original.call(this,text,...args)}')
        native_click('.mandarin-ipa-page .module-toolbar-primary>button.primary')
        end=time.monotonic()+30
        while not target.exists() and time.monotonic()<end:pause()
        assert target.exists();pause(300)
        paint=js('window.paint');ipa=[p for p in paint if 'PTB-Doulos' in p['font']];assert [p['text'] for p in ipa]==['ny','lye̞'] and all(p['color']=='#2345ab' for p in ipa)
        hanzi=[p for p in paint if p['text'] in ('女','略')];assert len(hanzi)==2 and all('KaiTi' in p['font'] and p['color']=='#ab4523' for p in hanzi)
        from PyQt6.QtGui import QImage
        im=QImage(str(target)).convertToFormat(QImage.Format.Format_RGBA8888);assert not im.isNull();pixels=im.bits().asstring(im.sizeInBytes())
        assert pixels.count(bytes([35,69,171,255]))>10 and pixels.count(bytes([171,69,35,255]))>10
        report['png']={'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'size':[im.width(),im.height()],'paint':paint}
        report['checks'].append('native QTest tone/export buttons; actual native file save; PNG pixels and fonts match independent choices')
        native_click('[aria-label=待转换汉字文本]');QTest.keyClick(w.view.focusProxy() or w.view,Qt.Key.Key_A,Qt.KeyboardModifier.ControlModifier);QTest.keyClicks(w.view.focusProxy() or w.view,'A1');pause();assert js('document.querySelector("[aria-label=待转换汉字文本]").value==="A1"')
        fill('待转换汉字文本','春江花月夜\n银行女略');click('保存本机草稿 *')
        click('关闭 汉字转国际音标');click('汉字转国际音标');until('!!document.querySelector(".mandarin-ipa-page")');assert js('document.querySelector("[aria-label=待转换汉字文本]").value===""&&document.querySelector("[aria-label=转换标准]").value==="Standard Chinese (Beijing)"');click('恢复本机草稿')
        report['checks'].append('native input keys, complete draft roundtrip, reopening empty with Beijing, manual recovery')
        for width,height in [(1440,900),(1100,700)]:
            for theme in ['light','dark']:
                for mode in ['side-by-side','stacked']:
                    w.resize(width,height);js('document.documentElement.dataset.theme='+json.dumps(theme));js('document.querySelector("input[value='+mode+']").click()');pause(200)
                    geometry=js('(()=>{const r=s=>{const b=document.querySelector(s).getBoundingClientRect();return {left:b.left,right:b.right,top:b.top,bottom:b.bottom}},s=document.querySelector(".m13-settings-section");return {settings:r(".m13-settings-section"),input:r(".m13-input-section"),output:r(".m13-result-section"),scroll:s.scrollHeight,client:s.clientHeight}})()')
                    assert geometry['settings']['right']<=geometry['input']['left']+1
                    if mode=='side-by-side':assert geometry['input']['right']<=geometry['output']['left']+1
                    else:assert geometry['input']['bottom']<=geometry['output']['top']+1
                    assert js('(()=>{const r=document.querySelector(".font-family-select>span").getBoundingClientRect();return r.width>70&&r.height<40})()'),'horizontal font label'
                    report['layouts'].append(dict(width=width,height=height,theme=theme,mode=mode,**geometry));snapshot(f'{theme}-{width}-{mode}.png')
        report['success']=True
    except Exception as error:
        report['error']=repr(error);snapshot('failure.png');raise
    finally:
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(out,flush=True);w.closing=True;w.close();pause()


if __name__=='__main__':main()
