"""M13-R3: verify the actual frozen package using private state and native Qt."""
import hashlib
import json
import os
import time
import traceback
from pathlib import Path


def verify(bundle:Path,out:Path):
    os.environ.setdefault('QT_QPA_PLATFORM','windows')
    os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS','--disable-gpu --disable-gpu-compositing --mute-audio')
    from PyQt6.QtCore import QEventLoop,QTimer,Qt,QPoint
    from PyQt6.QtGui import QImage
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication,QFileDialog
    from ptb_desktop.host import Workbench,register_scheme
    from ptb_worker.local_workspace import prepare_workspace
    out.mkdir(parents=True,exist_ok=False)
    database,cache=prepare_workspace(out/'state',bundle/'backend/migrations')
    register_scheme();app=QApplication(['M13-R3-frozen-QA'])
    w=Workbench(bundle/'frontend/dist',test=True,jobs_path=database,local_files_root=cache,vocal_profile=out/'vocal')
    w.resize(1440,900);w.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen);w.show()
    report=dict(success=False,checks=[],layouts=[],scope='frozen package; hidden native Windows Qt; private state; real pointer/keys and PNG',frontend={})
    for file in sorted((bundle/'frontend/dist/assets').glob('MandarinIpaPage-*')):
        report['frontend'][file.name]=hashlib.sha256(file.read_bytes()).hexdigest()
    def pause(ms=100):
        loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
    def js(code):
        loop=QEventLoop();result=[];w.page.runJavaScript(code,lambda value:(result.append(value),loop.quit()));QTimer.singleShot(6000,loop.quit);loop.exec()
        if not result:raise RuntimeError('JS timeout')
        return result[0]
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
    def pointer(selector):
        p=js('(()=>{const e=document.querySelector('+json.dumps(selector)+');e.scrollIntoView({block:"nearest"});const r=e.getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2}})()');pause()
        QTest.mouseClick(w.view.focusProxy() or w.view,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,QPoint(round(p['x']),round(p['y'])));pause(200)
    def caption():
        return js('(()=>{const l=document.querySelector(".font-family-select"),c=l.querySelector("span"),s=l.querySelector("select"),n=document.querySelector(".m13-font-note"),r=e=>{const b=e.getBoundingClientRect();return {left:b.left,right:b.right,top:b.top,bottom:b.bottom,width:b.width,height:b.height}};l.scrollIntoView({block:"nearest"});return {caption:r(c),select:r(s),note:r(n),lineHeight:parseFloat(getComputedStyle(c).lineHeight),display:getComputedStyle(l).display,font:getComputedStyle(c).fontSize,value:s.value,selectedIndex:s.selectedIndex,selectedText:s.selectedOptions[0]?.textContent,selectStyle:{font:getComputedStyle(s).font,lineHeight:getComputedStyle(s).lineHeight,color:getComputedStyle(s).color}}})()')
    try:
        until('!!document.querySelector("nav")');click('汉字转国际音标');until('!!document.querySelector(".mandarin-ipa-page")')
        assert js('document.querySelector("[aria-label=待转换汉字文本]").value===""&&document.querySelector("[aria-label=转换标准]").value==="Standard Chinese (Beijing)"')
        assert js('[...document.querySelectorAll(".mandarin-ipa-page button")].every(e=>e.textContent.trim()!=="转换设置")')
        until('document.querySelector("[aria-label=汉字字体]").options.length>100')
        report['font_options']=js('document.querySelector("[aria-label=汉字字体]").options.length')
        assert js('(()=>{const s=document.querySelector("[aria-label=汉字字体]");return s.selectedOptions[0]?.textContent==="系统默认"&&[...s.options].filter(o=>o.value==="").length===1})()')
        # System default is the precise state shown in the user's failure image.
        for width,height in [(1440,900),(1000,700)]:
            for theme in ['light','dark']:
                for body in [14,24]:
                    for value in ['', 'KaiTi']:
                        w.resize(width,height);js('document.documentElement.dataset.theme='+json.dumps(theme));js('document.documentElement.style.setProperty("--body-size",'+json.dumps(str(body)+'px')+');document.querySelector("[aria-label=转换与排版宽度]").dispatchEvent(new KeyboardEvent("keydown",{key:"Home",bubbles:true}))');pause(250)
                        fill('汉字字体',value);geo=caption();c,s,n=geo['caption'],geo['select'],geo['note']
                        assert geo['display']=='flex' and c['width']>=150 and c['height']<=geo['lineHeight']+1,geo
                        assert s['top']>=c['bottom']-1 and s['width']>=150 and n['top']>=s['bottom']-1,geo
                        assert js('document.querySelector(".m13-settings-section").getBoundingClientRect().width<=241'),geo
                        assert geo['selectedText']==('系统默认' if value=='' else 'KaiTi · 楷体'),geo
                        assert s['height']>=float(geo['font'].removesuffix('px'))*1.5+7,geo
                        report['layouts'].append(dict(width=width,height=height,theme=theme,body=body,**geo))
                        if value=='':pause(300);w.view.grab().save(str(out/f'{theme}-{width}-{body}-system-default.png'))
        report['checks'].append('16 frozen layouts: system-default/KaiTi, minimum 240px controls, two windows, light/dark, 14/24px body; caption exactly one line and no overlap')
        w.resize(1440,900);js('document.documentElement.dataset.theme="light";document.documentElement.style.setProperty("--body-size","14px");document.querySelector(".m13-workspace").style.setProperty("--panel-right","300px")')
        fill('待转换汉字文本','女略');fill('转换标准','Standard Chinese (Beijing)严');fill('汉字字体','KaiTi');fill('音标颜色','#2345ab');fill('汉字颜色','#ab4523');pointer('.m13-tone-toggle')
        assert js('[...document.querySelectorAll(".m13-mapped")].map(e=>e.dataset.value).join("|")==="ny|lye̞"')
        assert js('getComputedStyle(document.querySelector(".m13-ipa")).fontFamily.includes("PTB-Doulos")&&!getComputedStyle(document.querySelector(".m13-ipa")).fontFamily.includes("KaiTi")')
        target=out/'native-custom.png';QFileDialog.getSaveFileName=lambda *a,**k:(str(target),'PNG')
        pointer('.mandarin-ipa-page .module-toolbar-primary>button')
        end=time.monotonic()+30
        while not target.exists() and time.monotonic()<end:pause()
        assert target.exists();pause(300)
        im=QImage(str(target)).convertToFormat(QImage.Format.Format_RGBA8888);assert not im.isNull();pixels=im.bits().asstring(im.sizeInBytes());assert pixels.count(bytes([35,69,171,255]))>10 and pixels.count(bytes([171,69,35,255]))>10
        report['png']={'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'width':im.width(),'height':im.height()}
        report['checks'].append('native tone button, fixed IPA font, real native PNG save and exact color pixel readback')
        pointer('[aria-label=待转换汉字文本]');QTest.keyClick(w.view.focusProxy() or w.view,Qt.Key.Key_A,Qt.KeyboardModifier.ControlModifier);QTest.keyClicks(w.view.focusProxy() or w.view,'A1');pause();assert js('document.querySelector("[aria-label=待转换汉字文本]").value==="A1"')
        click('保存本机草稿 *');click('关闭 汉字转国际音标');click('汉字转国际音标');until('!!document.querySelector(".mandarin-ipa-page")');assert js('document.querySelector("[aria-label=待转换汉字文本]").value===""&&document.querySelector("[aria-label=转换标准]").value==="Standard Chinese (Beijing)"');click('恢复本机草稿');assert js('document.querySelector("[aria-label=待转换汉字文本]").value==="A1"')
        report['checks'].append('native input keys; saved draft roundtrip; empty Beijing reopen with manual recovery')
        fill('待转换汉字文本','春江花月夜\n银行女略');fill('汉字字体','');w.view.grab().save(str(out/'final-default-font.png'))
        report['success']=True
    except Exception:
        report['error']=traceback.format_exc();w.view.grab().save(str(out/'failure.png'))
    finally:
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');w.closing=True;w.close();pause()
    print(json.dumps({'success':report['success'],'out':str(out)}),flush=True)
    return 0 if report['success'] else 1
