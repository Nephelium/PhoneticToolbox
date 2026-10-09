"""M07-R2 hidden native Qt UI; clipboard/browser endpoints record instead of touching OS."""
import json
import os
import time
os.environ.setdefault('QT_QPA_PLATFORM','windows')
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS','--disable-gpu --disable-gpu-compositing --mute-audio')
from PyQt6.QtCore import QEventLoop,QTimer,Qt,QPoint
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication,QFileDialog
from ptb_desktop import host
from verify_m07_wiring import setup,ROOT


def main():
    out,db,cache=setup();inputs=out/'inputs';inputs.mkdir()
    for i in (0,1):(inputs/f'input{i}.wav').write_bytes((ROOT/f'output/validation/m07/baseline/round1/input{i}.wav').read_bytes())
    class Clipboard:
        writes=[]
        def setText(self,text):self.writes.append(text)
        def text(self):return self.writes[-1] if self.writes else ''
    clipboard=Clipboard();opened=[];requests=[]
    original_clipboard=host.QApplication.clipboard;original_open=host.QDesktopServices.openUrl
    host.QApplication.clipboard=lambda:clipboard
    host.QDesktopServices.openUrl=lambda url:(opened.append(url.toString()) or True)
    host.register_scheme();app=QApplication(['M07-R2-owned-QA'])
    w=host.Workbench(ROOT/'frontend/dist',test=True,jobs_path=db,local_files_root=cache,vocal_profile=out/'vocal')
    w.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen);w.resize(1920,1080);w.show();w.page.setAudioMuted(True)
    w.page.newWindowRequested.connect(lambda r:requests.append(dict(url=r.requestedUrl().toString(),user=r.isUserInitiated())))
    QFileDialog.getExistingDirectory=lambda *a,**k:str(inputs)
    report=dict(success=False,checks=[],layouts=[],scope='Windows native hidden Qt, built frontend, QTest pointer/Enter; clipboard and external browser endpoint records only; no physical DPI/audio',schema_applied=[])
    def pause(ms=100):
        loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
    def js(code):
        loop=QEventLoop();box=[];w.page.runJavaScript(code,lambda r:(box.append(r),loop.quit()));QTimer.singleShot(6000,loop.quit);loop.exec()
        assert box,'JS timeout';return box[0]
    def until(code):
        deadline=time.monotonic()+55
        while time.monotonic()<deadline:
            if js(code):return
            pause()
        raise AssertionError(code+' '+str(js('document.body.innerText.slice(-3000)')))
    def click(text):
        assert js('(()=>{const e=[...document.querySelectorAll("button")].find(e=>e.offsetParent&&e.textContent.trim()==='+json.dumps(text)+');if(!e||e.disabled)return false;e.click();return true})()'),text
        pause()
    def native(selector,key=False):
        point=js('(()=>{const e=document.querySelector('+json.dumps(selector)+');e.focus();const r=e.getBoundingClientRect();return [r.x+r.width/2,r.y+r.height/2]})()');pause()
        receiver=w.view.focusProxy() or w.view
        if key:QTest.keyClick(receiver,Qt.Key.Key_Return)
        else:QTest.mouseClick(receiver,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,QPoint(round(point[0]),round(point[1])))
        pause(300)
    try:
        until('!!document.querySelector("nav")&&document.fonts.status==="loaded"');click('发声类型合成');until('!!document.querySelector(".m07-attribution")')
        expected='Lu, Y., Liang, C., & Kong, J. (2025). Contribution of F0 and phonation to tone perception in the Zaiwa language. Journal of Phonetics, 110, 101413. https://doi.org/10.1016/j.wocn.2025.101413'
        assert js('document.querySelector(".m07-attribution .paper-citation").textContent')==expected
        native('.m07-attribution button');assert clipboard.writes==[expected]
        until('document.querySelector(".m07-attribution [role=status]")?.textContent==="引用已复制"')
        native('.m07-attribution a');native('.m07-attribution a:nth-child(2)',key=True)
        assert opened==['https://doi.org/10.1016/j.wocn.2025.101413','https://github.com/Luyao2025/Contribution-of-F0-and-phonation-to-tone-perception-in-the-Zaiwa-language']
        assert len(requests)==2 and all(r['user'] for r in requests)
        report['checks'].append('native footer copy through QWebChannel preserves full DOI citation; pointer DOI and Enter repository route to browser exactly once')
        native('.m07-attribution button:nth-of-type(2)');until('document.querySelector(".dialog-header")?.textContent.includes("发声类型合成改写说明")')
        assert js('document.querySelector("dialog[open]").textContent.includes("REAPER 算法作为 PhoneticToolbox 新增")')
        click('完整方法与来源');until('!!document.querySelector(".reference-row")');js('document.querySelectorAll(".reference-groups button")[1].click()')
        until('[...document.querySelectorAll(".reference-row")].some(e=>e.textContent.includes("载瓦语")&&e.textContent.includes("（2026-09-10）作者邮件许可"))');js('document.querySelector(".dialog-header button").click()');pause()
        report['checks'].append('native adaptation dialog explains Python/F0 and opens the complete module references with updated permission')
        click('打开音频目录');until('document.querySelector("select[aria-label=源音频]").options.length===3')
        for label,index in [('源音频',1),('目标音频',2)]:
            js('(()=>{const e=document.querySelector('+json.dumps('select[aria-label="'+label+'"]')+');e.value=e.options['+str(index)+'].value;e.dispatchEvent(new Event("change",{bubbles:true}))})()')
        until('[...document.querySelectorAll("button")].some(e=>e.textContent.trim()==="提取 F0"&&!e.disabled)');click('提取 F0');until('document.querySelector(".m07-page").getAttribute("aria-busy")==="false"')
        js('(()=>{const e=document.querySelector("[aria-label=连续统步数]");e.value="3";e.dispatchEvent(new Event("input",{bubbles:true}))})()');click('生成当前');until('document.querySelectorAll(".f0-plot path[data-curve=synthesis]").length===3')
        for width,height in [(1920,1080),(1440,900),(1280,800),(960,720)]:
            w.resize(width,height);pause(200)
            for theme in ('light','dark'):
                js('document.documentElement.dataset.theme='+json.dumps(theme));w.view.repaint();pause(1000)
                geo=js('(()=>{const r=document.querySelector(".m07-attribution").getBoundingClientRect(),b=document.querySelector(".m07-workspace").getBoundingClientRect();return {footer:{x:r.x,y:r.y,width:r.width,bottom:r.bottom,height:r.height},workspace:{x:b.x,width:b.width,bottom:b.bottom},screen:innerHeight,plots:[...document.querySelectorAll(".m07-plots>.module-section")].map(e=>e.getBoundingClientRect().bottom)}})()')
                assert geo['footer']['y']>=geo['workspace']['bottom']-1 and geo['footer']['bottom']<=geo['screen']+1,geo
                assert abs(geo['footer']['x']-geo['workspace']['x'])<1 and abs(geo['footer']['width']-geo['workspace']['width'])<1
                if width>=1440:assert all(bottom<=geo['footer']['y'] for bottom in geo['plots']),geo
                js('document.querySelector(".m07-workspace").scrollTop=10000');pause();assert abs(js('document.querySelector(".m07-attribution").getBoundingClientRect().y')-geo['footer']['y'])<1
                js('document.querySelector(".m07-workspace").scrollTop=0');w.view.grab().save(str(out/f'r2-qt-{width}-{theme}.png'));report['layouts'].append(dict(width=width,height=height,theme=theme,**geo))
        w.resize(1920,1080);w.view.setZoomFactor(1.5);js('document.documentElement.dataset.theme="light"');w.view.repaint();pause(1200)
        assert js('document.querySelector(".m07-attribution").getBoundingClientRect().bottom<=innerHeight+1')
        w.view.grab().save(str(out/'r2-qt-150percent.png'));report['checks'].append('eight native light/dark layouts and Qt 150 percent zoom: footer visible below all columns during internal scrolling')
        report.update(success=True,clipboard_writes=clipboard.writes,opened=opened,requests=requests)
    finally:
        (out/'r2-qt-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
        w.closing=True;w.close();w.page.deleteLater();app.processEvents();app.quit()
        host.QApplication.clipboard=original_clipboard;host.QDesktopServices.openUrl=original_open
        print(json.dumps(dict(out=str(out),success=report['success'],checks=report['checks']),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
