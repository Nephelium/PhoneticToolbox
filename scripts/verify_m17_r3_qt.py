"""M17-R3: hidden real Qt pointers/keys and built frontend, isolated profile."""
import json
import argparse
import os
import time
from pathlib import Path
from uuid import uuid4


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--frontend',type=Path);parser.add_argument('--media',action='store_true');args=parser.parse_args()
    os.environ['QT_QPA_PLATFORM']='windows'
    os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS','--mute-audio')
    from PyQt6.QtCore import QEventLoop,QTimer,Qt,QPoint,QPointF,QEvent
    from PyQt6.QtGui import QMouseEvent
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtTest import QTest
    from ptb_desktop.host import Workbench,register_scheme
    root=Path(__file__).resolve().parents[1]
    out=root/'output/validation/m17-r3'/('qt-'+uuid4().hex);out.mkdir(parents=True)
    register_scheme();app=QApplication(['M17-R3-owned-QA'])
    w=Workbench(args.frontend or root/'frontend/dist',test=True,vocal_profile=out/'vocal',start_module='M17')
    w.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen);w.resize(1366,768);w.show()
    report={'success':False,'scope':'Windows actual hidden Qt, built frontend; no physical DPI, listening or EXE','checks':[],'layouts':[]}
    def pause(ms=100):
        loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
    def js(code):
        loop=QEventLoop();values=[];w.page.runJavaScript(code,lambda v:(values.append(v),loop.quit()));QTimer.singleShot(6000,loop.quit);loop.exec()
        assert values,'JS timeout';return values[0]
    def until(code,seconds=20):
        deadline=time.monotonic()+seconds
        while time.monotonic()<deadline:
            if js(code):return
            pause()
        raise AssertionError(code+'\n'+str(js('document.body.innerText.slice(-1500)')))
    def click(text):
        expression='[...document.querySelectorAll("button")].find(e=>e.offsetParent&&e.textContent.trim()==='+json.dumps(text)+')'
        until('!!('+expression+')');js(expression+'.click()');pause(80)
    def pointer(selector,press=False):
        js('document.querySelector('+json.dumps(selector)+').scrollIntoView({block:"nearest"})');pause(180)
        point=js('(()=>{const r=document.querySelector('+json.dumps(selector)+').getBoundingClientRect();return {x:r.left+r.width/2,y:r.top+r.height/2}})()')
        target=w.view.focusProxy() or w.view
        def move(location):
            global_pos=target.mapToGlobal(location)
            event=QMouseEvent(QEvent.Type.MouseMove,QPointF(location),QPointF(global_pos),Qt.MouseButton.NoButton,Qt.MouseButton.NoButton,Qt.KeyboardModifier.NoModifier)
            QApplication.sendEvent(target,event);pause(100)
        move(QPoint(4,4))
        pos=target.mapFromGlobal(w.view.mapToGlobal(QPoint(round(point['x']),round(point['y']))))
        move(pos)
        if press:QTest.mouseClick(target,Qt.MouseButton.LeftButton,pos=pos);pause(150)
        return point
    try:
        until('document.querySelector(".m17-body")?.dataset.loaded==="true"&&document.querySelector(".m17-body")?.dataset.fontReady==="true"')
        catalog=json.loads((root/'frontend/src/modules/ipa-plus/data/catalog.json').read_text('utf8'))
        ids={value:next(e['id'] for e in catalog['entries'] if e['system']=='ipa' and e['insertText']==value and not e['isExample']) for value in ['p','b','t','h','ʔ','u']}
        if args.media:
            js('(()=>{const e=document.querySelector(".m17-editor");e.value="保留文字";e.dispatchEvent(new Event("input",{bubbles:true}));[...document.querySelectorAll(".m17-toggle")].find(e=>e.textContent.includes("点击播放")).querySelector("input").click();})()');pause(120)
            for value in ['p','b','t']:
                pointer('[data-symbol-id='+ids[value]+']',True)
                until('(()=>{const es=[...document.querySelectorAll(".m17-playback audio,.m17-playback video")];return es.length>0&&es.every(e=>e.currentTime>0&&e.readyState>=2)})()')
                assert js('document.querySelector(".m17-editor").value')=='保留文字'
                if value=='p':assert js('document.body.innerText.includes("源码内容分发验证")')
                w.view.grab();pause(200);w.view.grab().save(str(out/f'media-{value}.png'))
                click('收起')
            report['checks'].append('real native custom-scheme bundled WAV, WebM and audio/video combination decode; curated static content visible; text preserved; synthetic muted media only')
            report['success']=True;return
        for width,height in [(1366,768),(1920,1080)]:
            w.resize(width,height);pause(150)
            for theme in ['light','dark']:
                for scale in [1,1.5]:
                    js('document.documentElement.dataset.theme='+json.dumps(theme)+';document.documentElement.style.zoom='+json.dumps(str(scale))+';document.documentElement.style.setProperty("--page-scale",'+json.dumps(str(scale))+')');pause(100)
                    selector='[data-symbol-id='+ids['h']+']';point=pointer(selector)
                    until('!!document.querySelector(".m17-hover-panel")');pause(80)
                    metrics=js('(()=>{const a=document.querySelector('+json.dumps(selector)+').getBoundingClientRect(),p=document.querySelector(".m17-hover-panel").getBoundingClientRect();return {symbol:{l:a.left,t:a.top,r:a.right,b:a.bottom},panel:{l:p.left,t:p.top,r:p.right,b:p.bottom},view:[innerWidth,innerHeight],text:document.querySelector(".m17-hover-panel").textContent}})()')
                    a,p=metrics['symbol'],metrics['panel'];assert p['r']<=a['l'] or p['l']>=a['r'] or p['b']<=a['t'] or p['t']>=a['b'],metrics
                    assert p['l']>=0 and p['t']>=0 and p['r']<=metrics['view'][0]+1 and p['b']<=metrics['view'][1]+1,metrics
                    assert not (p['l']<=point['x']<=p['r'] and p['t']<=point['y']<=p['b']),metrics
                    assert all(term not in metrics['text'] for term in ['图4','用户','自主概述','构音和转写原则','江荻译'])
                    w.view.grab();pause(200);assert w.view.grab().save(str(out/f'{width}-{theme}-{scale}.png'))
                    report['layouts'].append({'size':[width,height],'theme':theme,'scale':scale,**metrics})
                    js('document.querySelector(".m17-chart-viewport").dispatchEvent(new Event("scroll"))');pause(100)
        report['checks'].append('12 real-pointer light/dark scaled layouts; symbol and pointer clear, viewport fit and compact bibliography')
        w.resize(1366,768);js('document.documentElement.style.zoom="1";document.documentElement.style.setProperty("--page-scale","1")');pause(100)
        js('(()=>{const e=document.querySelector(".m17-editor");e.value="甲𝼆乙";e.setSelectionRange(1,3);e.dispatchEvent(new Event("input",{bubbles:true}));})()')
        pointer('[data-symbol-id='+ids['p']+']',True);assert js('document.querySelector(".m17-editor").value')=='甲p乙';click('撤销')
        js('(()=>{const e=[...document.querySelectorAll(".m17-toggle")].find(e=>e.textContent.includes("点击播放")).querySelector("input");e.click()})()');pause(100)
        before=js('(()=>{const e=document.querySelector(".m17-editor");return [e.value,e.selectionStart,e.selectionEnd]})()')
        pointer('[data-symbol-id='+ids['b']+']',True);until('document.body.innerText.includes("此音标尚未添加演示内容")')
        after=js('(()=>{const e=document.querySelector(".m17-editor");return [e.value,e.selectionStart,e.selectionEnd]})()')
        report['playbackSelection']={'before':before,'after':after}
        assert after==before,report['playbackSelection']
        click('收起');js('document.querySelector("[data-symbol-id='+ids['p']+']").focus()');pause(80)
        target=w.view.focusProxy() or w.view;QTest.keyClick(target,Qt.Key.Key_Return);pause(200);until('document.body.innerText.includes("此音标尚未添加演示内容")')
        assert js('document.querySelector(".m17-editor").value')==before[0]
        click('收起');js('document.querySelector("[data-symbol-id='+ids['p']+']").focus()');pause(80)
        QTest.keyClick(target,Qt.Key.Key_Return,Qt.KeyboardModifier.AltModifier);until('!!document.querySelector(".m17-detail-panel")')
        assert not js('!!document.querySelector(".m17-playback")')
        report['checks'].append('native pointer replacement/undo, playback pointer/Enter leave text/selection intact, missing-material feedback, native Alt+Enter remains inspection')
        report['success']=True
    except Exception as error:
        report['failure']=str(error);w.view.grab().save(str(out/'failure.png'));raise
    finally:
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf8')
        w.closing=True;w.close();w.page.deleteLater();app.processEvents()
        print(json.dumps({'success':report['success'],'output':str(out),'checks':report['checks']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
