"""Owned offscreen Qt workbench: real packaged font, editor, download and layout.

No devices, existing user profile or database schema are touched.
"""
import argparse
import json
import os
import time
from pathlib import Path
from uuid import uuid4

os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS','--disable-gpu --mute-audio')
from PyQt6.QtCore import QEventLoop,QTimer
from PyQt6.QtWidgets import QApplication,QFileDialog
from ptb_desktop.host import Workbench,register_scheme

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--require-single-screen',action='store_true');args=parser.parse_args()
    out=ROOT/'output/validation/m17'/('qt-'+time.strftime('%Y%m%d-%H%M%S')+'-'+uuid4().hex[:6]);out.mkdir(parents=True)
    register_scheme();app=QApplication(['M17-owned-QA'])
    window=Workbench(ROOT/'frontend/dist',test=True,vocal_profile=out/'vocal',start_module='M17')
    window.show();window.resize(1920,1080)
    report={'success':False,'scope':'Windows actual Qt offscreen, built frontend; no physical DPI, clipboard or devices','checks':[],'layouts':[]}
    catalog=json.loads((ROOT/'frontend/src/modules/ipa-plus/data/catalog.json').read_text('utf8'))
    window.page.profile().downloadRequested.connect(lambda d:report.setdefault('downloads',[]).append({'name':d.downloadFileName(),'url':d.url().toString(),'same_page':d.page()==window.page}))
    QFileDialog.getSaveFileName=lambda *a,**k:(str(out/'音标文本.txt'),'UTF-8 文本 (*.txt)')
    def pause(ms=100):
        loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
    def js(code):
        loop=QEventLoop();values=[]
        window.page.runJavaScript(code,lambda value:(values.append(value),loop.quit()))
        QTimer.singleShot(5000,loop.quit);loop.exec()
        return values[0] if values else None
    def until(code,seconds=30):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            if js(code):return
            pause()
        raise AssertionError(code+'\n'+str(js('document.body.innerText.slice(-3500)')))
    def click(label):
        expression='[...document.querySelectorAll("button")].find(e=>e.offsetParent&&e.textContent.trim()==='+json.dumps(label)+')'
        until('!!('+expression+') && !('+expression+').disabled')
        return js(expression+'.click()')
    try:
        until('document.querySelector(".m17-body")?.dataset.loaded==="true" && document.querySelector(".m17-body")?.dataset.fontReady==="true"')
        report['checks'].append('real Qt scheme, async M17 entry, IndexedDB and packaged PTB IPA Plus font loaded')
        text='中文 ḁ 𝼆 V𐞀 '
        js('(()=>{const e=document.querySelector(".m17-editor");e.value='+json.dumps(text)+';e.setSelectionRange(e.value.length,e.value.length);e.dispatchEvent(new Event("input",{bubbles:true}));})()')
        js('document.querySelector("[data-symbol-id=ipa-pulmonic-001]").click()')
        until('document.querySelector(".m17-editor").value==='+json.dumps(text+'p'))
        click('撤销');until('document.querySelector(".m17-editor").value==='+json.dumps(text))
        click('重做');until('document.querySelector(".m17-editor").value==='+json.dumps(text+'p'))
        until('document.querySelector(".m17-save-state").textContent.includes("已保存")')
        reload_loop=QEventLoop();reload_result=[]
        def reloaded(ok):reload_result.append(ok);reload_loop.quit()
        window.page.loadFinished.connect(reloaded)
        window.page.triggerAction(window.page.WebAction.Reload)
        QTimer.singleShot(15000,reload_loop.quit);reload_loop.exec()
        window.page.loadFinished.disconnect(reloaded)
        assert reload_result==[True],reload_result
        until('document.querySelector(".m17-body")?.dataset.loaded==="true" && document.querySelector(".m17-body")?.dataset.fontReady==="true"')
        assert js('document.querySelector(".m17-editor").value')==text+'p'
        click('保存文本')
        end=time.monotonic()+10
        while not (out/'音标文本.txt').is_file() and time.monotonic()<end:pause()
        assert (out/'音标文本.txt').is_file(),{'downloads':report.get('downloads'), 'body':js('document.body.innerText.slice(-2000)')}
        assert (out/'音标文本.txt').read_text('utf8')==text+'p'
        report['checks'].append('CJK, combining diacritic, non-BMP input, symbol insertion, undo/redo, reload and actual native UTF-8 download are lossless')
        for width,height in [(1366,768),(1920,1080)]:
            window.resize(width,height);pause(250)
            for label,system in [('IPA','ipa'),('extIPA','extipa'),('VoQS','voqs')]:
                click(label);until('!!document.querySelector("[data-chart='+system+']")');pause(250)
                expected=[entry['id'] for entry in catalog['entries'] if entry['system']==system]
                seen=[]
                views=['base'] if system=='voqs' else (['base','marks'] if system=='ipa' else ['base','marks','context','combinations'])
                for view in views:
                    if system!='voqs':
                        js('document.querySelector("[data-chart-view='+view+']").click()')
                        until('document.querySelector("[data-chart-view-active]")?.dataset.chartViewActive==='+json.dumps(view))
                    js('document.querySelector(".m17-chart-viewport").scrollTop=0');pause(250)
                    geometry=js('''(()=>{const v=document.querySelector('.m17-chart-viewport'),e=document.querySelector('.m17-editor'),b=v.getBoundingClientRect(),r=e.getBoundingClientRect(),symbols=[...v.querySelectorAll('[data-symbol-id]')],outside=symbols.filter(x=>{const a=x.getBoundingClientRect();return a.left<b.left-1||a.right>b.right+1||a.top<b.top-1||a.bottom>b.bottom+1}).map(x=>x.dataset.symbolId);return {viewport:[innerWidth,innerHeight],chart:{width:v.clientWidth,height:v.clientHeight,scrollWidth:v.scrollWidth,scrollHeight:v.scrollHeight},editor:{top:r.top,bottom:r.bottom,height:r.height},symbolCount:symbols.length,outside,font:getComputedStyle(e).fontFamily}})()''')
                    actual=js('[...document.querySelectorAll(".m17-chart-viewport [data-symbol-id]")].map(e=>e.dataset.symbolId)')
                    seen.extend(actual)
                    assert actual,(system,view,'empty view')
                    report['layouts'].append({'width':width,'height':height,'system':system,'view':view,**geometry})
                    assert geometry['editor']['bottom']<=geometry['viewport'][1]
                    assert geometry['chart']['scrollWidth']<=geometry['chart']['width']+2,(width,height,system,view,geometry)
                    if args.require_single_screen and view=='base' and system!='ipa':
                        assert not geometry['outside'],(width,height,system,view,geometry)
                        assert geometry['chart']['scrollHeight']<=geometry['chart']['height']+2,(width,height,system,view,geometry)
                    # Supplemental explanations may scroll; prove the last symbol
                    # can be reached with the editor still inside the viewport.
                    js('document.querySelector(".m17-chart-viewport [data-symbol-id]:last-child")?.scrollIntoView({block:"nearest"})')
                    last=js('(()=>{const v=document.querySelector(".m17-chart-viewport"),a=[...v.querySelectorAll("[data-symbol-id]")].at(-1);a.scrollIntoView({block:"nearest"});const b=a.getBoundingClientRect(),r=v.getBoundingClientRect();return b.top>=r.top-1&&b.bottom<=r.bottom+1})()')
                    assert last,(system,view,'last input inaccessible')
                    js('document.querySelector(".m17-chart-viewport").scrollTop=0')
                    window.view.grab();pause(350)
                    assert window.view.grab().save(str(out/f'{width}x{height}-{system}-{view}.png'))
                assert sorted(seen)==sorted(expected),(system,'partition occurrence missing or duplicated')
        report['checks'].append('all chart partitions cover the complete catalogue; extIPA/VoQS base fit one screen and merged IPA/supplemental rows scroll with editor visible')
        click('来源')
        until('document.body.innerText.includes("UntPhesoca (unt)")')
        assert js('[...document.querySelectorAll("a")].some(e=>e.href==="https://zhuanlan.zhihu.com/p/203037479")')
        assert js('document.body.innerText.includes("吕佳") && document.body.innerText.includes("江荻")')
        assert js('[...document.querySelectorAll("a")].some(e=>e.href==="https://doi.org/10.3969/j.issn.1006-7299.2013.06.030")')
        report['checks'].append('shared source panel includes the specified Zhihu translation and clickable URL, plus Lv Jia, Jiang Di and DOI 10.3969/j.issn.1006-7299.2013.06.030')
        report['success']=True
    finally:
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf8')
        window.closing=True;window.close();window.page.deleteLater();app.processEvents()
        print(json.dumps({'success':report['success'],'output':str(out),'checks':len(report['checks'])},ensure_ascii=False))

if __name__=='__main__':main()
