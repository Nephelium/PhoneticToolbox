"""M10-R9: complete M17 consonants in real native save/load and compact layouts."""
import json
import os
import sqlite3
import time
from pathlib import Path
from uuid import uuid4


def main():
    os.environ['QT_QPA_PLATFORM']='windows'
    os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS','--mute-audio')
    from PyQt6.QtCore import QEventLoop,QTimer,Qt,QPoint,QPointF
    from PyQt6.QtGui import QWheelEvent
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtTest import QTest
    from ptb_desktop.host import Workbench,register_scheme
    from ptb_worker.local_acoustic_files import initialize_local_files
    from ptb_desktop.vocal_tract.runtime import Runtime
    root=Path(__file__).resolve().parents[1]
    out=root/'output/validation/m10-r9'/('qt-'+uuid4().hex);out.mkdir(parents=True)
    db=out/'jobs.sqlite3'
    with sqlite3.connect((root/'output/validation/p06/local-state.sqlite3').as_uri()+'?mode=ro',uri=True) as source,sqlite3.connect(db) as target:
        assert not source.execute("SELECT 1 FROM jobs WHERE state IN ('queued','running','cancel_requested')").fetchone()
        source.backup(target)
    cache=out/'cache';cache.mkdir();initialize_local_files(cache)
    profile=out/'vocal'
    native=Runtime(root/'resources/vocal_tract/native',profile,playback_allowed=False)
    try:
        legacy={'params':native.engine.presets['a'],'name':'旧名称演示','id':'legacy-free','duration':.2}
        slash={'params':native.engine.presets['i'],'name':'/i/','id':'legacy-i','duration':.2}
        native.invoke('presets/save',{'presets':[legacy,slash,{**legacy,'name':'[pʰ]','id':'legacy-aspirate'},{**legacy,'name':'ã','id':'legacy-custom'}]})
        native.profile.save_frames([{**legacy,'name':f'姿势 {i+1}'} for i in range(24)])
    finally:native.close()
    register_scheme();app=QApplication(['M10-R9-owned-QA'])
    w=Workbench(root/'frontend/dist',test=True,jobs_path=db,local_files_root=cache,vocal_profile=profile,start_module='M10')
    w.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen);w.resize(1440,900);w.show()
    report={'success':False,'checks':[],'layouts':[],'scope':'Windows hidden native Qt; isolated profile; no physical audio/DPI or frozen EXE'}
    def pause(ms=100):
        loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
    def js(code):
        loop=QEventLoop();box=[];w.page.runJavaScript(code,lambda value:(box.append(value),loop.quit()));QTimer.singleShot(6000,loop.quit);loop.exec()
        assert box,'JavaScript timeout';return box[0]
    def local(code):
        return js('(()=>{const f=document.querySelector("iframe[title=声道工作台]"),d=f?.contentDocument,win=f?.contentWindow;if(!d)return null;'+code+'})()')
    def until(code,seconds=45):
        deadline=time.monotonic()+seconds
        while time.monotonic()<deadline:
            if local('return '+code):return
            pause()
        raise AssertionError(code)
    def settled():
        until('d.body.dataset.engineState==="ready"&&d.body.dataset.posePending==="false"');pause(180)
    def click(selector):
        assert local('const b=d.querySelector('+json.dumps(selector)+');if(!b||b.disabled)return false;b.click();return true;'),selector
        pause(100)
    def input_value(selector,value,event='input'):
        assert local('const e=d.querySelector('+json.dumps(selector)+');if(!e)return false;e.value='+json.dumps(value)+';e.dispatchEvent(new Event('+json.dumps(event)+',{bubbles:true}));return true;')
        settled()
    def presets():return json.loads((profile/'presets.json').read_text('utf8'))['presets']
    def controls():
        return local('return {source:d.querySelector("#sourceMode").value,pressure:d.querySelector("#pressureRange").value,opening:d.querySelector("#openingRange").value,gap:d.querySelector("#chinkRange").value,f0:d.querySelector("#f0Range").value,width:d.querySelector("#lipWidthRange").value,params:[...d.querySelectorAll("#parameterGroups input")].map(e=>e.value),root:d.querySelector("#manualRoot").checked};')
    def wheel(panel,delta):
        point=local('const r=d.querySelector('+json.dumps(panel)+').getBoundingClientRect(),fr=f.getBoundingClientRect();return {x:fr.x+r.x+r.width/2,y:fr.y+r.y+Math.min(20,r.height/2)};')
        target=w.view.focusProxy() or w.view
        location=QPoint(round(point['x']),round(point['y']));global_point=w.view.mapToGlobal(location);target_point=target.mapFromGlobal(global_point)
        QTest.mouseMove(target,target_point);pause(100)
        event=QWheelEvent(QPointF(target_point),QPointF(global_point),QPoint(0,0),QPoint(0,delta),Qt.MouseButton.NoButton,Qt.KeyboardModifier.NoModifier,Qt.ScrollPhase.ScrollBegin,False)
        QApplication.sendEvent(target,event);pause(250)
        end=QWheelEvent(QPointF(target_point),QPointF(global_point),QPoint(0,0),QPoint(0,0),Qt.MouseButton.NoButton,Qt.KeyboardModifier.NoModifier,Qt.ScrollPhase.ScrollEnd,False)
        QApplication.sendEvent(target,end);pause(100)
    def host_until(code,seconds=30):
        deadline=time.monotonic()+seconds
        while time.monotonic()<deadline:
            if js(code):return
            pause(100)
        raise AssertionError(code)
    def host_click(text):
        expression='[...document.querySelectorAll("button")].find(e=>e.offsetParent&&e.textContent.trim()==='+json.dumps(text)+')'
        host_until('!!('+expression+')&&!('+expression+').disabled');js(expression+'.click()');pause(180)
    def fonts(size):
        host_click('设置')
        for label in ['正文基础字号','图表基础字号']:
            selector=json.dumps('input[aria-label="'+label+'"]')
            assert js('(()=>{const e=[...document.querySelectorAll('+selector+')].find(e=>e.offsetParent);if(!e)return false;e.value='+str(size)+';e.dispatchEvent(new Event("input",{bubbles:true}));return true;})()')
        host_click('应用字体');until('d.documentElement.style.getPropertyValue("--body-size")==='+json.dumps(str(size)+'px'))
        host_until('[...document.querySelectorAll(".font-settings [role=status]")].some(e=>e.offsetParent&&e.textContent.includes("字体已应用"))')
        assert js('(()=>{const e=[...document.querySelectorAll('+json.dumps('button[aria-label="关闭 设置"]')+')].find(e=>e.offsetParent);if(!e)return false;e.click();return true;})()');pause(250)
    def pointer_click(selector):
        local('d.querySelector('+json.dumps(selector)+').scrollIntoView({block:"center",inline:"nearest"});return true;');pause(180)
        point=local('const e=d.querySelector('+json.dumps(selector)+'),r=e.getBoundingClientRect(),fr=f.getBoundingClientRect();return {x:fr.x+r.x+r.width/2,y:fr.y+r.y+r.height/2,disabled:e.disabled};')
        assert not point['disabled'],selector
        target=w.view.focusProxy() or w.view
        p=target.mapFromGlobal(w.view.mapToGlobal(QPoint(round(point['x']),round(point['y']))))
        QTest.mouseMove(target,p);QTest.mouseClick(target,Qt.MouseButton.LeftButton,pos=p);pause(150)
    def selector(symbol):return '.preset-consonants [data-ipa='+json.dumps(symbol,ensure_ascii=False)+']'
    catalog=json.loads((root/'frontend/src/modules/ipa-plus/data/catalog.json').read_text('utf8'))
    matrix=next(s for s in catalog['charts']['ipa'] if s['id']=='extended')
    entries={e['id']:e for e in catalog['entries']}
    expected_rows=[{'label':r['label'],'cells':[[entries[i]['insertText'] for i in c['ids']] for c in r['cells']]} for r in matrix['rows']]
    def complete():
        actual=local('const t=d.querySelector(".preset-consonants table");return {columns:[...t.tHead.rows[0].cells].slice(1).map(e=>e.textContent),rows:[...t.tBodies[0].rows].map(r=>({label:r.cells[0].textContent,cells:[...r.cells].slice(1).map(c=>[...c.querySelectorAll("button")].map(e=>e.dataset.ipa))})),buttons:[...d.querySelectorAll(".preset-consonants button")].length};')
        assert actual=={'columns':matrix['columns'],'rows':expected_rows,'buttons':186},actual
        assert local('return d.querySelectorAll(".preset-vowels [data-ipa]").length===28&&d.querySelectorAll(".preset-symbol-flow [data-ipa]:not([data-custom-ipa])").length===20;')
    try:
        settled();fonts(14)
        local('win.qaErrors=[];win.addEventListener("error",e=>win.qaErrors.push(e.message));')
        click('#customPreset');complete()
        flags=local('return [...d.querySelectorAll(".preset-consonants button")].filter(e=>["pʰ","m̥"].includes(e.dataset.ipa)).map(e=>({symbol:e.dataset.ipa,disabled:e.disabled}));');assert flags==[{'symbol':'pʰ','disabled':False},{'symbol':'m̥','disabled':True}],flags
        assert local('return d.querySelectorAll("#customPresetSymbols button").length===2&&![...d.querySelectorAll("#customPresetSymbols button")].some(e=>e.dataset.ipa==="pʰ");')
        click('#presetCancel')
        report['checks'].append('save/load use all 14 M17 columns, 13 rows, 186 exact consonant entries; 28 vowels and 20 extras retained; old [pʰ] resolves in table and custom ã/free name remain below')
        expected={}
        for mode,symbol,pitch,width in [('voiceless','pʰ',220,118),('voiced','m̥',205,104),('whisper','t͡s',180,112),('voiceless','pʼ',201,107),('voiced','ɓ',185,115),('whisper','ʘ',175,120)]:
            click('#tab-sound');input_value('#sourceMode',mode,'change');input_value('#pressureRange',930)
            input_value('#openingRange',.77);input_value('#chinkRange',2.2);input_value('#f0Range',pitch)
            click('#tab-organs');input_value('#lipWidthRange',width)
            expected[symbol]=controls();click('#savePreset');complete();pointer_click(selector(symbol))
            until('!d.querySelector("#presetDialog").open')
            stored=next(p for p in presets() if p['name']==symbol)
            assert stored['source']['mode']==mode and stored['source']['pressure_pa']==930 and stored['source']['opening_mm']==.77 and stored['source']['posterior_gap_mm2']==2.2 and stored['f0']==pitch
            assert stored['source']['vibration']==(1 if mode=='voiced' else 0)
        assert next(p['id'] for p in presets() if p['name']=='pʰ')=='legacy-aspirate'
        for symbol in expected:
            click('#customPreset');complete();pointer_click(selector(symbol));settled();assert controls()==expected[symbol],(symbol,controls(),expected[symbol])
        report['checks'].append('real Qt pointer saves and loads aspirated, voiceless nasal, affricate, ejective, implosive and click labels; native JSON and all restored organ/source/F0 controls match exactly for three source modes; old id retained')
        total=len(presets());identifier=next(p['id'] for p in presets() if p['name']=='t͡s')
        expected['t͡s']=controls();click('#savePreset');click('#presetInputButton');input_value('#presetInput','t͡s');click('#presetInputSave');until('!d.querySelector("#presetDialog").open')
        assert len(presets())==total and next(p['id'] for p in presets() if p['name']=='t͡s')==identifier
        custom='ãː';custom_expected=controls()
        click('#savePreset');click('#presetInputButton');input_value('#presetInput',custom)
        QTest.keyClick(w.view.focusProxy() or w.view,Qt.Key.Key_Return);until('!d.querySelector("#presetDialog").open')
        assert next(p for p in presets() if p['name']==custom)['source']['mode']==custom_expected['source']
        click('#customPreset');click('[data-custom-ipa][data-ipa='+json.dumps(custom,ensure_ascii=False)+']');settled();assert controls()==custom_expected
        before=(profile/'presets.json').read_bytes();click('#savePreset');click('#presetInputButton');click('#presetInputSave')
        assert local('return d.querySelector("#presetStatus").textContent==="请输入音标。"');input_value('#presetInput','a'*81);click('#presetInputSave')
        assert local('return d.querySelector("#presetStatus").textContent.includes("80")') and (profile/'presets.json').read_bytes()==before
        click('#presetCancel')
        report['checks'].append('manual input of expanded affricate updates same id/count; non-chart ãː uses real Enter and below-chart loading; invalid empty/81-codepoint inputs and cancel do not write')
        import stat
        (profile/'presets.json').chmod(stat.S_IREAD)
        try:
            click('#savePreset');click(selector('kʰ'));until('d.querySelector("#presetStatus").textContent.startsWith("保存失败")')
            assert local('return d.querySelector("#presetDialog").open') and (profile/'presets.json').read_bytes()==before
        finally:(profile/'presets.json').chmod(stat.S_IREAD|stat.S_IWRITE)
        click(selector('kʰ'));until('!d.querySelector("#presetDialog").open')
        local('d.body.dataset.qaReload="old";f.contentWindow.location.reload();return true;')
        until('d.body?.dataset.qaReload!=="old"&&d.body?.dataset.engineState==="ready"&&d.body?.dataset.posePending==="false"');pause(180)
        for symbol in expected:
            click('#customPreset');pointer_click(selector(symbol));settled();assert controls()==expected[symbol],(symbol,controls(),expected[symbol])
        click('#customPreset');click('[data-custom-ipa][data-ipa='+json.dumps(custom,ensure_ascii=False)+']');settled();assert controls()==custom_expected
        report['checks'].append('real disk failure keeps file and dialog, retry works; full expanded poses and custom source settings survive real iframe reopen')
        for theme in ['light','dark']:
            js('document.documentElement.dataset.theme='+json.dumps(theme));pause(180)
            for width,height,columns,size in [(1920,1080,'three',14),(1440,900,'three',14),(1200,800,'two',14),(1000,700,'two',14),(1440,900,'three',18),(1440,900,'three',24)]:
                w.resize(width,height);fonts(size);input_value('#columnLayout',columns,'change');pause(220)
                click('#savePreset');local('d.querySelector("#presetDialog").scrollTop=0;d.querySelector(".preset-consonants").scrollLeft=0;return true;');pause(180);complete()
                layout=local('const e=d.querySelector("#presetDialog"),r=e.getBoundingClientRect(),t=d.querySelector(".preset-consonants"),b=d.querySelector(".preset-consonants button");return {x:r.x,y:r.y,right:r.right,bottom:r.bottom,width:win.innerWidth,height:win.innerHeight,dialogWidth:r.width,scrollHeight:e.scrollHeight,clientHeight:e.clientHeight,tableScrollWidth:t.scrollWidth,tableClientWidth:t.clientWidth,symbolFont:win.getComputedStyle(b).fontSize,symbolFamily:win.getComputedStyle(b).fontFamily};')
                assert layout['x']>=0 and layout['y']>=0 and layout['right']<=layout['width']+1 and layout['bottom']<=layout['height']+1 and 'Doulos' in layout['symbolFamily'],layout
                assert 18<=float(layout['symbolFont'].replace('px',''))<=24
                bad=local('return [...d.querySelectorAll("#presetDialog *")].filter(e=>e.getClientRects().length&&[...e.childNodes].some(n=>n.nodeType===3&&n.textContent.trim())&&parseFloat(win.getComputedStyle(e).fontSize)<11.999).map(e=>e.textContent);');assert bad==[],bad
                overflow=local('return [...d.querySelectorAll(".preset-consonants button")].filter(e=>{const a=e.getBoundingClientRect(),b=e.closest("td").getBoundingClientRect();return a.left<b.left-.5||a.right>b.right+.5||a.top<b.top-.5||a.bottom>b.bottom+.5;}).map(e=>e.dataset.ipa);');assert overflow==[],overflow
                assert local('return [...d.querySelectorAll(".preset-consonants button")].every(e=>!e.disabled)')
                w.view.grab().save(str(out/f'{theme}-{width}-{size}px-save.png'))
                if layout['tableScrollWidth']>layout['tableClientWidth']+2:
                    local('const e=d.querySelector(".preset-consonants");e.scrollLeft=e.scrollWidth;return true;');pause(150)
                    assert local('const a=d.querySelector(".preset-consonants").getBoundingClientRect(),b=d.querySelector(".preset-consonants tr").lastElementChild.getBoundingClientRect();return b.right<=a.right+1&&b.left>=a.left;')
                    w.view.grab().save(str(out/f'{theme}-{width}-{size}px-last-column.png'))
                if layout['scrollHeight']>layout['clientHeight']+2:
                    wheel('#presetDialog',-720);assert local('return d.querySelector("#presetDialog").scrollTop')>0
                local('d.querySelector("#customPresetSection").scrollIntoView({block:"nearest"});return true;');pause(150)
                assert local('const r=d.querySelector("#customPresetSymbols").getBoundingClientRect();return r.top>=0&&r.bottom<=win.innerHeight;')
                w.view.grab().save(str(out/f'{theme}-{width}-{size}px-custom.png'))
                click('#presetCancel');report['layouts'].append({'theme':theme,'windowWidth':width,'windowHeight':height,'columns':columns,'bodySize':size,**layout})
        report['checks'].append('12 light/dark/window/font layouts: all 186 buttons fit their own cells with wrapping, every text >=12px; 18-24px consonants; native vertical wheel and last-column horizontal reach; custom area reachable')
        w.resize(1440,900);fonts(14)
        host_click('国际音标表Plus')
        host_until('document.querySelector(".m17-body")?.dataset.loaded==="true"&&document.querySelector(".m17-body")?.dataset.fontReady==="true"')
        assert js('!document.querySelector(".m17-related-chart")&&![...document.querySelectorAll(".ipa-plus-page button")].some(e=>e.textContent.includes("查看所选介绍"))')
        assert js('document.querySelectorAll(".m17-section-extended [data-symbol-id]").length')==186
        aspirate=next(e['id'] for e in catalog['entries'] if e['insertText']=='pʰ' and e['system']=='ipa')
        s='[data-symbol-id='+json.dumps(aspirate)+']'
        js('document.querySelector('+json.dumps(s)+').click()');host_until('document.querySelector(".m17-editor").value==="pʰ"')
        host_click('撤销');host_until('document.querySelector(".m17-editor").value===""');host_click('重做');host_until('document.querySelector(".m17-editor").value==="pʰ"')
        js('document.querySelector('+json.dumps(s)+').dispatchEvent(new MouseEvent("mouseenter",{bubbles:false}))');host_until('!!document.querySelector(".m17-hover-panel")')
        js('document.querySelector('+json.dumps(s)+').dispatchEvent(new MouseEvent("contextmenu",{bubbles:true,cancelable:true}))');host_until('!!document.querySelector(".m17-detail-panel")')
        host_click('收起');pause(150)
        assert js('(()=>{const e=document.querySelector('+json.dumps(s)+');e.scrollIntoView({block:"nearest"});e.focus();return document.activeElement===e;})()')
        QTest.keyClick(w.view.focusProxy() or w.view,Qt.Key.Key_Return,Qt.KeyboardModifier.AltModifier);host_until('!!document.querySelector(".m17-detail-panel")');host_click('收起')
        for theme in ['light','dark']:
            js('document.documentElement.dataset.theme='+json.dumps(theme));pause(180)
            for width,height in [(1920,1080),(1440,900),(1000,700)]:
                w.resize(width,height);pause(250)
                assert js('!document.querySelector(".m17-related-chart")&&document.querySelectorAll(".m17-section-extended [data-symbol-id]").length===186&&!document.querySelector(".m17-chart-meta button")')
                w.view.grab().save(str(out/f'm17-{theme}-{width}.png'))
        report['checks'].append('M17 six light/dark layouts: both requested elements removed; 186 matrix entries retained; aspirate insertion/undo/redo, hover, contextmenu and real Alt+Enter details work')
        report['success']=True
    except Exception as error:
        report['failure']=str(error);w.view.grab().save(str(out/'failure.png'));raise
    finally:
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf8')
        w.closing=True;w.close();w.page.deleteLater();app.processEvents()
        print(json.dumps({'out':str(out),'success':report['success'],'checks':report['checks']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
