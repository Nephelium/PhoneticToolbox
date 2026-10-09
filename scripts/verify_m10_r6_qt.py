"""M10-R6/R7: real native engine/profile, hidden owned Qt, no audio devices."""
import json
import os
import sqlite3
import time
from pathlib import Path
from uuid import uuid4


def main(custom_ipa=False):
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
    out=root/('output/validation/m10-r7' if custom_ipa else 'output/validation/m10-r6')/('qt-'+uuid4().hex);out.mkdir(parents=True)
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
        native.invoke('presets/save',{'presets':[legacy,slash]})
        native.profile.save_frames([{**legacy,'name':f'姿势 {i+1}'} for i in range(24)])
    finally:native.close()
    register_scheme();app=QApplication(['M10-R7-owned-QA' if custom_ipa else 'M10-R6-owned-QA'])
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
    def settled():until('d.body.dataset.engineState==="ready"&&d.body.dataset.posePending==="false"')
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
    try:
        settled();assert local('return !d.querySelector("header.masthead,footer,#engineStatus,#statusText")&&d.querySelector("#customPreset").tagName==="BUTTON"')
        local('win.qaErrors=[];win.addEventListener("error",e=>win.qaErrors.push(e.message));')
        click('#customPreset');assert local('return d.querySelector("#presetDialog").open&&d.querySelectorAll("[data-ipa]:not([data-custom-ipa])").length===107')
        assert local('return d.querySelector("[data-ipa=i]").disabled===false&&d.querySelector("[data-ipa=p]").disabled===true')
        click('#customPresetSymbols [data-custom-ipa]');settled()
        assert controls()['source']=='voiced';report['checks'].append('107 IPA symbols; missing entries disabled; legacy free name and /i/ available')
        expected={}
        for mode,symbol,pitch,width in [('voiced','a',190,115),('voiceless','p',220,120),('whisper','ɧ',180,105)]:
            click('#tab-sound');input_value('#sourceMode',mode,'change');input_value('#pressureRange',930)
            input_value('#openingRange',.77);input_value('#chinkRange',2.2);input_value('#f0Range',pitch)
            click('#tab-organs');input_value('#lipWidthRange',width)
            expected[symbol]=controls();click('#savePreset')
            assert local('return d.querySelector("#presetDialog").open&&!d.querySelector("#presetName")')
            click('[data-ipa="'+symbol+'"]');until('!d.querySelector("#presetDialog").open')
            stored=next(p for p in presets() if p['name']==symbol)
            assert stored['source']['mode']==mode and stored['source']['pressure_pa']==930 and stored['source']['opening_mm']==.77 and stored['source']['posterior_gap_mm2']==2.2 and stored['f0']==pitch
            assert stored['source']['vibration']==(1 if mode=='voiced' else 0)
        for symbol in ['a','p','ɧ']:
            click('#customPreset');click('[data-ipa="'+symbol+'"]');settled();assert controls()==expected[symbol],(symbol,controls(),expected[symbol])
        report['checks'].append('native JSON save and exact UI restoration for voiced, voiceless and whisper: full source, F0, width and organ params')
        count=len(presets());old_id=next(p['id'] for p in presets() if p['name']=='p')
        click('#tab-sound');input_value('#sourceMode','voiced','change');input_value('#f0Range',235)
        click('#savePreset');click('[data-ipa=p]');until('!d.querySelector("#presetDialog").open')
        assert len(presets())==count and next(p for p in presets() if p['name']=='p')['id']==old_id
        assert next(p for p in presets() if p['name']=='p')['source']['mode']=='voiced'
        report['checks'].append('same IPA replaces one pose using its existing id; unrelated and legacy records preserved')
        before=(profile/'presets.json').read_bytes();click('#savePreset');click('#presetCancel');assert (profile/'presets.json').read_bytes()==before
        # Cause an actual isolated disk error by keeping the target read-only.
        import stat
        (profile/'presets.json').chmod(stat.S_IREAD)
        try:
            click('#savePreset');click('[data-ipa=b]');until('d.querySelector("#presetStatus").textContent.startsWith("保存失败")')
            assert local('return d.querySelector("#presetDialog").open&&!d.querySelector("[data-ipa=b]").disabled')
            assert (profile/'presets.json').read_bytes()==before
        finally:(profile/'presets.json').chmod(stat.S_IREAD|stat.S_IWRITE)
        click('[data-ipa=b]');until('!d.querySelector("#presetDialog").open');assert len(presets())==count+1
        report['checks'].append('cancel leaves profile untouched; actual disk failure preserves old file and dialog, retry succeeds')
        custom_expected={}
        if custom_ipa:
            for symbol,mode in [('t͡s','voiceless'),('ã','voiced'),('n̥','whisper')]:
                click('#tab-sound');input_value('#sourceMode',mode,'change');input_value('#pressureRange',1110)
                input_value('#openingRange',.45);input_value('#chinkRange',3.3);input_value('#f0Range',205)
                custom_expected[symbol]=controls();click('#savePreset');click('#presetInputButton')
                assert local('return !d.querySelector("#presetInputRow").hidden&&d.activeElement.id==="presetInput"')
                input_value('#presetInput',symbol)
                if symbol=='ã':QTest.keyClick(w.view.focusProxy() or w.view,Qt.Key.Key_Return)
                else:click('#presetInputSave')
                until('!d.querySelector("#presetDialog").open')
                stored=next(p for p in presets() if p['name']==symbol)
                assert stored['source']['mode']==mode and stored['source']['pressure_pa']==1110 and stored['f0']==205
                click('#customPreset');assert local('return d.querySelector("#presetInputButton").hidden&&!!d.querySelector('+json.dumps('[data-custom-ipa][data-ipa="'+symbol+'"]')+')')
                click('[data-custom-ipa][data-ipa="'+symbol+'"]');settled();assert controls()==custom_expected[symbol]
            report['checks'].append('custom t͡s, ã and n̥: exact Unicode, button/real Enter submission, below-chart buttons, three source modes and F0 restored')
            total=len(presets());identifier=next(p['id'] for p in presets() if p['name']=='t͡s')
            custom_expected['t͡s']=controls();click('#savePreset');click('#presetInputButton');input_value('#presetInput','t͡s');click('#presetInputSave');until('!d.querySelector("#presetDialog").open')
            assert len(presets())==total and next(p['id'] for p in presets() if p['name']=='t͡s')==identifier
            before=(profile/'presets.json').read_bytes();click('#savePreset');click('#presetInputButton');click('#presetInputSave')
            assert local('return d.querySelector("#presetStatus").textContent==="请输入音标。"&&d.querySelector("#presetDialog").open')
            input_value('#presetInput','a'*81);click('#presetInputSave')
            assert local('return d.querySelector("#presetStatus").textContent.includes("80")')
            assert (profile/'presets.json').read_bytes()==before;click('#presetInputCancel');assert local('return d.querySelector("#presetInputRow").hidden');click('#presetCancel')
            report['checks'].append('custom name update retains id/count; empty and long input rejected without writes; input collapse/cancel')
        # Reopen the iframe through the real host/profile, without new user data.
        assert local('return win.qaErrors.length')==0
        local('d.body.dataset.qaReload="old";f.contentWindow.location.reload();return true;')
        until('d.body?.dataset.qaReload!=="old"&&d.body?.dataset.engineState==="ready"&&d.body?.dataset.posePending==="false"')
        local('win.qaErrors=[];win.addEventListener("error",e=>win.qaErrors.push(e.message));')
        click('#customPreset');click('[data-ipa="ɧ"]');settled();assert controls()==expected['ɧ']
        if custom_ipa:
            for symbol in ['t͡s','ã','n̥']:
                click('#customPreset');click('[data-custom-ipa][data-ipa="'+symbol+'"]');settled();assert controls()==custom_expected[symbol]
            report['checks'].append('all custom IPA buttons and complete pose/source settings survive iframe reopen')
        report['checks'].append('iframe reopens and restores the native persisted whisper pose')
        for theme in ['light','dark']:
            js('document.documentElement.dataset.theme='+json.dumps(theme));pause(200)
            for width,height,columns in [(1440,900,'three'),(1200,800,'two'),(1000,700,'two')]:
                w.resize(width,height);input_value('#columnLayout',columns,'change');pause(300)
                for tab in ['organs','sound','motion']:
                    click('#tab-'+tab)
                    metric=local('const p=d.querySelector("#panel-'+tab+'"),r=p.getBoundingClientRect();return {scroll:p.scrollHeight,client:p.clientHeight,top:r.top,bottom:r.bottom,height:win.innerHeight,overflow:win.getComputedStyle(p).overflowY};')
                    assert metric['overflow']=='auto' and metric['bottom']<=metric['height']+1,metric
                    model_before=local('return d.querySelector("#viewport").getBoundingClientRect().height')
                    local('d.querySelector("#panel-'+tab+'").scrollTop=0;return true;')
                    if metric['scroll']>metric['client']+2:
                        wheel('#panel-'+tab,-480);assert local('return d.querySelector("#panel-'+tab+'").scrollTop')>0,(tab,metric)
                        wheel('#panel-'+tab,480);assert local('return d.querySelector("#panel-'+tab+'").scrollTop')==0
                    assert local('return d.querySelector("#viewport").getBoundingClientRect().height')==model_before
                report['layouts'].append({'theme':theme,'width':width,'height':height,'columns':columns})
                click('#tab-organs');click('#savePreset');pause(200)
                if custom_ipa:
                    assert local('return d.querySelectorAll("#customPresetSymbols [data-custom-ipa]").length===4')
                    click('#presetInputButton');input_value('#presetInput','kʷ');pause(250)
                    assert local('return d.querySelector("#presetInput").value==="kʷ"')
                dialog=local('const e=d.querySelector("#presetDialog"),r=e.getBoundingClientRect();return {x:r.x,y:r.y,right:r.right,bottom:r.bottom,width:win.innerWidth,height:win.innerHeight,font:win.getComputedStyle(e.querySelector("[data-ipa]")).fontFamily};')
                assert dialog['x']>=0 and dialog['y']>=0 and dialog['right']<=dialog['width']+1 and dialog['bottom']<=dialog['height']+1 and 'Doulos' in dialog['font'],dialog
                w.view.grab().save(str(out/f'{theme}-{width}-{columns}-save.png'))
                if custom_ipa:
                    local('d.querySelector("#presetDialog").scrollTop=0;return true;')
                    if local('const e=d.querySelector("#presetDialog");return e.scrollHeight>e.clientHeight+2;'):
                        wheel('#presetDialog',-720);assert local('return d.querySelector("#presetDialog").scrollTop')>0
                    local('d.querySelector("#customPresetSection").scrollIntoView({block:"nearest"});return true;');pause(200)
                    assert local('const r=d.querySelector("#customPresetSymbols").getBoundingClientRect();return r.top>=0&&r.bottom<=win.innerHeight;')
                    w.view.grab().save(str(out/f'{theme}-{width}-{columns}-custom.png'))
                click('#presetCancel');w.view.grab().save(str(out/f'{theme}-{width}-{columns}-workspace.png'))
        report['checks'].append('six light/dark and two/three-column layouts; actual Qt wheel down/up in overflowing panels; model remains fixed; compact dialog and Doulos font')
        click('#aboutButton');assert local('return d.querySelector("#aboutDialog").open&&!!d.querySelector("#sharedReferences")');click('#closeAbout')
        assert local('return win.qaErrors?.length||0')==0
        report['checks'].append('compact model/source entry and moved layout selector; no JavaScript errors')
        report['success']=True
    except Exception as error:
        report['failure']=str(error);w.view.grab().save(str(out/'failure.png'));raise
    finally:
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf8')
        w.closing=True;w.close();w.page.deleteLater();app.processEvents()
        print(json.dumps({'out':str(out),'success':report['success'],'checks':report['checks']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--custom-ipa',action='store_true')
    main(custom_ipa=parser.parse_args().custom_ipa)
