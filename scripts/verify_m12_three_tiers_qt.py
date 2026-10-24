"""Actual isolated Qt host/native TextGrid save and compact keyboard checks."""
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import struct
import subprocess
import sys
import time
import wave

os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS','--disable-gpu --mute-audio')
from PyQt6.QtCore import QEventLoop, QPoint, Qt, QTimer
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QFileDialog
from PyQt6 import sip
from verification_artifacts import verification_run
from workbench_source import configure

ROOT=Path(__file__).resolve().parents[1]
def main():
    configure()
    from ptb_desktop.host import Workbench,register_scheme
    from ptb_worker.local_acoustic_files import initialize_local_files
    from phonetic_core.annotation import parse_document
    register_scheme();app=QApplication(['M12 three tiers QA'])
    with verification_run('m12','three-tiers-qt','Two synthetic 3s PCM16 16kHz sine WAVs; original three interval/one point tier; new empty grid; actual Qt native saves and keyboard; no physical devices') as (out,scratch):
        print(out,flush=True)
        inputs=scratch/'inputs';inputs.mkdir()
        for name in ('A','B'):
            with wave.open(str(inputs/(name+'.wav')),'wb') as stream:
                stream.setparams((2 if name=='A' else 1,2,16000,0,'NONE','not compressed'))
                stream.writeframes(b''.join(struct.pack('<hh',0,round(5000*math.sin(i*2*math.pi*200/16000)) if 9600<=i<14400 else 0) if name=='A' else struct.pack('<h',0) for i in range(48000)))
        text='"ooTextFile short" "TextGrid" 0 3 <exists> 4 '+' '.join(f'"IntervalTier" "{n}" 0 3 3 0 0.5 "" 0.5 1 "{n}" 1 3 ""' for n in ('words','phones','tones'))+' "TextTier" "events" 0 3 1 0.2 "保留"'
        (inputs/'A.TextGrid').write_text(text,encoding='utf-8')
        originals={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.iterdir()}
        cache=scratch/'cache';cache.mkdir();initialize_local_files(cache);database=scratch/'jobs.sqlite3'
        with sqlite3.connect(database) as connection:
            for migration in ('002_jobs_sqlite.sql','005_acoustic_batches_sqlite.sql'):connection.executescript((ROOT/'backend/migrations'/migration).read_text('utf-8'))
        window=Workbench(ROOT/'frontend/dist',test=True,jobs_path=database,local_files_root=cache,vocal_profile=scratch/'vocal',start_module='M12');window.resize(1600,1100);window.show()
        picker=QFileDialog.getExistingDirectory;QFileDialog.getExistingDirectory=lambda *a,**k:str(inputs)
        report={'success':False,'checks':[]}
        def pause(ms=80):
            loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
        def js(code):
            loop=QEventLoop();values=[];window.page.runJavaScript(code,lambda value:(values.append(value),loop.quit()));QTimer.singleShot(5000,loop.quit);loop.exec()
            if not values:raise AssertionError('JavaScript timeout: '+code)
            return values[0]
        def until(code):
            deadline=time.monotonic()+30
            while time.monotonic()<deadline:
                if js(code):return
                pause()
            raise AssertionError(code+'\n'+str(js('document.body.innerText.slice(-4000)')))
        def click(name):
            expression='[...document.querySelectorAll("button")].find(b=>b.textContent.trim().replace(/\\s*\\*$/,"")==='+json.dumps(name)+')'
            until('!!('+expression+')&&!('+expression+').disabled');js(expression+'.click()');pause()
        def native(selector):
            js('document.querySelector('+json.dumps(selector)+').scrollIntoView({block:"center"})');pause()
            coords=js('(()=>{const r=document.querySelector('+json.dumps(selector)+').getBoundingClientRect();return [r.x+r.width/2,r.y+r.height/2]})()')
            QTest.mouseClick(window.view.focusProxy() or window.view,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,QPoint(round(coords[0]),round(coords[1])));pause()
        def choose(name):
            js('[...document.querySelectorAll(".annotation-file-list button")].find(b=>b.textContent.trim()==='+json.dumps(name)+').click()');until('document.querySelector(".annotation-page").getAttribute("aria-busy")==="false"');pause(200)
        def select(time,row):
            js('document.querySelector(".annotation-grid").scrollIntoView({block:"center"})');pause()
            coords=js('(()=>{const r=document.querySelector(".annotation-grid").getBoundingClientRect();return [r.x+r.width*'+str(time/3)+',r.y+r.height*'+str((row+.5)/3)+']})()')
            QTest.mouseClick(window.view.focusProxy() or window.view,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,QPoint(round(coords[0]),round(coords[1])));pause()
        input_selector='[aria-label="编辑选中标注文本"]'
        input_js='document.querySelector('+json.dumps(input_selector)+')'
        editor_js='document.querySelector(".annotation-grid").__vueParentComponent.props.editor'
        try:
            until('!!document.querySelector(".annotation-page")');click('打开音频目录');until('document.querySelectorAll(".annotation-file-list button").length===2');choose('A.wav')
            assert js('document.querySelector(".annotation-grid").dataset.tierCount')=='3'
            assert js('document.querySelector(".tier-link-controls")===null&&document.querySelector("[aria-label=强度起点]")===null')
            click('吸附与关联');native('.tier-link-controls input:nth-child(1)') # snap is an independent control
            js('[...document.querySelectorAll(".tier-link-controls label")].find(l=>l.textContent.includes("第一层 ↔ 第二层")).querySelector("input").click()')
            click('完成');assert js('document.querySelector(".tier-link-controls")===null')
            select(.75,0);click('强度贴合');assert js('document.querySelector("[aria-label=强度起点]").value')=='0.5';assert js('document.querySelector("[aria-label=强度终点]").value')=='1';assert js('document.querySelector("[aria-label=强度贴合声道]").value')=='';assert js('[...document.querySelectorAll("dialog button")].find(b=>b.textContent.trim()==="执行贴合").disabled');js('(()=>{const e=document.querySelector("[aria-label=强度贴合声道]");e.value="1";e.dispatchEvent(new Event("change",{bubbles:true}));})()');click('执行贴合');assert js('document.querySelector("dialog").textContent.includes("第 2 声道")');click('关闭');assert js('document.querySelector("[aria-label=强度起点]")===null');report['checks'].append('Qt top toolbar opens isolated snap/link and intensity modals; closed fields absent and current selection retained')
            select(.75,2);native(input_selector)
            target=window.view.focusProxy() or window.view
            QTest.keyClick(target,Qt.Key.Key_A,Qt.KeyboardModifier.ControlModifier);QTest.keyClicks(target,'ab');QTest.keyClick(target,Qt.Key.Key_Return);pause()
            click('音标键盘');until('!!document.querySelector(".ipa-keyboard .ipa-key")')
            native(input_selector);QTest.keyClick(target,Qt.Key.Key_Home);pause();QTest.keyClick(target,Qt.Key.Key_Right,Qt.KeyboardModifier.ShiftModifier);pause()
            report['selection_before_key']=js('({text:'+input_js+'.value,start:'+input_js+'.selectionStart,end:'+input_js+'.selectionEnd,focused:document.activeElement==='+input_js+'})')
            js('window.__keyEvents=[];for(const kind of ["pointerdown","mousedown","mouseup","click","blur","focus","change"]){document.addEventListener(kind,e=>{const i='+input_js+';window.__keyEvents.push({kind,target:e.target.className,start:i.selectionStart,end:i.selectionEnd,focus:document.activeElement===i});},true);}')
            # Exact canonical symbol selected by its displayed text, then actual Qt click.
            js('[...document.querySelectorAll(".ipa-keyboard .ipa-key")].find(b=>b.textContent==="æ").dataset.qtTarget="true"')
            native('.ipa-keyboard [data-qt-target]');report['key_events']=js('window.__keyEvents');assert js(input_js+'.value')=='æb',str(report['selection_before_key'])+' -> '+str(js(input_js+'.value'))+' / '+str(report['key_events'])
            QTest.keyClick(target,Qt.Key.Key_Z,Qt.KeyboardModifier.ControlModifier);pause();assert js(input_js+'.value')=='ab'
            native('.ipa-keyboard [data-qt-target]');assert js(input_js+'.value')=='æb'
            report['checks'].append('actual Qt text selection plus native IPA key inserts into third tier; independent word/phone texts unchanged')
            for theme in ('light','dark'):
                js('document.documentElement.dataset.theme='+json.dumps(theme));pause(200)
                for system in ('IPA','extIPA','VoQS'):
                    js('[...document.querySelectorAll(".ipa-keyboard nav button")].find(e=>e.textContent==='+json.dumps(system)+').click()');pause()
                    current_zoom=100
                    for zoom in (60,100,180,100):
                        label='放大音标键盘字号' if zoom>current_zoom else '缩小音标键盘字号'
                        for _ in range(abs(zoom-current_zoom)//10):js('document.querySelector('+json.dumps('[aria-label="'+label+'"]')+').click()')
                        pause(100);current_zoom=zoom
                        metrics=js('(()=>{const p=document.querySelector(".ipa-keyboard"),c=p.querySelector(".compact-sections"),font=e=>parseFloat(getComputedStyle(e).fontSize);return {key:font(c.querySelector(".ipa-key")),labels:[...c.querySelectorAll("h3,th,.vowel-heading")].filter(e=>e.textContent.trim()).map(font),width:p.getBoundingClientRect().width,view:innerWidth};})()')
                        assert abs(metrics['key']-19*zoom/100)<.02,metrics
                        assert all(abs(value-metrics['key'])<.02 for value in metrics['labels']),metrics
                        assert abs(metrics['width']-min(760*zoom/100,metrics['view']-16))<1,metrics
                js('[...document.querySelectorAll(".ipa-keyboard nav button")].find(e=>e.textContent==="IPA").click()');pause()
                window.view.grab().save(str(out/(theme+'.png')))
            report['checks'].append('actual Qt IPA/extIPA/VoQS light/dark at 60/100/180 percent: text and symbol fonts match, panel width follows zoom and viewport clamp')
            js('document.querySelector("[aria-label=关闭音标键盘]").click()');click('保存 TextGrid');until('document.querySelector(".annotation-page").getAttribute("aria-busy")==="false"');pause(200)
            saved=inputs/'A_自动保存.TextGrid';doc=parse_document(saved.read_text('utf-8'));assert doc['tiers'][2]['intervals'][1]['text']=='æb';assert doc['tiers'][3]['points'][0]['mark']=='保留'
            assert doc['tiers'][0]['intervals'][1]['xmin']>.55;assert doc['tiers'][0]['intervals'][1]['xmax']<.95;assert doc['tiers'][0]['intervals'][1]['text']=='words';assert doc['tiers'][1]['intervals'][1]['text']=='phones'
            choose('B.wav');click('创建标注层')
            for label,value in [('新建音节层名','syllables'),('新建音素层名','phonemes'),('新建第三层名','tone')]:
                js('(()=>{const e=document.querySelector('+json.dumps('[aria-label="'+label+'"]')+');e.value='+json.dumps(value)+';e.dispatchEvent(new Event("input",{bubbles:true}));})()')
            click('创建层');assert js('document.querySelector(".annotation-grid").dataset.tierCount')=='3';click('保存 TextGrid');pause(200);assert len(parse_document((inputs/'B_自动保存.TextGrid').read_text('utf-8'))['tiers'])==3
            report['checks'].append('three custom names create three actual native saved interval tiers')
            choose('A.wav');select(.75,2);assert js(input_js+'.value')=='æb';report['checks'].append('native TextGrid save/reopen retains IPA, third role and unrelated point tier')
            click('帮助');until('document.body.innerText.includes("用浮动国际音标键盘输入")');assert js('document.body.innerText.includes("选择三层与层间关联")');report['checks'].append('production in-app help contains both new manual sections')
            assert all(hashlib.sha256((inputs/name).read_bytes()).hexdigest()==digest for name,digest in originals.items());report['checks'].append('original WAV and TextGrid hashes unchanged');report['success']=True
        except Exception as exc:
            report['error']=repr(exc);report['body']=js('document.body.innerText.slice(-6000)');window.view.grab().save(str(out/'failure.png'));raise
        finally:
            (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');QFileDialog.getExistingDirectory=picker;window.closing=True;window.close();app.processEvents();sip.delete(window);app.processEvents();sip.delete(app);time.sleep(.2)
        print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':
    if '--child' in sys.argv:main()
    else:
        result=subprocess.run([sys.executable,'-B','-X','utf8',__file__,'--child'],capture_output=True,text=True,encoding='utf-8',errors='replace')
        print(result.stdout,flush=True);print(result.stderr,flush=True)
        # WebEngine helper processes finish when the Qt child interpreter exits.
        # Retry the same validated finalizer after that exit, including failures.
        prefix=str(ROOT/'output/validation/m12/three-tiers-qt-')
        paths=[Path(line.strip()) for line in result.stdout.splitlines() if line.strip().startswith(prefix)]
        if paths:
            clean=subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',str(ROOT/'scripts/cleanup_test_scratch.ps1'),'-RunDirectory',str(paths[0])],capture_output=True,text=True,encoding='utf-8',errors='replace')
            print(clean.stdout+clean.stderr,flush=True)
            (paths[0]/'console.log').write_text(result.stdout+result.stderr+clean.stdout+clean.stderr,encoding='utf-8')
            if clean.returncode:raise RuntimeError('Qt scratch retained; see cleanup.json')
        raise SystemExit(result.returncode)
