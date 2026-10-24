"""Actual Windows Qt, native multi-value keys/save, synthetic Praat outputs only."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import subprocess
from contextlib import nullcontext
sys.path.insert(0,str(Path(__file__).resolve().parent))
for key in ('QT_ENABLE_HIGHDPI_SCALING','QT_SCALE_FACTOR','QT_SCREEN_SCALE_FACTORS'):os.environ.pop(key,None)
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS','--disable-gpu --mute-audio')
from verification_artifacts import verification_run
from verify_m08_variants import fixtures,audit,ROOT
from workbench_source import configure


def main():
    configure()
    from PyQt6.QtCore import QEventLoop,QTimer,Qt,QPoint
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication,QFileDialog
    from ptb_desktop.host import Workbench,register_scheme
    register_scheme();app=QApplication(['M08-variants-owned-QA'])
    with nullcontext((Path(sys.argv[-2]),Path(sys.argv[-1]))) as (out,scratch):
        print(out,flush=True);inputs,saved,db,cache=fixtures(scratch)
        original={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.iterdir()}
        w=Workbench(ROOT/'frontend/dist',test=True,jobs_path=db,local_files_root=cache,vocal_profile=scratch/'vocal',start_module='M08')
        w.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen,True);w.resize(w.screen().geometry().size());w.showMaximized();w.view.setZoomFactor(1)
        output_choices=[];cancel_output=[False]
        def choose(*a,**kw):
            if '结果' in str(a):output_choices.append(str(saved));return '' if cancel_output[0] else str(saved)
            return str(inputs)
        old_picker=QFileDialog.getExistingDirectory;QFileDialog.getExistingDirectory=choose
        report=dict(success=False,checks=[],scope='Windows source Qt, actual QWebChannel/local service/Praat; native dialog destination injected; no physical IME, Mac or final package')
        def pause(ms=80):
            loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
        def js(code):
            loop=QEventLoop();values=[];w.page.runJavaScript(code,lambda value:(values.append(value),loop.quit()));QTimer.singleShot(8000,loop.quit);loop.exec()
            assert values,'JS timeout';return values[0]
        def until(code,seconds=120):
            end=time.monotonic()+seconds
            while time.monotonic()<end:
                if js(code):return
                pause()
            raise AssertionError(code+' '+str(js('document.body.innerText.slice(-5000)')))
        def click(text):
            assert js('(()=>{const name='+json.dumps(text)+';const e=[...document.querySelectorAll("button")].find(e=>e.offsetParent&&(e.textContent.trim()===name||e.getAttribute("aria-label")===name));if(!e||e.disabled)return false;e.click();return true})()'),text
            pause()
        def fill(label,value):
            js('(()=>{const e=document.querySelector('+json.dumps('[aria-label="'+label+'"]')+');e.value='+json.dumps(value)+';e.dispatchEvent(new Event("input",{bubbles:true}));e.dispatchEvent(new Event("change",{bubbles:true}));})()');pause()
        def native(selector):
            p=js('(()=>{const e=document.querySelector('+json.dumps(selector)+');e.scrollIntoView({block:"center"});const r=e.getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2}})()');pause()
            QTest.mouseClick(w.view.focusProxy() or w.view,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,QPoint(round(p['x']),round(p['y'])));pause()
        def keys(label,value):
            selector='[aria-label="'+label+'"]';native(selector);target=w.view.focusProxy() or w.view
            QTest.keyClick(target,Qt.Key.Key_A,Qt.KeyboardModifier.ControlModifier);QTest.keyClicks(target,value,Qt.KeyboardModifier.NoModifier,30);pause(200)
            assert js('document.querySelector('+json.dumps(selector)+').value')==value
        def idle():until('document.querySelector(".m08-page")?.getAttribute("aria-busy")==="false"')
        def snapshot(name):
            w.view.repaint();pause(500);image=w.view.grab().toImage()
            if image.width()==2561 and image.height()==1601:image=image.copy(0,0,2560,1600)
            image.save(str(out/name));return dict(width=image.width(),height=image.height(),dpr=w.devicePixelRatioF())
        try:
            until('!!document.querySelector(".m08-page")');click('收起侧栏');click('打开音频目录');until('document.querySelector("[aria-label=变速变调音频]").options.length===3')
            source_ids=js('[...document.querySelector("[aria-label=变速变调音频]").options].filter(e=>e.value).map(e=>e.value)')
            fill('变速变调音频',source_ids[0]);until('!!document.querySelector("svg.m08-curve")');idle()
            fill('参考线 Hz','200');click('添加参考线');fill('参考线方向','time');keys('参考线 秒','0.5');click('添加参考线')
            assert js('document.querySelectorAll(".m08-curve .reference").length')==2
            assert js('(()=>{const e=document.querySelector(".time-reference"),s=e.ownerSVGElement;return Math.abs(e.getBBox().x-(60+(s.viewBox.baseVal.width-80)*.5))<.01})()')
            snapshot('time-reference.png');report['checks'].append('actual Qt horizontal Hz and vertical seconds references coexist at exact time coordinate')
            keys('语速倍率','0.8, 1.2');keys('音高倍率','1, 1.2');click('保存编辑草稿');click('合成整段')
            until('document.querySelector("dialog[open]")?.getAttribute("aria-label")==="批量保存" || document.querySelector("dialog[open]")?.innerText.includes("保存所选音频")');idle()
            assert js('document.querySelectorAll("dialog[open] .file-list input").length')==4
            click('取消');report['checks'].append('native QTest multi-value input, four actual edited-F0 syntheses, completion save prompt')
            click('批量变速变调');fill('音高调整方式','hz');keys('语速倍率','0.8, 1.2');keys('音高偏移 Hz','-20, 20');click('全选音频')
            assert js('!document.querySelector("[aria-label=音高倍率]")')
            js('document.documentElement.dataset.theme="light";document.documentElement.dataset.palette="absolutely"')
            # Capture the actual batch controls before the submission changes status.
            report['manual_screenshot']=snapshot('batch-settings.png')
            assert report['manual_screenshot']==dict(width=2560,height=1600,dpr=1.5),report['manual_screenshot']
            click('开始批量处理');until('document.querySelectorAll("dialog[open] .file-list input").length===8');idle()
            cancel_output[0]=True;click('保存所选音频');until('document.body.innerText.includes("已取消保存，生成结果保留")');assert not list(saved.iterdir());click('取消')
            cancel_output[0]=False;click('保存最近批次');click('保存所选音频');until('!document.querySelector("dialog[open]")');idle()
            assert len(list(saved.glob('*.wav')))==8 and len(output_choices)==2
            values=audit(scratch)
            for row in values:
                import re
                speed=float(re.search(r'speed-([\d.]+)',row['name'])[1]);shift=float(re.search(r'pitch-([+-][\d.]+)Hz',row['name'])[1]);base=220 if row['name'].startswith('原音 二') else 150
                assert abs(row['duration']-1/speed)<.001 and abs(row['f0']-base-shift)<1,row
            report['saved']=values;report['checks'].append('batch exclusive Hz/speed combinations; eight actual WAVs, one chosen directory, cancel/retry, duration and measured F0')
            click('保存最近批次');assert js('document.querySelector(".export-numbering input").checked') is False
            native('.export-numbering input');assert js('document.querySelector(".export-numbering input").checked') is True
            snapshot('numbering-dialog.png');click('保存所选音频');until('!document.querySelector("dialog[open]")');idle()
            names=js('[...document.querySelectorAll(".save-location li")].map(e=>e.textContent)')
            assert len(names)==8
            for i,name in enumerate(names):
                assert name.startswith(str(i+1)+'_'),name
                assert (saved/name).read_bytes()==(saved/name.split('_',1)[1]).read_bytes()
            click('保存最近批次');assert js('document.querySelector(".export-numbering input").checked') is False;click('取消')
            report['checks'].append('native Qt checkbox defaults off, opted-in eight file prefixes preserve WAV bytes, reopens unchecked')
            native('.task-row:last-child .task-result');until('!!document.querySelector("dialog[open]")');click('放弃并切换');idle()
            until('!!document.querySelector(".task-result[aria-pressed=true]")')
            assert js('document.querySelector("[aria-label=变速变调音频]").value')==source_ids[1]
            report['checks'].append('native pointer selects task from another source, draft guard and correct source/output selection')
            for theme in ('light','dark'):
                w.resize(1280,800);js('document.documentElement.dataset.theme='+json.dumps(theme));pause(300)
                assert js('(()=>{const e=document.querySelector(".m08-page");return e.scrollWidth<=e.clientWidth+2})()')
                snapshot('qt-'+theme+'.png')
            report['checks'].append('actual built Qt source in light/dark at 150 percent DPR, no horizontal overflow')
            assert original=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.iterdir()}
            report['success']=True
        except BaseException as exc:
            report['failure']=str(exc);snapshot('failed.png');raise
        finally:
            (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
            QFileDialog.getExistingDirectory=old_picker;w.closing=True;w.close();pause(400);app.quit()


if __name__=='__main__':
    if '--worker' in sys.argv:main()
    else:
        with verification_run('m08-variants','qt','Actual hidden Qt; new empty SQLite; two 1s PCM16 150/220 Hz tones; native keys; original hashes; output readback') as (out,scratch):
            print(out,flush=True)
            result=subprocess.run([sys.executable,'-B','-X','utf8',str(Path(__file__).resolve()),'--worker',str(out),str(scratch)],capture_output=True,text=True,encoding='utf8',errors='replace',creationflags=subprocess.CREATE_NO_WINDOW)
            (out/'qt.log').write_text(result.stdout+result.stderr,encoding='utf8')
            if result.returncode:raise RuntimeError(result.stdout+result.stderr)
