"""M11-R1 actual hidden Qt, grants, coexisting transcripts, real MFA and save."""
import hashlib
import argparse
import json
import os
from pathlib import Path
import shutil
import sqlite3
import time
from uuid import uuid4
os.environ.setdefault('QT_QPA_PLATFORM','windows')
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS','--disable-gpu --disable-gpu-compositing --mute-audio')
from PyQt6.QtCore import QEventLoop,QTimer,Qt,QPoint
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication,QFileDialog
from ptb_desktop.host import Workbench,register_scheme
from ptb_worker.local_acoustic_files import initialize_local_files
from ptb_worker.mfa.probe import generate

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--close-only',action='store_true');args=parser.parse_args()
    out=ROOT/'output/validation/m11-r1'/('qt-'+uuid4().hex);out.mkdir(parents=True);print(out,flush=True)
    components=out/'components';components.mkdir()
    shutil.copyfile(ROOT/'output/validation/m11/qt-642b638d008c497abb576e5db63dcbe7/components/registry.json',components/'registry.json')
    os.environ['PTB_M11_COMPONENT_ROOT']=str(components)
    inputs=out/'public';generate(inputs,word='啊')
    # Both source files exist. Old alignment times are deliberately different.
    (inputs/'probe.TextGrid').write_text('''File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 2.8
tiers? <exists>
size = 2
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        xmin = 0
        xmax = 2.8
        intervals: size = 1
        intervals [1]:
            xmin = 0
            xmax = 2.8
            text = "啊 啊 啊 啊"
    item [2]:
        class = "IntervalTier"
        name = "phones"
        xmin = 0
        xmax = 2.8
        intervals: size = 1
        intervals [1]:
            xmin = 0
            xmax = 2.8
            text = "PHONES_MUST_NOT_BE_READ"
''',encoding='utf8')
    dictionary=out/'公开测试.dict';dictionary.write_text('啊\ta˥˥\n',encoding='utf8')
    saved=out/'saved';saved.mkdir();db=out/'jobs.sqlite3'
    # Read-only snapshot of existing schema; only this copy receives test jobs.
    with sqlite3.connect((ROOT/'output/validation/p06/local-state.sqlite3').as_uri()+'?mode=ro',uri=True) as source,sqlite3.connect(db) as target:
        source.backup(target)
    with sqlite3.connect(db) as copy:
        assert not copy.execute("SELECT 1 FROM jobs WHERE state IN ('queued','running','cancel_requested')").fetchone()
    cache=out/'cache';cache.mkdir();initialize_local_files(cache)
    originals={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.iterdir()}
    register_scheme();app=QApplication(['M11-R1-owned-QA'])
    w=Workbench(ROOT/'frontend/dist',test=True,jobs_path=db,local_files_root=cache,vocal_profile=out/'vocal-profile')
    w.resize(1920,1080);w.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen);w.show();w.page.setAudioMuted(True)
    QFileDialog.getExistingDirectory=lambda *a,**k:str(inputs if 'corpus' in str(a) else saved)
    QFileDialog.getOpenFileName=lambda *a,**k:(str(dictionary),'')
    report=dict(success=False,checks=[],layouts=[],scope='actual Windows Qt/QWebChannel hidden native window; public generated audio; no build/install/real-user database DDL')
    def pause(ms=100):
        loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
    def js(code):
        loop=QEventLoop();box=[];w.page.runJavaScript(code,lambda value:(box.append(value),loop.quit()));QTimer.singleShot(6000,loop.quit);loop.exec()
        if not box:raise RuntimeError('JS timeout')
        return box[0]
    def until(code,seconds=120):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            if js(code):return
            pause()
        raise AssertionError(code+' '+str(js('document.body.innerText.slice(-5000)')))
    def click(text):
        assert js('(()=>{const e=[...document.querySelectorAll("button")].find(b=>b.offsetParent&&b.textContent.trim()==='+json.dumps(text)+');if(!e||e.disabled)return false;e.click();return true})()'),text
        pause()
    def select(label,value):
        js('(()=>{const e=[...document.querySelectorAll(".mfa-page label")].find(e=>e.childNodes[0].textContent.trim()==='+json.dumps(label)+').querySelector("select");e.value='+json.dumps(value)+';e.dispatchEvent(new Event("change",{bubbles:true}));})()')
        pause()
    def idle():until('document.querySelector(".mfa-page").getAttribute("aria-busy")==="false"')
    def can_start():return js('[...document.querySelectorAll(".mfa-page button")].find(b=>b.textContent.trim()==="开始对齐").disabled===false')
    try:
        until('!!document.querySelector("nav")');click('MFA 自动标注');until('!!document.querySelector(".mfa-page")');idle()
        if args.close_only:
            report['checks'].append('actual Qt hidden window startup and owned shutdown only')
            report['success']=True
            return
        until('document.querySelector(".mfa-form select").options.length>1')
        assert js('[...document.querySelectorAll(".parameter-row input")].map(e=>e.value)')==['100','400']
        assert js('document.querySelectorAll(".mfa-form select").length')==3
        # Native focus movement is tested without opening a native dropdown
        # popup in a hidden window. File-chooser calls are captured below.
        js('window.nativeKeys=[];document.addEventListener("keydown",e=>window.nativeKeys.push({key:e.key,trusted:e.isTrusted}),true);document.querySelectorAll(".mfa-form select")[2].focus()')
        QTest.keyClick(w.view.focusProxy() or w.view,Qt.Key.Key_Tab);pause(100)
        js('(()=>{const e=document.querySelectorAll(".mfa-page input[type=file]")[1];window.originalInputClick=e.click.bind(e);e.click=()=>window.dictionaryPickerRequested=true;})()')
        select('发音词典','import');assert js('window.dictionaryPickerRequested')
        # Actual file payload follows the same native frontend input handler.
        raw=dictionary.read_text('utf8')
        js('(()=>{const f=new File(['+json.dumps(raw)+'],"公开测试.dict"),d=new DataTransfer();d.items.add(f);const e=document.querySelectorAll(".mfa-page input[type=file]")[1];e.files=d.files;e.dispatchEvent(new Event("change",{bubbles:true}));})()');idle()
        assert js('document.querySelectorAll(".mfa-form select")[2].selectedOptions[0].textContent.includes("公开测试.dict")')
        click('选择语料目录');idle();assert js('document.body.innerText.includes("选择转写来源")');assert not can_start()
        select('转写来源','.lab');click('选择语料目录');idle();assert js('document.querySelector(".corpus-list").textContent.includes(".lab")');assert can_start()
        select('转写来源','.TextGrid');assert not can_start()
        click('选择语料目录');idle();assert js('document.querySelector(".corpus-list").textContent.includes(".TextGrid")');assert can_start()
        report['checks'].append('100/400 new defaults; dictionary dropdown import and selection; same-folder LAB/TextGrid coexistence; changed source disables stale submission')
        click('开始对齐');idle();until('document.querySelectorAll(".result-row").length===2',180)
        click('保存完整结果到输出目录');idle();until('document.body.innerText.includes("已保存 2 个文件")')
        provenance=json.loads(next(saved.rglob('m11-provenance.json')).read_text('utf8'))
        assert provenance['inputs'][0]['transcript_format']=='.TextGrid'
        assert provenance['execution']==dict(device='cpu',num_jobs=1,use_threading=True,use_mp=False,kernel_cache=True)
        assert provenance['transcript_adaptations'][0]['mode']=='words-to-whole-recording-transcript'
        assert all('PHONES_MUST_NOT_BE_READ' not in str(t) for t in provenance['textgrids'])
        report['textgrid_job']=provenance
        report['checks'].append('real TextGrid-only words extraction, ignored phones and old boundaries; actual persistent worker and hashed two-file save/provenance')
        # Restore registered dictionary and verify selection survives cancelled file dialog.
        model_id=json.loads((components/'registry.json').read_text('utf8'))['models'][0]['id']
        select('发音词典',model_id);assert js('document.querySelectorAll(".mfa-form select")[2].value')==model_id
        select('发音词典','import');assert js('document.querySelectorAll(".mfa-form select")[2].value')==model_id
        select('发音词典',js('[...document.querySelectorAll(".mfa-form select")[2].options].find(o=>o.textContent.includes("公开测试.dict")).value'))
        select('转写来源','.lab');click('选择语料目录');idle();click('开始对齐');idle()
        until('document.querySelectorAll(".result-row").length===2&&document.querySelector(".job-state").textContent.includes("完整结果")',180)
        click('保存完整结果到输出目录');idle()
        results=[json.loads(p.read_text('utf8')) for p in saved.rglob('m11-provenance.json')]
        assert len(results)==2
        lab=next(p for p in results if p['inputs'][0]['transcript_format']=='.lab')
        assert lab['textgrids']==provenance['textgrids']
        report['checks'].append('registered/custom dictionary switching and cancelled-picker preservation; real LAB and reconstructed words TextGrid alignments exactly identical')
        for width,height in [(1920,1080),(1440,900),(1280,800)]:
            w.resize(width,height);pause(200)
            for theme in ('light','dark'):
                js('document.documentElement.dataset.theme='+json.dumps(theme));pause(700)
                assert js('document.documentElement.scrollWidth<=innerWidth')
                geo=js('[...document.querySelectorAll(".mfa-form select")].map(e=>{const r=e.getBoundingClientRect();return {width:r.width,height:r.height}})')
                assert max(g['width'] for g in geo)-min(g['width'] for g in geo)<2
                w.view.grab().save(str(out/f'qt-{width}-{theme}.png'));report['layouts'].append(dict(width=width,height=height,theme=theme,controls=geo))
        assert originals=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.iterdir()}
        report['native_keys']=js('window.nativeKeys');assert any(e['trusted'] for e in report['native_keys'])
        report['checks'].append('six native light/dark layouts and matching dropdown widths; original corpus hashes unchanged')
        report['success']=True
    finally:
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
        if not report['success']:w.view.grab().save(str(out/'failure.png'))
        # Test-owned shutdown bypasses the user-facing unsaved confirmation.
        # The production close guard remains unchanged.
        w.closing=True;w.close();pause(300);app.quit();print(json.dumps(dict(success=report['success'],out=str(out),checks=report['checks']),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
