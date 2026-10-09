"""M11-R2 real native resource selection, failure retention and hashed task inputs."""
import argparse
import hashlib
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


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(bundle,out,lab=None,registry=None):
    bundle,out=Path(bundle),Path(out)
    out.mkdir(parents=True);print(out,flush=True)
    components=out/'components';components.mkdir()
    original_registry=Path(registry) if registry else ROOT/'output/m11c-028b881d/registry.json'
    shutil.copyfile(original_registry,components/'registry.json')
    before_registry=json.loads((components/'registry.json').read_text('utf8'))
    os.environ['PTB_M11_COMPONENT_ROOT']=str(components)
    resources=Path.home()/'Desktop/PhoneticToolbox/mfa_models'
    model=resources/'acoustic/mandarin.zip';dictionary=resources/'dictionary/mandarin_pinyin_tab.dict'
    expected=dict(model_sha256=digest(model),dictionary_sha256=digest(dictionary))
    inputs=out/'public';generate(inputs,word='a1')
    source_hashes={str(p):digest(p) for p in (model,dictionary)}
    natural=out/'natural'
    if lab:
        natural.mkdir()
        for source in (lab,lab.with_suffix('.wav')):
            source_hashes[str(source)]=digest(source);shutil.copyfile(source,natural/source.name)
    original_registry_hash=digest(original_registry)
    saved=out/'saved';saved.mkdir()
    from ptb_worker.local_workspace import prepare_workspace
    db,cache=prepare_workspace(out/'state',bundle/'backend/migrations')
    with sqlite3.connect(db) as target:assert not target.execute("SELECT 1 FROM jobs WHERE state IN ('queued','running','cancel_requested')").fetchone()
    register_scheme();app=QApplication(['M11-R2-owned-QA'])
    w=Workbench(bundle/'frontend/dist',test=True,jobs_path=db,local_files_root=cache,vocal_profile=out/'vocal-profile')
    w.resize(1920,1080);w.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen);w.show();w.page.setAudioMuted(True)
    chosen_model=str(model);chosen_dictionary=str(dictionary);chosen_corpus=inputs
    picker_calls=[]
    def file_picker(*a,**k):
        title=str(a[1]);picker_calls.append(title)
        return (chosen_model if title.endswith('model') else chosen_dictionary,'')
    QFileDialog.getOpenFileName=file_picker
    QFileDialog.getExistingDirectory=lambda *a,**k:str(chosen_corpus if 'corpus' in str(a) else saved)
    report=dict(success=False,checks=[],layouts=[],expected=expected,scope='actual hidden Windows Qt/QWebChannel and MFA 3.3.8; isolated component registry and database; source files read-only')
    def pause(ms=100):
        loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
    def js(code):
        loop=QEventLoop();box=[];w.page.runJavaScript(code,lambda value:(box.append(value),loop.quit()));QTimer.singleShot(6000,loop.quit);loop.exec()
        if not box:raise RuntimeError('JS timeout')
        return box[0]
    def until(code,seconds=180):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            if js(code):return
            pause()
        raise AssertionError(code+' '+str(js('document.querySelector(".mfa-page")?.innerText.slice(-5000)')))
    def idle():until('document.querySelector(".mfa-page").getAttribute("aria-busy")==="false"')
    def click(label):
        point=js('(()=>{const e=[...document.querySelectorAll("button")].find(e=>e.offsetParent&&e.textContent.trim()==='+json.dumps(label)+');if(!e||e.disabled)return null;e.scrollIntoView({block:"nearest"});const r=e.getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2}})()')
        assert point,label
        QTest.mouseClick(w.view.focusProxy() or w.view,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,QPoint(round(point['x']),round(point['y'])));pause()
    def select(label,value):
        assert js('(()=>{const e=[...document.querySelectorAll(".mfa-page label")].find(e=>e.childNodes[0].textContent.trim()==='+json.dumps(label)+')?.querySelector("select");if(!e||e.disabled)return false;e.value='+json.dumps(value)+';e.dispatchEvent(new Event("change",{bubbles:true}));return true;})()')
        pause();idle()
    def selected(label):return js('[...document.querySelectorAll(".mfa-form label")].find(e=>e.childNodes[0].textContent.trim()==='+json.dumps(label)+').querySelector("select").selectedOptions[0]?.textContent')
    def can_start():return js('[...document.querySelectorAll(".mfa-page button")].find(e=>e.textContent.trim()==="开始对齐").disabled===false')
    def task():
        click('选择语料目录');idle();assert can_start()
        click('开始对齐');idle();until('document.querySelectorAll(".result-row").length===2&&document.querySelector(".job-state")?.textContent.includes("完整结果")')
        click('保存完整结果到输出目录');idle()
        path=max(saved.rglob('m11-provenance.json'),key=lambda p:p.stat().st_mtime)
        latest=json.loads(path.read_text('utf8'))
        assert latest['model']['sha256']==expected['model_sha256']
        assert latest['dictionary_sha256']==expected['dictionary_sha256']
        with sqlite3.connect(db.as_uri()+'?mode=ro',uri=True) as store:
            row=store.execute("SELECT snapshot FROM jobs WHERE json_extract(snapshot,'$.operation')='mfa_alignment' ORDER BY created_at DESC LIMIT 1").fetchone()
        assert row is not None and json.loads(row[0])['request']['dictionary'] is None
        return latest
    try:
        until('!!document.querySelector("nav")');click('MFA 自动标注');until('!!document.querySelector(".mfa-page")');idle()
        until('document.querySelector(".mfa-form select").options.length>1')
        js('window.nativeClicks=[];document.addEventListener("click",e=>window.nativeClicks.push(e.isTrusted),true)')
        before_model=selected('声学模型')
        chosen_model='';select('声学模型','import');assert selected('声学模型')==before_model
        chosen_model=str(model)
        select('转写来源','.lab');click('选择语料目录');idle();assert can_start()
        select('发音词典','import');assert '待检查' in selected('发音词典') and not can_start()
        click('检查并应用所选资源');idle()
        assert js('document.querySelector(".mfa-page").innerText.includes("声学模型与词典音素集不匹配")')
        assert json.loads((components/'registry.json').read_text('utf8'))==before_registry
        assert selected('声学模型')==before_model and not can_start()
        report['checks'].append('cancelled top model picker keeps selected resource; exact pinyin dictionary with old IPA model fails real MFA; old registry retained and submission blocked')
        select('声学模型','import');assert 'mandarin.zip' in selected('声学模型') and '待检查' in selected('声学模型')
        w.view.grab().save(str(out/'pending-resources.png'))
        click('检查并应用所选资源');idle()
        assert selected('声学模型')=='mandarin.zip' and selected('发音词典')=='mandarin_pinyin_tab.dict' and can_start()
        registered=json.loads((components/'registry.json').read_text('utf8'))
        existed=any(m['model_sha256']==expected['model_sha256'] and m['dictionary_sha256']==expected['dictionary_sha256'] for m in before_registry['models'])
        assert len(registered['models'])==len(before_registry['models'])+(0 if existed else 1)
        applied=next(m for m in registered['models'] if m['model_sha256']==expected['model_sha256'])
        assert applied['dictionary_sha256']==expected['dictionary_sha256']
        assert js('Object.entries(localStorage).filter(([key])=>key.startsWith("ptb.v3.m11:")).some(([,value])=>JSON.parse(value).model==='+json.dumps(applied['id'])+')')
        receipt=json.loads(Path(registered['runtimes'][0]['receipt']).read_text('utf8'))
        assert receipt['probe_word']=='a1'
        report['receipt']=receipt
        report['checks'].append('top model and dictionary import; reuse existing environment without picking a directory; real a1 self-test and automatic pair application; old model remains')
        chosen_dictionary='';select('发音词典','import');assert selected('发音词典')=='mandarin_pinyin_tab.dict' and can_start()
        chosen_dictionary=str(dictionary)
        select('声学模型','import');assert not can_start();click('取消待检查选择');idle();assert selected('声学模型')=='mandarin.zip' and can_start()
        report['checks'].append('dictionary picker cancellation and explicit discard preserve applied pair')
        report['public_task']=task()
        report['checks'].append('actual public a1 audio task and saved provenance use exact selected model/dictionary; custom dictionary override cleared')
        if lab:
            chosen_corpus=natural;report['natural_task']=task()
            labels=[e[2] for tg in report['natural_task']['textgrids'] for tier in tg['tiers'] if tier['name'].endswith('words') for e in tier['entries'] if e[2]]
            assert labels==lab.read_text('utf8').split()
            report['checks'].append('specified real WAV/LAB task completes; six word labels exactly match source LAB; original bytes unchanged; boundary accuracy not asserted')
        click('组件安装与环境检查');idle()
        assert not js('[...document.querySelectorAll(".component-manager .resource-line")].some(e=>e.textContent.includes("声学模型")||e.textContent.includes("配套词典"))')
        for width,height in [(1920,1080),(1440,900),(1280,800)]:
            w.resize(width,height);pause(300)
            for theme in ('light','dark'):
                js('document.documentElement.dataset.theme='+json.dumps(theme));pause(700)
                assert js('document.documentElement.scrollWidth<=innerWidth')
                geo=js('[...document.querySelectorAll(".mfa-form select")].map(e=>{const r=e.getBoundingClientRect();return {width:r.width,height:r.height}})')
                assert max(g['width'] for g in geo)-min(g['width'] for g in geo)<2
                w.view.grab().save(str(out/f'qt-{width}-{theme}.png'));report['layouts'].append(dict(width=width,height=height,theme=theme,controls=geo))
        report['checks'].append('component manager has only environment/offline controls; six light/dark layouts; native pointer activation')
        assert all(digest(p)==h for p,h in source_hashes.items()) and digest(original_registry)==original_registry_hash
        report['picker_calls']=picker_calls;report['native_clicks']=js('window.nativeClicks');assert any(report['native_clicks'])
        report['original_hashes_unchanged']=True;report['success']=True
    finally:
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
        if not report['success']:w.view.grab().save(str(out/'failure.png'))
        w.closing=True;w.close();pause(300);app.quit();print(json.dumps(dict(success=report['success'],out=str(out),checks=report['checks']),ensure_ascii=False),flush=True)
    return 0 if report['success'] else 1


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--lab',type=Path);args=parser.parse_args()
    out=ROOT/'output/validation/m11-r2'/('qt-'+uuid4().hex)
    return verify(ROOT,out,args.lab)


if __name__=='__main__':raise SystemExit(main())
