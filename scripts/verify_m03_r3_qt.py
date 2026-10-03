"""M03-D owned Qt, real local API/child, isolated copy of existing test schema."""
import json
import argparse
import os
import sqlite3
import time
import hashlib
from pathlib import Path
from uuid import uuid4
import numpy as np
from scipy.io import wavfile
from PyQt6.QtCore import QEventLoop,QTimer,Qt
from PyQt6.QtWidgets import QApplication,QFileDialog
from ptb_desktop.host import Workbench,register_scheme
from ptb_worker.local_acoustic_files import initialize_local_files

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser();extra=parser.add_mutually_exclusive_group();extra.add_argument('--include-private',action='store_true');extra.add_argument('--ranges',action='store_true');options=parser.parse_args()
    out=ROOT/'output/validation/m03-ui'/('qt-r3-'+uuid4().hex);out.mkdir(parents=True)
    db=out/'jobs.sqlite3'
    with sqlite3.connect((ROOT/'output/validation/p06/local-state.sqlite3').as_uri()+'?mode=ro',uri=True) as source,sqlite3.connect(db) as target:
        assert not source.execute("SELECT 1 FROM jobs WHERE state IN ('queued','running','cancel_requested')").fetchone();source.backup(target)
    cache=out/'cache';cache.mkdir();initialize_local_files(cache)
    inputs=out/'inputs';inputs.mkdir();saved=out/'saved';saved.mkdir()
    private=Path(r'C:\Users\13680\Desktop\project\音频数据\EGG测试\3.wav')
    original_hash=hashlib.sha256(private.read_bytes()).hexdigest()
    (inputs/'3.wav').write_bytes(private.read_bytes())
    originals={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.iterdir()}
    os.environ['PTB_EGG_PYTHON']=str(ROOT/'.venv/m03-compatible/python.exe')
    register_scheme();app=QApplication(['M03-owned-QA']);app.setApplicationName('M03-owned-QA')
    w=Workbench(ROOT/'frontend/dist',test=True,jobs_path=db,local_files_root=cache,vocal_profile=out/'vocal-profile')
    w.resize(1440,1000);w.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating);w.show()
    save_target=[saved/'EGG-IF-four.png']
    QFileDialog.getSaveFileName=lambda *a,**k:(str(save_target[0]),'PNG (*.png)')
    checks=[];report=dict(success=False,checks=checks,schema_applied=[])
    QFileDialog.getExistingDirectory=lambda *a,**k:str(saved if '结果' in str(a) else inputs)
    def js(code):
        loop=QEventLoop();box=[];w.page.runJavaScript(code,lambda v:(box.append(v),loop.quit()));QTimer.singleShot(5000,loop.quit);loop.exec();return box[0] if box else None
    def pause(milliseconds=100):loop=QEventLoop();QTimer.singleShot(milliseconds,loop.quit);loop.exec()
    def until(code,seconds=60):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            if js(code):return
            pause()
        raise AssertionError('UI timeout: '+code+'; '+str(js('document.querySelector(".egg-page")?.innerText.slice(0,1800)')))
    def click(text):js('[...document.querySelectorAll("button")].find(b=>b.offsetParent&&b.textContent.trim()==='+json.dumps(text,ensure_ascii=False)+')?.click()')
    def fill(label,value):js('(()=>{const e=document.querySelector('+json.dumps('input[aria-label="'+label+'"]')+');e.value='+json.dumps(value)+';e.dispatchEvent(new Event("input",{bubbles:true}));e.dispatchEvent(new Event("change",{bubbles:true}));})()')
    def select(name):js('(()=>{const e=document.querySelector(".egg-source select");e.value=[...e.options].find(o=>o.textContent==='+json.dumps(name,ensure_ascii=False)+').value;e.dispatchEvent(new Event("change",{bubbles:true}));})()')
    def ready():until('document.querySelectorAll(".egg-four-plots .scientific-plot>svg").length===4')
    def wait_exports():until('![...document.querySelectorAll(".egg-history .task-row [role=status]")].some(e=>["排队中","运行中","正在取消"].includes(e.textContent))',120)

    def idle():until('[...document.querySelectorAll(".egg-page button")].some(b=>b.textContent==="保存 CSV / 三图"&&!b.disabled)',60)
    try:
        until('document.documentElement.style.getPropertyValue("--font").length>0');click('EGG 信号分析');until('!!document.querySelector(".egg-page")')
        click('打开 WAV 目录');until('document.querySelectorAll(".egg-source option").length>=2')
        started=time.monotonic();select('3.wav');idle();ready();checks.append(dict(check='real source first display',seconds=time.monotonic()-started))
        assert js('document.querySelector(".module-toolbar")?.innerText.includes("保存参数草稿")')
        assert not js('document.querySelector(".egg-page").innerText.includes("声门活动")')
        assert js("document.querySelector('input[aria-label=\"EGG 高通频率\"]')?.type")=="number"
        assert js("document.querySelector('input[aria-label=\"EGG 低通频率\"]')?.value")=="2000"
        before=js('document.querySelector(".egg-history summary").innerText')
        started=time.monotonic();fill('EGG 选区起点',40);fill('EGG 选区时长',.5);pause(40);idle();checks.append(dict(check='Qt ROI update',seconds=time.monotonic()-started))
        started=time.monotonic();fill('EGG 微观窗口',100);pause(40);idle();checks.append(dict(check='Qt micro update',seconds=time.monotonic()-started))
        assert js('document.querySelector(".egg-history summary").innerText')==before
        checks.append('Qt gestures do not append persistent tasks')
        w.view.grab().save(str(out/'r3-four-plots.png'))
        click('自动 dB');pause(60);idle()
        assert js("Number(document.querySelector('input[aria-label=\"EGG dB 上限\"]').value)-Number(document.querySelector('input[aria-label=\"EGG dB 下限\"]').value)")==50
        checks.append('Qt one-shot automatic 50 dB display range')
        click('逆滤波 IF');until('document.querySelectorAll(".inverse-grid .scientific-plot>svg").length===4',120)
        until('document.querySelectorAll("dialog .audio-transport").length===2')
        assert js('document.querySelector(".inverse-actions").innerText.includes("选择目录保存完整结果")')
        assert js('document.querySelector(".inverse-actions").innerText.includes("返回分析")')
        assert js('document.querySelector("dialog").innerText.includes("225 个固定 3 ms 窗")')
        checks.append('Qt IF opens automatically, actions at top, actual overlap diagnostic')
        from PyQt6.QtGui import QImage
        for index,(label,count) in enumerate([('全部分开 · 6 图',6),('音频 + IF · 4 图',4),('EGG + IF · 4 图',4),('音频 + EGG + IF · 2 图',2)]):
            click(label);until('document.querySelectorAll(".inverse-grid .scientific-plot>svg").length==='+str(count));pause(150)
            save_target[0]=saved/f'EGG-IF-{index}-{count}.png'
            click(f'保存当前 {count} 图 PNG');deadline=time.monotonic()+30
            while not save_target[0].exists() and time.monotonic()<deadline:pause(100)
            im=QImage(str(save_target[0]))
            assert not im.isNull() and im.width()>1000 and im.height()>500
            assert abs(im.dotsPerMeterX()*.0254-300)<1
            w.view.grab().save(str(out/f'r3-inverse-{count}-{index}.png'))
            checks.append(dict(check='Qt native PNG uses current layout',panels=count,file=save_target[0].name))
        js("document.querySelector('.egg-wave-options input').click()")
        until('[...document.querySelectorAll("dialog h3")].some(e=>e.textContent==="滤波 EGG")')
        assert js('document.querySelectorAll("dialog .audio-transport").length')==2
        checks.append('Qt optional full EGG display with two existing audio players')
        click('选择目录保存完整结果')
        deadline=time.monotonic()+30
        while not list(saved.glob('*.ptb.json')) and time.monotonic()<deadline:pause(100)
        until('document.querySelector("dialog").innerText.includes("已保存")')
        meta=json.loads(next(saved.glob('*.ptb.json')).read_text('utf-8'))
        assert len(meta['inverse_view']['full_egg_values'])==22050
        assert meta['inverse_view']['sample_rate_hz']==44100
        checks.append('Qt immutable full filtered EGG and metadata saved')
        assert hashlib.sha256(private.read_bytes()).hexdigest()==original_hash
        report['success']=True
    except Exception as exc:
        report['error']=str(exc);w.view.grab().save(str(out/'failed.png'));raise
    finally:
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(out,flush=True);w.close();pause(300)
if __name__=='__main__':main()
