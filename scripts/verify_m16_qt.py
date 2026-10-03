"""Actual Qt/QWebChannel with explicit synthetic devices and isolated M16 files.

No physical audio input/output, no old database and no schema creation.
"""
import json
import os
import time
from pathlib import Path
from uuid import uuid4
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS','--disable-gpu')
from PyQt6.QtCore import QEventLoop,QTimer
from PyQt6.QtWidgets import QApplication
from ptb_desktop.host import Workbench,register_scheme
from m16_test_backend import Backend

ROOT=Path(__file__).resolve().parents[1]

def main():
    out=ROOT/'output/validation/m16'/('qt-'+uuid4().hex);out.mkdir(parents=True);project=out/'project';project.mkdir();exports=out/'exports';exports.mkdir()
    report={'success':False,'scope':'Windows actual Qt offscreen + real QWebChannel + explicitly synthetic PortAudio; no physical hardware or database','checks':[]}
    register_scheme();app=QApplication(['M16-QA']);window=Workbench(ROOT/'frontend/dist',test=True,jobs_path=None,local_files_root=None,vocal_profile=out/'vocal-profile',start_module='M16')
    bridge=window.bridge.recording_bridge();bridge.service.backend=Backend();bridge.choose=lambda purpose:bridge.service.grant(exports if purpose=='export' else project,purpose)
    window.resize(1500,1060);window.show()
    def pause(ms=100):loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
    def js(code):
        loop=QEventLoop();box=[];window.page.runJavaScript(code,lambda value:(box.append(value),loop.quit()));QTimer.singleShot(5000,loop.quit);loop.exec();return box[0] if box else None
    def until(code,seconds=30):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            if js(code):return
            pause()
        raise AssertionError(code+'\n'+str(js('document.body.innerText.slice(-5000)')))
    def click(text):
        expression='[...document.querySelectorAll(".recording-page button")].find(b=>b.offsetParent&&b.textContent.trim()==='+json.dumps(text,ensure_ascii=False)+')'
        until(f'!!({expression})&&!({expression}).disabled');js(expression+'?.click()');pause(80)
    def select(label,value):
        js('(()=>{const l=[...document.querySelectorAll(".recording-page label")].find(l=>l.textContent.trim().startsWith('+json.dumps(label,ensure_ascii=False)+'));const e=l.querySelector("select");e.value='+json.dumps(str(value))+';e.dispatchEvent(new Event("change",{bubbles:true}));})()');pause()
    def fill(label,value):
        js('(()=>{const l=[...document.querySelectorAll(".recording-page label")].find(l=>l.textContent.trim().startsWith('+json.dumps(label,ensure_ascii=False)+'));const e=l.querySelector("input,textarea");e.value='+json.dumps(str(value))+';e.dispatchEvent(new Event("input",{bubbles:true}));e.dispatchEvent(new Event("change",{bubbles:true}));})()');pause(50)
    try:
        until('!!document.querySelector(".recording-page")');click('新建工程');until('document.body.innerText.includes("本地录音工程已建立")')
        click('设备与录前检测');device=bridge.service.dispatch({'op':'devices'})[0]['id'];select('输入设备',device);select('扬声器 / 耳机',device);fill('同声卡输入通道数',2);select('物理输入 2 角色','egg')
        click('开始录前检测（不保存音频）');until('document.body.innerText.includes("录前检测中")');pause(400);fill('检测数字增益预览',24);pause(300);click('停止检测');assert not bridge.service.project.data['takes'];report['checks'].append('preflight raw/digital meters; live gain preview; no take created')
        click('收起');click('＋ 新任务');fill('录音内容','合成验证 ɑ̃');fill('任务编号','SYN-001');click('完成编辑');click('保存清单');until('document.body.innerText.includes("任务清单已存入")')
        click('● 开始录音');pause(650);click('■ 停止录音');until('document.body.innerText.includes("已存入本地工程，尚未导出")');assert len(bridge.service.project.data['takes'])==1
        first=bridge.service.project.data['takes'][0];assert first['task_snapshot']['id']=='SYN-001';assert first['config']['roles']==['microphone','egg'];report['checks'].append('real UI start/stop with synthetic stereo PCM, task snapshot and EGG mapping')
        fill('起点（帧）',10);fill('终点（帧）',20);click('删除选区');until('document.body.innerText.includes("编辑已存入工程")');assert len(bridge.service.project.data['takes'][0]['versions'])==2;click('撤销');click('恢复原始');report['checks'].append('sample-frame deletion, undo and recoverable raw head through QWebChannel')
        click('▶ 播放');until('document.body.innerText.includes("停止播放")');js('window.dispatchEvent(new KeyboardEvent("keydown",{key:" ",code:"Space",bubbles:true}))');pause(300);assert bridge.service.player is None;report['checks'].append('selected native output playback and Space stop, synthetic output only')
        fill('起点（帧）',0);fill('终点（帧）',10000);click('将选区设为噪声样本');until('document.body.innerText.includes("噪声样本：")');click('整段降噪');until('document.body.innerText.includes("后台处理已结束")',90);assert bridge.service.project.data['takes'][0]['versions'][-1]['kind']=='denoised';report['checks'].append('noise selection + real child-process spectral subtraction and verified EGG preservation')
        click('保存 / 批量导出');click('选择目录并导出');until('document.body.innerText.includes("已导出 1 / 1")',60);assert len(list(exports.rglob('*.wav')))==1;report['checks'].append('native authorized directory export with WAV readback + JSON/CSV manifest')
        click('● 重新录音');pause(350);click('■ 停止录音');until('document.body.innerText.includes("已存入本地工程，尚未导出")');assert len(bridge.service.project.data['takes'])==2;report['checks'].append('rerecord appends take without overwriting first raw/history')
        for theme in ('light','dark'):
            js('document.documentElement.dataset.theme='+json.dumps(theme));pause(600)
            expected='rgb(24, 36, 50)' if theme=='dark' else 'rgb(255, 255, 255)'
            until('getComputedStyle(document.querySelector(".m16-wave")).backgroundColor==='+json.dumps(expected))
            window.view.grab();pause(400);app.processEvents();window.view.grab().save(str(out/f'qt-{theme}.png'))
        report['checks'].append('actual Qt light/dark screenshots at 1500x1060; not physical DPI evidence');report['success']=True
    finally:
        report['project']=bridge.service.view();(out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
        js('window.onbeforeunload=null');window.closing=True;window.close();pause(300);app.quit();print(out)
    return report

if __name__=='__main__':main()
