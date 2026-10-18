"""Current M16 manual figures from an owned Qt window and synthetic input."""
from __future__ import annotations
import base64
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
os.environ['QT_QPA_PLATFORM']='windows'
os.environ['QT_SCALE_FACTOR']='1'
os.environ['QT_SCREEN_SCALE_FACTORS']='1.5'
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS','--mute-audio')
from workbench_source import configure
from verification_artifacts import verification_run


def main():
    configure()
    from PyQt6.QtCore import QEventLoop,QTimer,Qt
    from PyQt6.QtWidgets import QApplication
    from ptb_desktop.host import Workbench,register_scheme
    from m16_test_backend import Backend

    class DemoBackend(Backend):
        def query_devices(self):
            return [dict(name='示例声卡·合成双通道',hostapi=0,max_input_channels=2,max_output_channels=2,default_samplerate=48000),dict(name='示例声卡·合成单通道',hostapi=0,max_input_channels=1,max_output_channels=0,default_samplerate=48000)]
    register_scheme();app=QApplication(['M16 owned manual update'])
    with verification_run('m16','manual-update','Current Windows Qt; synthetic endpoints only; primary-screen dimensions, DPR 1.5/page 100%, light palette, hidden maximized window') as (out,scratch):
        project=scratch/'示例录音工程';project.mkdir()
        window=Workbench(ROOT/'frontend/dist',test=True,vocal_profile=scratch/'vocal',start_module='M05')
        window.resize(window.screen().geometry().size())
        window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
        window.showMaximized();window.page.setAudioMuted(True)
        bridge=window.bridge.recording_bridge();bridge.service.backend=DemoBackend();bridge.choose=lambda purpose:bridge.service.grant(project,purpose)
        window.profile.downloadRequested.disconnect(window.save_download)
        downloads=[]
        def download(item):
            item.setDownloadDirectory(str(scratch));item.setDownloadFileName(Path(item.suggestedFileName()).name);downloads.append(item);item.accept()
        window.profile.downloadRequested.connect(download)
        report=dict(success=False,checks=[],captures=[],scope='Actual Windows Qt/QWebChannel; synthetic microphone/EGG-role signals, no physical device or user project')
        def pause(ms=100):
            loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
        def js(code):
            loop=QEventLoop();values=[];window.page.runJavaScript(code,lambda v:(values.append(v),loop.quit()));QTimer.singleShot(6000,loop.quit);loop.exec()
            if not values:raise RuntimeError('JavaScript timeout: '+code)
            return values[0]
        def until(code,seconds=35):
            end=time.monotonic()+seconds
            while time.monotonic()<end:
                if js(code):return
                pause()
            raise AssertionError(code+'\n'+str(js('document.body.innerText.slice(-2500)')))
        def click(text):
            expr='[...document.querySelectorAll("button")].find(b=>b.offsetParent&&b.textContent.trim()==='+json.dumps(text,ensure_ascii=False)+')'
            until('!!('+expr+')&&!('+expr+').disabled');js(expr+'.click()');pause(150)
        def field(label,value):
            js('(()=>{const l=[...document.querySelectorAll(".recording-page label")].find(l=>l.textContent.trim().startsWith('+json.dumps(label,ensure_ascii=False)+'));const e=l.querySelector("input,select,textarea");e.value='+json.dumps(str(value),ensure_ascii=False)+';e.dispatchEvent(new Event("input",{bubbles:true}));e.dispatchEvent(new Event("change",{bubbles:true}));})()');pause(150)
        def capture(name,anchor,selector=None,crop=None):
            if selector:js('document.querySelector('+json.dumps(selector)+').scrollIntoView({block:"center"})')
            pause(500);window.view.grab();pause(250);pix=window.view.grab()
            assert window.isMaximized() and pix.devicePixelRatio()==1.5
            assert pix.width() in (2560,2561) and pix.height() in (1600,1601),(pix.width(),pix.height())
            pix=pix.copy(0,0,2560,1600)
            if crop:
                box=js('document.querySelector('+json.dumps(crop)+').getBoundingClientRect().toJSON()')
                pix=pix.copy(round(box['x']*1.5),round(box['y']*1.5),round(box['width']*1.5),round(box['height']*1.5))
            file=out/(name+'.png');assert pix.save(str(file))
            report['captures'].append(dict(name=name,anchor=anchor,file=str(file),width=pix.width(),height=pix.height(),sha256=hashlib.sha256(file.read_bytes()).hexdigest(),theme=js('document.documentElement.dataset.theme'),dpr=1.5,pageScale=js('document.documentElement.style.zoom'),maximized=True,cropSelector=crop))
        try:
            until('!!document.querySelector(".module-workbench .workbench-left[aria-label=设备与采集]")')
            assert not js('document.body.innerText.includes("页面载入失败")')
            report['checks'].append('M05 restored: actual Qt renders device and video workbench without opening hardware')
            click('录音');until('document.querySelector(".recording-page")?.getAttribute("aria-busy")==="false"')
            js('document.documentElement.dataset.theme="light";document.documentElement.dataset.palette="absolutely"')
            js('document.querySelector("button[aria-label=收起侧栏]")?.click()')
            click('● 开始录音');capture('m16-preparation-required','m16-quick-start');click('知道了')
            click('新建工程');click('导入任务');capture('m16-import-help','m16-import-controls')
            click('下载 XLSX 示例')
            end=time.monotonic()+15
            while time.monotonic()<end and (not downloads or not downloads[0].isFinished()):pause()
            example=scratch/'录音任务表格示例.xlsx';assert example.is_file()
            from openpyxl import load_workbook
            wb=load_workbook(example,read_only=True);assert wb['录音任务']['E3'].value==2;wb.close()
            click('关闭');payload=base64.b64encode(example.read_bytes()).decode()
            js('(()=>{const raw=Uint8Array.from(atob('+json.dumps(payload)+'),c=>c.charCodeAt(0)),file=new File([raw],"录音任务表格示例.xlsx"),dt=new DataTransfer();dt.items.add(file);const e=document.querySelector(".recording-page input[type=file]");e.files=dt.files;e.dispatchEvent(new Event("change",{bubbles:true}));})()')
            until('document.body.innerText.includes("展开后 3 条任务")');capture('m16-import-preview','m16-task-import');click('确认导入');click('保存清单')
            click('设备与录前检测');ds=bridge.service.dispatch({'op':'devices'});field('输入设备',ds[1]['id']);capture('m16-single-channel','m16-device-controls','.devices-panel')
            field('输入设备',ds[0]['id']);field('同声卡输入通道数',2);field('物理输入 2 角色','egg');field('扬声器 / 耳机',ds[0]['id'])
            click('开始录前检测（不保存音频）');pause(350);capture('m16-preflight','m16-preparation','.devices-panel');click('停止检测');click('收起')
            click('● 开始录音');pause(1500);click('■ 停止录音');until('document.body.innerText.includes("已存入本地工程，尚未导出")')
            field('起点（帧）',0);field('终点（帧）',10000);click('将选区设为噪声样本');capture('m16-noise-sample','m16-processing')
            click('整段降噪');until('document.body.innerText.includes("后台处理已结束")',60);capture('m16-denoised-version','m16-processing')
            js('(()=>{const e=document.querySelector(".m16-wave-lanes"),b=e.getBoundingClientRect();e.dispatchEvent(new WheelEvent("wheel",{ctrlKey:true,deltaY:-120,bubbles:true,cancelable:true,clientX:b.left+b.width*.6}));})()');pause(350);capture('m16-time-zoom','m16-spectrogram');click('整段总览')
            click('保存 / 批量导出');field('导出范围','selected');field('WAV 格式','PCM_24');capture('m16-export-options','m16-export-workflow');capture('m16-export-options-detail','m16-export-controls',crop='.m16-dialog')
            field('导出范围','versions');assert js('document.querySelector("select[aria-label=导出版本]").disabled');capture('m16-export-history','m16-export-controls');click('取消')
            report['checks'].extend(['Actual XLSX download and import expands three tasks','Current single/stereo device levels and running preflight','Real noise selection and child-process denoise','Ctrl-wheel frame zoom and selected/history export controls'])
            report['success']=True
        finally:
            (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
            js('window.onbeforeunload=null');window.closing=True;window.close();pause(300);window.deleteLater();pause(800);app.quit();print(out,flush=True)


if __name__=='__main__':main()
