"""Current M16 manual figures from an owned Qt window and synthetic input."""
from __future__ import annotations
import argparse
import re
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
for key in ('QT_ENABLE_HIGHDPI_SCALING','QT_SCALE_FACTOR','QT_SCREEN_SCALE_FACTORS'):os.environ.pop(key,None)
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS','--mute-audio')
from workbench_source import configure
from verification_artifacts import verification_run
from manual.capture_locale import CaptureLocale


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--locale',choices=('en','ja'),required=True);parser.add_argument('--frontend-dist',type=Path,default=ROOT/'frontend/dist');parser.add_argument('--scene',action='append',default=[]);args=parser.parse_args()
    args.frontend_dist=args.frontend_dist.resolve(strict=True)
    ui=CaptureLocale(args.locale,('recording','workbench','components'))
    sources={str(i):dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for i,p in enumerate([Path(__file__),args.frontend_dist/'index.html',ROOT/'frontend/src/modules/recording/RecordingPage.vue'])}
    configure()
    from PyQt6 import sip
    from ptb_worker.local_workspace import prepare_workspace
    from PyQt6.QtCore import QEventLoop,QTimer,Qt
    from PyQt6.QtWidgets import QApplication
    from ptb_desktop.host import Workbench,register_scheme
    from m16_test_backend import Backend

    class DemoBackend(Backend):
        def query_devices(self):
            return [dict(name='示例声卡·合成双通道',hostapi=0,max_input_channels=2,max_output_channels=2,default_samplerate=48000),dict(name='示例声卡·合成单通道',hostapi=0,max_input_channels=1,max_output_channels=0,default_samplerate=48000)]
    register_scheme();app=QApplication(['M16 owned manual update'])
    with verification_run('m16-localized-update',args.locale,'Current Windows Qt; synthetic endpoints only; primary-screen dimensions, DPR 1.5/page 100%, light palette, hidden maximized window') as (out,scratch):
        project=scratch/'示例录音工程';project.mkdir()
        db,cache=prepare_workspace(scratch/'state',ROOT/'backend/migrations')
        window=Workbench(args.frontend_dist,test=True,jobs_path=db,local_files_root=cache,vocal_profile=scratch/'vocal',start_module='M16')
        window.resize(window.screen().geometry().size())
        window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
        window.showMaximized();window.page.setAudioMuted(True)
        bridge=window.bridge.recording_bridge();bridge.service.backend=DemoBackend();bridge.choose=lambda purpose:bridge.service.grant(project,purpose)
        window.profile.downloadRequested.disconnect(window.save_download)
        downloads=[]
        def download(item):
            item.setDownloadDirectory(str(scratch));item.setDownloadFileName(Path(item.suggestedFileName()).name);downloads.append(item);item.accept()
        window.profile.downloadRequested.connect(download)
        report=dict(success=False,locale=args.locale,sourceChapter='m16',sources=sources,checks=[],captures=[],scope='Actual Windows Qt/QWebChannel; synthetic microphone/EGG-role signals, no physical device or user project')
        def pause(ms=100):
            loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
        def js(code):
            code=re.sub(r'\[aria-label=([^\]"\']+)\]',lambda m:'[aria-label=\\"'+ui.label(m.group(1))+'\\"]',code)
            for text in ('已存入本地工程，尚未导出','后台处理已结束'):
                code=code.replace(json.dumps(text,ensure_ascii=False),json.dumps(ui.label(text),ensure_ascii=False))
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
            text=ui.key('modules.recording.text.afd4b783536b') if text=='收起' else text[:2]+ui.label(text[2:]) if text.startswith(('● ','■ ','▶ ')) and text not in ui.source.values() else ui.label(text)
            expr='[...document.querySelectorAll("button")].find(b=>b.offsetParent&&b.textContent.trim()==='+json.dumps(text,ensure_ascii=False)+')'
            until('!!('+expr+')&&!('+expr+').disabled');js(expr+'.click()');pause(150)
        def field(label, value, tag='input,textarea'):
            if label=='录音历史':
                target='document.querySelector(".take-select select")'
            else:
                match=re.fullmatch(r'物理输入 (\d+) 角色',label)
                label=ui.key('modules.recording.text.9f8ea3fb5121',{'v0':int(match[1])}) if match else ui.label(label)
                target='(()=>{const a='+json.dumps(label)+';return [...document.querySelectorAll(".recording-page input,.recording-page select,.recording-page textarea")].find(e=>e.getAttribute("aria-label")===a)??[...document.querySelectorAll(".recording-page label")].find(l=>l.textContent.trim().toLocaleLowerCase().startsWith(a.toLocaleLowerCase()))?.querySelector('+json.dumps(tag)+')})()'
            until('!!('+target+')&&!('+target+').disabled')
            assert js('(()=>{const e='+target+';if(!e||e.disabled)return false;e.value='+json.dumps(str(value))+';e.dispatchEvent(new Event("input",{bubbles:true}));e.dispatchEvent(new Event("change",{bubbles:true}));return true})()'),label
            pause(150)
        def capture(name,anchor,selector=None,crop=None):
            if args.scene and name not in args.scene:return
            if selector:js('document.querySelector('+json.dumps(selector)+').scrollIntoView({block:"center"})')
            pause(500);image=window.view.grab().toImage();raw=[image.width(),image.height()]
            assert window.isMaximized() and image.devicePixelRatio()==1.5 and raw in ([2560,1600],[2561,1601]),raw
            if raw!=[2560,1600]:image=image.copy(0,0,2560,1600)
            file=out/(name+'.png');assert image.save(str(file))
            regions={}
            for key,query in {'dialog':'.m16-dialog','device':'.devices-panel'}.items():
                box=js('(()=>{const e=document.querySelector('+json.dumps(query)+');if(!e)return null;const r=e.getBoundingClientRect();return [r.x,r.y,r.width,r.height]})()')
                if box:regions[key]=dict(selector=query,box=box,units='css-px')
            report['captures'].append(dict(name=name,file=file.name,image=[2560,1600],raw_image=raw,maximized=True,sha256=hashlib.sha256(file.read_bytes()).hexdigest(),regions=regions,auxiliary=bool(crop),**ui.metrics(js)))
            print('Captured '+name,flush=True)
        try:
            until('!!document.querySelector(".recording-page")')
            ui.prepare_qt(js,until)
            for text in (ui.key('workbench.settings'),ui.label('浅色')):
                expression='[...document.querySelectorAll("button")].find(b=>b.offsetParent&&(b.getAttribute("aria-label")==='+json.dumps(text)+'||b.textContent.trim()==='+json.dumps(text)+'))'
                until('!!('+expression+')');js(expression+'.click()');pause(180)
            until('document.documentElement.dataset.theme==="light"')
            js('document.querySelector("#tab-M16").click()');pause(180)
            js('[...document.querySelectorAll("button")].find(b=>b.getAttribute("aria-label")==='+json.dumps(ui.key('workbench.collapse_sidebar'))+')?.click()')
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
            until('document.querySelectorAll(".import-rows p").length===3');capture('m16-import-preview','m16-task-import');click('确认导入');click('保存清单')
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
            report['sourceChangesDuringRun']=[item['path'] for item in sources.values() if hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()!=item['sha256']]
            (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
            window.closing=True;window.close();app.processEvents();sip.delete(window);app.processEvents();sip.delete(app);print(out,flush=True)


if __name__=='__main__':main()
