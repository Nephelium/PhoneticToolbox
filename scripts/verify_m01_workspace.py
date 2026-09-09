"""M01-E owned Qt window, real Qt directory dialog, shared UI and shutdown.

Only new synthetic files below output/validation/m01 are created. No IAB calls.
"""
from pathlib import Path
from uuid import uuid4
import json
import time
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication,QFileDialog,QLineEdit
from ptb_desktop.host import register_scheme,Workbench

ROOT=Path(__file__).resolve().parents[1]


def main():
    folder=ROOT/'output/validation/m01'/('workspace-'+uuid4().hex);folder.mkdir(parents=True)
    inputs=folder/'输入';inputs.mkdir()
    wav=(ROOT/'frontend/src/assets/SYN-EGG-44100.wav').read_bytes()
    for i in range(17):(inputs/f'声调{i:02}.wav').write_bytes(wav)
    (inputs/'声调00.TextGrid').write_text('File type = "ooTextFile short"\n"TextGrid"\n0\n.8\n<exists>\n2\n"IntervalTier"\n"音节"\n0\n.8\n2\n0\n.4\n"阴平"\n.4\n.8\n"上声"\n"IntervalTier"\n"IPA"\n0\n.8\n1\n0\n.8\n"ɑ̃˥"\n',encoding='utf-8')
    register_scheme();app=QApplication(['PhoneticToolbox']);window=Workbench(ROOT/'frontend/dist',test=True);window.show()
    observations=[];started=time.monotonic();stage=0;running=False;result={'success':False}
    click=lambda label:"[...document.querySelectorAll('button')].find(b=>b.textContent.trim()==="+json.dumps(label,ensure_ascii=False)+")?.click()"
    def dialog_accept():
        dialog=app.activeModalWidget()
        if isinstance(dialog,QFileDialog):
            # QFileSystemModel loads asynchronously. Select the child in its parent,
            # then verify the real dialog selection before accepting the grant.
            dialog.setDirectory(str(inputs.parent))
            def accept_selected():
                if time.monotonic()-started>40:return
                edit=dialog.findChild(QLineEdit,'fileNameEdit')
                if edit is not None:edit.setText(str(inputs))
                selected=dialog.selectedFiles()
                result['dialog_selection']={'selected':selected,'expected':str(inputs)}
                if len(selected)==1 and Path(selected[0]).resolve()==inputs.resolve():dialog.accept()
                else:QTimer.singleShot(150,accept_selected)
            QTimer.singleShot(150,accept_selected)
        else:QTimer.singleShot(100,dialog_accept)
    checks=[
        ("document.querySelector('.host-badge')?.textContent==='本地桌面'",click('参数估计')),
        ("!!document.querySelector('.m01-page')",click('选择音频目录')),
        ("document.querySelectorAll('.m01-file-list .file-row').length===17","document.querySelector('.m01-file-list .file-row')?.click()"),
        ("document.querySelectorAll('.m01-intervals button').length===2 && document.querySelectorAll('.wave-track').length===1", "document.querySelector('.display-options input')?.click()"),
        ("document.querySelectorAll('.wave-track').length===2", "document.querySelector('.display-options input')?.click();document.querySelectorAll('.display-options input')[1]?.click()"),
        ("!!document.querySelector('.spectrogram-canvas canvas') && document.querySelectorAll('.wave-track').length===1",click('编辑14项设置')),
        ("document.querySelectorAll('.m01-settings label').length===10",click('REAPER设置 · 4')),
        ("document.querySelectorAll('.m01-settings label').length===4",click('取消')),
        ("!document.querySelector('dialog')",click('选择输出参数')),
        ("document.querySelectorAll('.parameter-grid input').length===80",click('全不选')),
        ("document.querySelector('dialog .primary')?.disabled===true",click('取消')),
        ("!document.querySelector('dialog')","document.querySelector('.m01-intervals button')?.click()"),
        ("document.querySelector('.m01-batch-bar strong')?.textContent==='处理列表中的 17 个文件'", "document.querySelector('.signal-panel').scrollTop=0"),
        ("(()=>{const r=s=>document.querySelector(s).getBoundingClientRect();return r('.wave-track .time-axis').bottom<r('.signal-panel').bottom && r('.spectrogram-canvas').bottom<=r('.signal-panel').bottom && Math.abs(r('.m01-output-row').top-r('.m01-directory-bar>.primary').top)<5 && !!document.querySelector('.playback-seek input')})()",None),
    ]
    def finish(code,error=None):
        timer.stop();result.update(success=code==0,observations=observations,error=error)
        if code:
            window.provider.close();window.service.close();window.closing=True
        window.close()
        QTimer.singleShot(5000,close_timeout)
    def close_timeout():
        result.update(success=False,error='normal window close timed out')
        window.provider.close();window.service.close();window.closing=True;window.close();app.exit(1)
    def capture(value):
        result['layout']=value
        window.page.runJavaScript("document.documentElement.dataset.theme='light'",lambda _:QTimer.singleShot(250,light))
    def light():
        window.view.grab().save(str(folder/'qt-m01-light.png'))
        window.page.runJavaScript("document.documentElement.dataset.theme='dark'",lambda _:QTimer.singleShot(250,dark))
    def dark():
        window.view.grab().save(str(folder/'qt-m01-dark.png'));finish(0)
    def received(ok):
        nonlocal stage,running
        running=False
        if not ok:return
        observations.append({'stage':stage,'passed':True})
        action=checks[stage][1];stage+=1
        if action:
            if stage==2:QTimer.singleShot(150,dialog_accept)
            window.page.runJavaScript(action)
        else:
            timer.stop()
            window.page.runJavaScript("({files:document.querySelectorAll('.m01-file-list .file-row').length,channels:document.querySelectorAll('.wave-track').length,spectrogram:!!document.querySelector('.spectrogram-canvas canvas'),rowHeight:document.querySelector('.m01-file-list .file-row').getBoundingClientRect().height,waveHeight:document.querySelector('.wave-track svg').getBoundingClientRect().height,labels:[...document.querySelectorAll('.m01-intervals button')].map(x=>x.textContent),width:innerWidth,scroll:document.querySelector('main').scrollWidth,mainWidth:document.querySelector('main').clientWidth,body:document.body.innerText})",capture)
    def tick():
        nonlocal running
        if time.monotonic()-started>45:
            window.view.grab().save(str(folder/'failed.png'))
            finish(1,'timeout at stage '+str(stage));return
        if running or stage>=len(checks):return
        running=True;window.page.runJavaScript(checks[stage][0],received)
    timer=QTimer();timer.timeout.connect(tick);timer.start(100)
    code=app.exec();result['service_exit_code']=window.service.exit_code
    result['normal_close']=window.closing and window.provider.closed and window.service.exit_code==0
    result['success']=result['success'] and result['normal_close'] and code==0
    (folder/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'report':str(folder/'report.json'),'success':result['success'],'service_exit_code':window.service.exit_code},ensure_ascii=False),flush=True)
    window.page.deleteLater();app.processEvents();return 0 if result['success'] else 1


if __name__=='__main__':raise SystemExit(main())
