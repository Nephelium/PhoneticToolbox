"""M01-F2 real Qt dialog, shared buttons, actual worker, files and normal close."""
import json
from pathlib import Path
import time
from uuid import uuid4
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication,QFileDialog,QLineEdit
from ptb_desktop.host import register_scheme,Workbench
from ptb_worker.local_acoustic_files import initialize_local_files
from baseline_support import create_fixture,RECIPES

ROOT=Path(__file__).resolve().parents[1]


def main():
    folder=ROOT/'output/validation/m01'/('task-window-'+uuid4().hex);folder.mkdir()
    inputs=folder/'输入';inputs.mkdir();cache=folder/'cache';cache.mkdir();initialize_local_files(cache)
    wav=create_fixture(inputs,RECIPES[0])
    (inputs/(wav.stem+'.TextGrid')).write_text('File type = "ooTextFile short"\n"TextGrid"\n0\n.8\n<exists>\n1\n"IntervalTier"\n"音节"\n0\n.8\n2\n0\n.4\n"阴平"\n.4\n.8\n"上声"\n',encoding='utf-8')
    register_scheme();app=QApplication(['PhoneticToolbox']);window=Workbench(ROOT/'frontend/dist',test=True,
        jobs_path=ROOT/'output/validation/p06/local-state.sqlite3',local_files_root=cache,reaper_binary=ROOT/'phonetic_toolbox/core/acoustic/reaper.exe');window.show()
    result={'success':False,'stages':[]};start=time.monotonic();stage=0;pending=False
    click=lambda label:"[...document.querySelectorAll('button')].find(b=>b.textContent.trim()==="+json.dumps(label,ensure_ascii=False)+")?.click()"
    def picker():
        dialog=app.activeModalWidget()
        if not isinstance(dialog,QFileDialog):QTimer.singleShot(100,picker);return
        dialog.setDirectory(str(inputs.parent))
        def accept():
            if time.monotonic()-start>100:return
            edit=dialog.findChild(QLineEdit,'fileNameEdit')
            if edit:edit.setText(str(inputs))
            selected=dialog.selectedFiles()
            if len(selected)==1 and Path(selected[0]).resolve()==inputs.resolve():dialog.accept()
            else:QTimer.singleShot(150,accept)
        QTimer.singleShot(150,accept)
    checks=[
        ("document.querySelector('.host-badge')?.textContent==='本地桌面'",click('参数估计')),
        ("!!document.querySelector('.m01-page')",click('选择音频目录')),
        ("document.querySelectorAll('.m01-file-list .file-row').length===1","document.querySelector('.m01-file-list .file-row')?.click()"),
        ("document.querySelectorAll('.m01-intervals button').length===2 && [...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='开始全列表分析'&&!b.disabled)",click('开始全列表分析')),
        ("document.querySelector('.m01-results strong')?.textContent==='1 / 1 已完成' && document.querySelector('.m01-page')?.textContent.includes('已保存3个结果文件') && !!document.querySelector('.m01-tiers input[type=checkbox]')","document.querySelector('.m01-tiers input[type=checkbox]')?.click()"),
        ("document.querySelector('.m01-tiers input[type=checkbox]')?.checked && [...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='保存当前层切分音频'&&!b.disabled)",click('保存当前层切分音频')),
        ("document.querySelector('.m01-results strong')?.textContent==='1 / 1 已完成'",None)]
    def finish(ok,error=None):
        timer.stop();result.update(success=ok,error=error)
        if not ok:window.view.grab().save(str(folder/'failed.png'));window.provider.close();window.service.close();window.closing=True
        window.close()
        QTimer.singleShot(5000,lambda:app.exit(1))
    def received(ok):
        nonlocal stage,pending
        pending=False
        if not ok:return
        if stage==4 and not (inputs/(wav.stem+'.ptb.json')).exists():return
        if stage==6 and len(list(inputs.glob('*.ptb.sqlite')))<3:return
        result['stages'].append(stage);action=checks[stage][1];stage+=1
        if stage==2:QTimer.singleShot(100,picker)
        if action:window.page.runJavaScript(action)
        else:
            timer.stop();window.page.runJavaScript("document.documentElement.dataset.theme='light'",lambda _:QTimer.singleShot(200,light))
    def light():
        window.view.grab().save(str(folder/'qt-light.png'));window.page.runJavaScript("document.documentElement.dataset.theme='dark'",lambda _:QTimer.singleShot(200,dark))
    def dark():
        window.view.grab().save(str(folder/'qt-dark.png'));window.resize(1000,760);QTimer.singleShot(300,narrow)
    def narrow():
        window.view.grab().save(str(folder/'qt-narrow.png'));finish(True)
    def tick():
        nonlocal pending
        if time.monotonic()-start>110:finish(False,'timeout at stage '+str(stage));return
        if pending or stage>=len(checks):return
        pending=True;window.page.runJavaScript(checks[stage][0],received)
    timer=QTimer();timer.timeout.connect(tick);timer.start(150)
    code=app.exec();result['normal_close']=window.closing and window.provider.closed and window.service.exit_code==0 and code==0
    result['success']=result['success'] and result['normal_close'];result['outputs']=[p.name for p in inputs.iterdir()]
    (folder/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(report=str(folder/'report.json'),**result),ensure_ascii=False),flush=True)
    window.page.deleteLater();app.processEvents();return 0 if result['success'] else 1


if __name__=='__main__':raise SystemExit(main())
