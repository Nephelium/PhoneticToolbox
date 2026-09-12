"""Actual Qt time proportions, selection, zoom, and readable aligned tiers."""
import json
from pathlib import Path
import sys
import time
from uuid import uuid4
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication,QFileDialog,QLineEdit
from ptb_desktop.host import register_scheme,Workbench
from ptb_worker.local_workspace import prepare_workspace

ROOT=Path(__file__).resolve().parents[1]
inputs=Path(sys.argv[1]).resolve(strict=True)
out=ROOT/'output/validation/desktop-repair'/('timeline-'+uuid4().hex)
database,cache=prepare_workspace(out/'state',ROOT/'backend/migrations')
register_scheme();app=QApplication(['TextGrid time track validation'])
window=Workbench(ROOT/'frontend/dist',test=True,jobs_path=database,local_files_root=cache,
                 reaper_binary=ROOT/'phonetic_toolbox/core/acoustic/reaper.exe')
window.show()
click=lambda label:"[...document.querySelectorAll('button')].find(b=>b.offsetParent&&b.textContent.trim()==="+json.dumps(label,ensure_ascii=False)+")?.click()"
geometry="""(()=>{const lane=document.querySelector('.textgrid-lane'),wave=document.querySelector('.wave-track svg'),parts=[...document.querySelectorAll('.textgrid-interval')];if(!lane||!wave||parts.length!==3)return false;const a=lane.getBoundingClientRect(),b=wave.getBoundingClientRect();return Math.abs(a.left-b.left)<1&&Math.abs(a.width-b.width)<1&&parts.every((p,i)=>Math.abs(p.getBoundingClientRect().width/a.width-[.1,.2,.7][i])<.001)})()"""
stages=[
    ('!!document.querySelector(".host-badge")',click('参数估计')),
    ('!!document.querySelector(".m01-page")',click('选择音频目录')),
    ('document.querySelectorAll(".m01-file-entry").length===1','document.querySelector(".m01-file-entry button").click()'),
    (geometry,'document.querySelectorAll(".textgrid-interval")[1].click();document.querySelector(".textgrid-timeline").scrollIntoView({block:"center"})'),
    ('[...document.querySelectorAll(".selection-controls input")].map(e=>Number(e.value)).join(",")==="0.1,0.3"',None),
    ('true','document.querySelector(`button[aria-label="放大波形"]`).click()'),
    ('Math.abs(parseFloat(document.querySelectorAll(".textgrid-interval")[1].style.width)-40)<.001&&document.querySelectorAll(".textgrid-interval")[2].dataset.xmax==="1"',None),
]
index=0;pending=False;started=time.monotonic();checks=[];success=False
def close(ok):
    global success
    success=ok;timer.stop();dialogs.stop();window.closing=True;window.close()
def received(ok):
    global index,pending
    pending=False
    if not ok:return
    checks.append(index)
    if index in (4,6):window.view.grab().save(str(out/f'timeline-{index}.png'))
    action=stages[index][1];index+=1
    if action:window.page.runJavaScript(action)
    if index==len(stages):close(True)
def tick():
    global pending
    if time.monotonic()-started>60:close(False);return
    if pending:return
    pending=True;window.page.runJavaScript(stages[index][0],received)
def dialog_tick():
    dialog=app.activeModalWidget()
    if isinstance(dialog,QFileDialog):
        edit=dialog.findChild(QLineEdit,'fileNameEdit')
        if edit:
            dialog.setDirectory(str(inputs.parent));edit.setText(str(inputs))
            if dialog.selectedFiles() and Path(dialog.selectedFiles()[0]).absolute()==inputs:dialog.accept()
timer=QTimer();timer.timeout.connect(tick);timer.start(500)
dialogs=QTimer();dialogs.timeout.connect(dialog_tick);dialogs.start(200)
app.exec()
(out/'report.json').write_text(json.dumps({'success':success,'checks':checks},indent=2),encoding='utf-8')
print(out/'report.json');window.page.deleteLater();app.processEvents()
raise SystemExit(0 if success else 1)
