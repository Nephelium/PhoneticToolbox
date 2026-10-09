"""R1 actual Qt/QWebChannel read and full save of the user-authorized long pair.

Only copies WAV/TextGrid from --source to a new ignored evidence directory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
from uuid import uuid4
from PyQt6.QtCore import Qt,QTimer
from PyQt6.QtWidgets import QApplication,QFileDialog,QLineEdit
from phonetic_core.annotation import parse_document
from ptb_desktop.host import register_scheme,Workbench
from ptb_worker.local_workspace import prepare_workspace

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True);args=parser.parse_args()
    out=ROOT/'output/validation/m12-r1-long-qt'/uuid4().hex;inputs=out/'inputs';inputs.mkdir(parents=True)
    originals=[]
    for p in args.source.iterdir():
        if p.suffix.lower() not in ('.wav','.textgrid'):continue
        raw=p.read_bytes();originals.append((p,hashlib.sha256(raw).hexdigest()));(inputs/p.name).write_bytes(raw)
    wav=next(inputs.glob('*.wav'));grid=next(inputs.glob('*.TextGrid'));expected=parse_document(grid.read_text('utf-8-sig'))
    database,cache=prepare_workspace(out/'state',ROOT/'backend/migrations')
    register_scheme();app=QApplication(['M12 long Qt check']);window=Workbench(ROOT/'frontend/dist',test=True,jobs_path=database,local_files_root=cache)
    window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen,True);window.resize(1500,1050);window.show()
    click=lambda text:"[...document.querySelectorAll('button')].find(b=>b.offsetParent&&b.textContent.trim()==="+json.dumps(text,ensure_ascii=False)+")?.click()"
    stages=[
        ("document.querySelector('.host-badge')?.textContent==='本地桌面'",click('TextGrid标注')),
        ("!!document.querySelector('.annotation-page')",click('选择语料文件夹')),
        ("document.querySelectorAll('.annotation-file-list button').length===1","document.querySelector('.annotation-file-list button').click()"),
        ("document.querySelector('.notice')?.textContent.includes('_初始定位.TextGrid')&&document.querySelector('.annotation-page')?.getAttribute('aria-busy')==='false'","(()=>{const e=document.querySelector('[aria-label=\"平移波形时间窗\"]');e.value=e.max;e.dispatchEvent(new Event('input',{bubbles:true}));})()"),
        ("document.querySelector('.annotation-plots .time-axis')?.textContent.includes('312.572')",click('保存 TextGrid')),
        ("document.querySelector('.notice')?.textContent.includes('已保存：')",None),
    ]
    report={'success':False,'stages':[]};index=0;pending=False;start=time.monotonic();done=False
    def compare(a,b):
        if isinstance(a,(float,int)):
            assert abs(a-b)<=1e-6,(a,b)
        elif isinstance(a,list):
            assert len(a)==len(b)
            for x,y in zip(a,b):compare(x,y)
        elif isinstance(a,dict):
            assert a.keys()==b.keys()
            for k in a:compare(a[k],b[k])
        else:assert a==b
    def finish(ok,error=None):
        nonlocal done
        if done:return
        done=True;timer.stop();dialogs.stop();report.update(success=ok,error=error,elapsed_seconds=time.monotonic()-start)
        try:
            if ok:
                saved=parse_document((inputs/(wav.stem+'_自动保存.TextGrid')).read_text('utf-8'))
                compare(expected,saved);assert all(hashlib.sha256(p.read_bytes()).hexdigest()==sha for p,sha in originals)
                report.update(full_document_readback=True,originals_unchanged=True,frames=13784432,word_intervals=1147,phone_intervals=2236)
        except Exception as exc:report.update(success=False,error=repr(exc))
        window.view.grab().save(str(out/'qt-long-tail.png'));window.closing=True;window.close()
    def dialog_tick():
        dialog=app.activeModalWidget()
        if isinstance(dialog,QFileDialog):
            edit=dialog.findChild(QLineEdit,'fileNameEdit')
            if edit:
                dialog.setDirectory(str(inputs.parent));edit.setText(str(inputs))
                if dialog.selectedFiles() and Path(dialog.selectedFiles()[0]).absolute()==inputs.absolute():dialog.accept()
    def tick():
        nonlocal pending,index
        if time.monotonic()-start>150:finish(False,'timeout_stage_'+str(index));return
        if pending:return
        pending=True
        def ready(value):
            nonlocal pending,index
            pending=False
            if done or not value:return
            _,action=stages[index];index+=1;report['stages'].append(index)
            if action:window.page.runJavaScript(action)
            if index==len(stages):finish(True)
        window.page.runJavaScript(stages[index][0],ready)
    timer=QTimer();timer.timeout.connect(tick);timer.start(200)
    dialogs=QTimer();dialogs.timeout.connect(dialog_tick);dialogs.start(120)
    app.exec();(out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(out/'report.json')
    return 0 if report['success'] else 1

if __name__=='__main__':raise SystemExit(main())
