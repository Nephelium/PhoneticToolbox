"""M12 installed wheels and actual Qt scheme/QWebChannel/native folder picker."""
import json
from pathlib import Path
import time
from uuid import uuid4
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import QApplication, QFileDialog, QLineEdit
from ptb_desktop.host import register_scheme, Workbench
from ptb_worker.local_acoustic_files import initialize_local_files
from phonetic_core.annotation import parse_document
from m12_ui_bridge import fixtures

ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'output/validation/m12-qt'/uuid4().hex;out.mkdir(parents=True)
    inputs=fixtures(out);cache=out/'cache';cache.mkdir();initialize_local_files(cache)
    register_scheme();app=QApplication(['M12 Qt verification'])
    window=Workbench(ROOT/'frontend/dist',test=True,jobs_path=ROOT/'output/validation/p06/local-state.sqlite3',local_files_root=cache)
    window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen,True);window.resize(1500,1050);window.show()
    click=lambda label:"[...document.querySelectorAll('button')].find(b=>b.offsetParent&&b.textContent.trim()==="+json.dumps(label,ensure_ascii=False)+")?.click()"
    stages=[
      ("document.querySelector('.host-badge')?.textContent==='本地桌面'",click('语音标注对齐'),False),
      ("!!document.querySelector('.annotation-page')",click('选择语料文件夹'),inputs),
      ("document.querySelectorAll('.annotation-file-list button').length===2","document.querySelector('.annotation-file-list button').click()",False),
      ("!!document.querySelector('.annotation-grid')&&document.querySelector('.annotation-page').getAttribute('aria-busy')==='false'", "(()=>{const c=document.querySelector('.annotation-grid'),b=c.getBoundingClientRect();c.dispatchEvent(new MouseEvent('dblclick',{clientX:b.x+b.width*.15,clientY:b.y+b.height*.2,bubbles:true}));})()",False),
      ("!document.querySelector('[aria-label=\"编辑选中标注文本\"]').disabled", "(()=>{const e=document.querySelector('[aria-label=\"编辑选中标注文本\"]');e.value='Qt 中文 æ';e.dispatchEvent(new Event('input',{bubbles:true}));e.dispatchEvent(new Event('change',{bubbles:true}));})()",False),
      ("document.querySelector('.annotation-page').textContent.includes('保存 TextGrid *')",click('保存 TextGrid *'),False),
      ("document.querySelector('.notice')?.textContent.includes('已保存：')", "(()=>{const e=document.querySelector('[aria-label=\"唇形共同偏移毫秒\"]');e.value='-21';e.dispatchEvent(new Event('change',{bubbles:true}));})()",False),
      ("document.querySelector('.annotation-page').textContent.includes('保存唇偏 *')",click('保存唇偏 *'),False),
      ("document.querySelector('.notice')?.textContent.includes('唇偏已独立保存')",click('下载当前 TextGrid'),out/'qt-download.TextGrid'),
      ("document.querySelector('.notice')?.textContent.includes('已发起下载')",click('下载安全唇形 JSON'),out/'qt-download.lip.json'),
      ("document.querySelector('.notice')?.textContent.includes('已发起下载')",None,False),
    ]
    index=0;pending=False;dialog_pending=False;started=time.monotonic();report={'success':False,'stages':[]}
    def finish(success,error=None):
        timer.stop();dialog_timer.stop();report.update(success=success,error=error)
        if success:
            doc=parse_document((inputs/'audio_recording_webedit.TextGrid').read_text('utf-8'))
            assert any(i['text']=='Qt 中文 æ' for i in doc['tiers'][0]['intervals'])
            import pickle
            assert pickle.loads((inputs/'audio_recording.pkl').read_bytes())['metadata']['lip_manual_offset']==-.021
            report['actual_output_readback']=True
            assert (out/'qt-download.TextGrid').read_bytes()==(inputs/'audio_recording_webedit.TextGrid').read_bytes()
            assert json.loads((out/'qt-download.lip.json').read_text('utf-8'))['data']['metadata']['lip_manual_offset']==-.021
            report['native_download_readback']=True
        def dark(_):
            QTimer.singleShot(400,close)
        def light(_):
            QTimer.singleShot(400,take_light)
        def take_light():
            window.view.grab().save(str(out/'qt-light.png'))
            window.page.runJavaScript("document.documentElement.dataset.theme='dark'",dark)
        def close():
            window.view.grab().save(str(out/'qt-dark.png'));window.closing=True;window.close()
        window.page.runJavaScript("document.documentElement.dataset.theme='light'",light)
    def dialog_tick():
        nonlocal dialog_pending
        if not dialog_pending:return
        dialog=app.activeModalWidget()
        if isinstance(dialog,QFileDialog):
            edit=dialog.findChild(QLineEdit,'fileNameEdit')
            if edit:
                selected=dialog_pending
                dialog.setDirectory(str(selected.parent));edit.setText(str(selected))
                if dialog.selectedFiles() and Path(dialog.selectedFiles()[0]).absolute()==selected.absolute():dialog_pending=False;dialog.accept()
    def tick():
        nonlocal pending,index,dialog_pending
        if pending:return
        if time.monotonic()-started>100:finish(False,'stage_timeout_'+str(index));return
        if index==9 and not (out/'qt-download.TextGrid').is_file():return
        if index==10 and not (out/'qt-download.lip.json').is_file():return
        pending=True
        def ready(value):
            nonlocal pending,index,dialog_pending
            pending=False
            if not value:return
            condition,action,pick=stages[index];report['stages'].append(index+1);index+=1
            if action:
                dialog_pending=pick;window.page.runJavaScript(action)
            if index==len(stages):finish(True)
        window.page.runJavaScript(stages[index][0],ready)
    timer=QTimer();timer.timeout.connect(tick);timer.start(160)
    dialog_timer=QTimer();dialog_timer.timeout.connect(dialog_tick);dialog_timer.start(120)
    app.exec();(out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(out/'report.json')
    if not report['success']:raise SystemExit(1)


if __name__=='__main__':main()
