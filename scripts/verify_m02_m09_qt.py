"""Real owned Qt workbench controls, native dialogs, SVG export and reconstruction."""
import json
from pathlib import Path
import time
from uuid import uuid4
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication,QFileDialog,QLineEdit
from ptb_desktop.host import register_scheme,Workbench
from ptb_worker.local_acoustic_files import initialize_local_files
from verify_m02_m09_local import fixtures,DB

ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'output/validation/m02-m09'/('qt-'+uuid4().hex);out.mkdir(parents=True)
    inputs=out/'inputs';inputs.mkdir();fixtures(inputs);cache=out/'cache';cache.mkdir();initialize_local_files(cache)
    register_scheme();app=QApplication(['PhoneticToolbox']);window=Workbench(ROOT/'frontend/dist',test=True,jobs_path=DB,local_files_root=cache,reaper_binary=ROOT/'phonetic_toolbox/core/acoustic/reaper.exe');window.show()
    click=lambda label:"[...document.querySelectorAll('button')].find(b=>b.offsetParent&&b.textContent.trim()==="+json.dumps(label,ensure_ascii=False)+")?.click()"
    select_image="(()=>{const e=document.querySelector('select[aria-label=\"语谱图图片\"]');e.value=[...e.options].find(o=>o.textContent==='spectrogram.png').value;e.dispatchEvent(new Event('change',{bubbles:true}));})()"
    stages=[
        ("document.querySelector('.host-badge')?.textContent==='本地桌面'",click('参数显示'),None),
        ("!!document.querySelector('.m02-page')",click('选择音频目录'),str(inputs)),
        ("document.querySelectorAll('.m02-files button').length===1",click('tone.wav'),None),
        ("!!document.querySelector('.empty-plot')&&document.querySelectorAll('.m02-parameters input').length===4&&document.querySelectorAll('.m02-parameters input:checked').length===0",click('全选可见参数'),None),
        ("document.querySelectorAll('.m02-parameters input:checked').length===3",click('将 3 项分配到图窗'),None),
        ("document.querySelectorAll('.parameter-curve').length===3",click('清空勾选'),None),
        ("document.querySelectorAll('.parameter-figure svg').length===1&&document.querySelectorAll('.m02-parameters input').length===4&&document.querySelectorAll('.shared-plot-area').length===1&&document.querySelectorAll('.parameter-curve').length===3",click('新建图窗'),None),
        ("document.querySelectorAll('.parameter-figure').length===2&&document.querySelector('.empty-plot')?.getBoundingClientRect().height>=280","document.querySelectorAll('.m02-parameters input')[0].click()",None),
        ("document.querySelectorAll('.m02-parameters input:checked').length===1","document.querySelectorAll('.m02-parameters input')[1].click()",None),
        ("document.querySelectorAll('.m02-parameters input:checked').length===2",click('将 2 项分配到图窗'),None),
        ("document.querySelectorAll('.parameter-figure svg').length===2&&[...document.querySelectorAll('.parameter-figure svg')].every(s=>s.getBoundingClientRect().height>=400&&s.querySelectorAll('.shared-plot-area').length===1)&&document.querySelectorAll('.parameter-figure')[1].querySelectorAll('.parameter-curve').length===2","(()=>{const e=document.querySelector('.image-format');e.value='svg';e.dispatchEvent(new Event('change',{bubbles:true}));})();"+click('保存当前图'),str(out/'parameters.svg')),
        ("true",click('保存绘图配置'),None),
        ("true",click('语谱图转音频'),None),
        ("!!document.querySelector('.m09-page')",click('选择图片目录'),str(inputs)),
        ("document.querySelector('select[aria-label=\"语谱图图片\"]')?.options.length===2",select_image,None),
        ("!!document.querySelector('.image-frame img')?.naturalWidth",click('开始重建'),None),
        ("!!document.querySelector('.comparison img') && document.querySelectorAll('.comparison img').length===2 && !!document.querySelector('.m09-results .wave-viewport')",click('保存重建结果'),str(out)),
        ("document.querySelector('.m09-page')?.textContent.includes('已保存 4 个结果文件')",None,None)]
    report={'success':False,'stages':[]};index=0;pending=False;started=time.monotonic();dialog_path=None
    def dialog_tick():
        nonlocal dialog_path
        if not dialog_path:return
        dialog=app.activeModalWidget()
        if isinstance(dialog,QFileDialog):
            edit=dialog.findChild(QLineEdit,'fileNameEdit')
            if edit:
                dialog.setDirectory(str(Path(dialog_path).parent));edit.setText(dialog_path)
                selected=dialog.selectedFiles()
                if selected and Path(selected[0]).absolute()==Path(dialog_path).absolute():dialog_path=None;dialog.accept()
    def finish(success,error=None):
        timer.stop();dialog_timer.stop();report.update(success=success,error=error)
        window.view.grab().save(str(out/('qt-light.png' if success else 'failed.png')))
        def dark(_):
            window.view.grab().save(str(out/'qt-dark.png'));window.resize(1000,740);QTimer.singleShot(500,close)
        def close():
            window.view.grab().save(str(out/'qt-narrow.png'));window.closing=True;window.close()
        window.page.runJavaScript("document.documentElement.dataset.theme='dark'",lambda _:QTimer.singleShot(500,lambda:dark(None)))
    def received(ok):
        nonlocal index,pending,dialog_path
        pending=False
        if not ok:return
        if index==8 and not (out/'parameters.svg').exists():return
        if index==14 and not (out/'reconstructed.wav').exists():return
        if index==8:window.view.grab().save(str(out/'m02-fixed.png'))
        report['stages'].append(index);action=stages[index][1];dialog_path=stages[index][2];index+=1
        if action:window.page.runJavaScript(action)
        else:finish(True)
    def tick():
        nonlocal pending
        if time.monotonic()-started>180:finish(False,'Timeout at '+str(index));return
        if pending or index>=len(stages):return
        pending=True;window.page.runJavaScript(stages[index][0],received)
    timer=QTimer();timer.timeout.connect(tick);timer.start(300)
    dialog_timer=QTimer();dialog_timer.timeout.connect(dialog_tick);dialog_timer.start(200)
    app.exec();report['normal_close']=window.closing
    if report['success']:
        import soundfile as sf
        svg=(out/'parameters.svg').read_text('utf-8');assert '<svg' in svg and '时间' in svg
        metadata=json.loads((out/'reconstruction.ptb.json').read_text('utf-8'));audio,sr=sf.read(out/'reconstructed.wav');assert sr==metadata['sample_rate'] and len(audio)==metadata['samples']
        report['export_readback']=True
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({'report':str(out/'report.json'),**report},ensure_ascii=False))
    window.page.deleteLater();app.processEvents();return 0 if report['success'] else 1


if __name__=='__main__':raise SystemExit(main())
