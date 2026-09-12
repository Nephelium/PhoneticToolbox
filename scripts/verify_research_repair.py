"""Actual frozen UI/worker regression, invoked by the owned repair EXE only."""
import hashlib
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import time
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import QApplication, QFileDialog, QLineEdit, QWidget
from PyQt6.QtTest import QTest
from ptb_desktop.host import register_scheme, Workbench
from ptb_desktop.capture import capture_hidden
from verify_m02_m09_local import fixtures


def verify(bundle, database, files, reaper, out):
    out = Path(out);out.mkdir(parents=True, exist_ok=True)
    inputs = out / 'inputs';inputs.mkdir();fixtures(inputs)
    (inputs / 'tone.TextGrid').write_text('''File type = "ooTextFile"
Object class = "TextGrid"
0
1
<exists>
1
"IntervalTier"
"word"
0
1
3
0
0.1
""
0.1
0.3
"ɑ̃˥"
0.3
1
"long interval"
''', encoding='utf-8')
    with closing(sqlite3.connect(inputs / 'tone.ptb.sqlite')) as connection:
        with connection:
            connection.execute('CREATE TABLE params(Time_s REAL,pF0 REAL,rF0 REAL,Intensity REAL,TextGrid TEXT)')
            connection.executemany('INSERT INTO params VALUES(?,?,?,?,?)', [(i / 100, 200+i/10, 201, 60, 'ɑ̃˥') for i in range(100)])
    originals = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.iterdir()}
    register_scheme();app = QApplication(['Research repair verification']);app.setQuitOnLastWindowClosed(False)
    # Cover this screen with owned colors before capture; no personal desktop
    # image is saved. Verify disappearance using a tiny synthetic center crop.
    backdrop = QWidget();backdrop.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
    backdrop.setStyleSheet('background:rgb(12,34,56)');backdrop.setGeometry(app.primaryScreen().geometry());backdrop.show();backdrop.raise_();QTest.qWait(500)
    marker = QWidget();marker.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint);marker.setStyleSheet('background:red');marker.resize(200,200)
    center = app.primaryScreen().geometry().center();marker.move(center.x()-100,center.y()-100);marker.show();marker.raise_();QTest.qWait(300)
    shot = capture_hidden(marker).toImage();color = shot.pixelColor(shot.width()//2,shot.height()//2)
    assert (color.red(),color.green(),color.blue()) == (12,34,56), color.name()
    assert marker.isVisible()
    shot.copy(shot.width()//2-2,shot.height()//2-2,4,4).save(str(out/'capture-hidden.png'))
    marker.close();backdrop.close()
    window = Workbench(bundle/'frontend/dist', test=True, jobs_path=database, local_files_root=files,
                       reaper_binary=reaper, vocal_resources=bundle/'resources/vocal_tract/native', vocal_profile=out/'vocal-profile')
    window.show()
    click = lambda text: "[...document.querySelectorAll('button')].find(b=>b.offsetParent&&b.textContent.trim()==="+json.dumps(text,ensure_ascii=False)+")?.click()"
    choose_image = "(()=>{const e=document.querySelector('select[aria-label=\"语谱图图片\"]');e.value=[...e.options].find(o=>o.textContent==='spectrogram.png').value;e.dispatchEvent(new Event('change',{bubbles:true}));})()"
    stages = [
        ('document.querySelector(".host-badge")?.textContent==="本地桌面"',click('参数估计'),None),
        ('!!document.querySelector(".m01-page")',click('选择音频目录'),str(inputs)),
        ('document.querySelectorAll(".m01-file-entry").length===1','document.querySelector(".m01-file-entry button").click()',None),
        ('document.querySelectorAll(".textgrid-interval").length===3','(()=>{const e=[...document.querySelectorAll("label")].find(e=>e.textContent.includes("显示语谱图（Praat）"));e?.querySelector("input")?.click()})()',None),
        ('!!document.querySelector(".spectrogram-canvas canvas")',click('开始全列表分析'),None),
        ('!![...document.querySelectorAll("button")].find(b=>b.textContent.includes("保存已完成结果")&&!b.disabled)',click('参数显示'),None),
        ('!!document.querySelector(".m02-page")',click('选择音频目录'),str(inputs)),
        ('document.querySelectorAll(".m02-files button").length===1',click('tone.wav'),None),
        ('document.querySelectorAll(".parameter-curve").length===3',click('语谱图转音频'),None),
        ('!!document.querySelector(".m09-page")',click('选择图片目录'),str(inputs)),
        ('[...document.querySelectorAll(`select[aria-label="语谱图图片"] option`)].some(o=>o.textContent==="spectrogram.png")',choose_image,None),
        ('!!document.querySelector(".image-frame img")?.naturalWidth',click('开始重建'),None),
        ('document.querySelectorAll(".comparison img").length===2',click('保存重建结果'),str(out)),
        ('document.querySelector(".m09-page")?.textContent.includes("已保存 4 个结果文件")',None,None),
    ]
    report = {'success':False,'capture_hidden':True,'stages':[]};index=0;pending=False;dialog_path=None;started=time.monotonic()
    def finish(ok,error=None):
        timer.stop();dialogs.stop();report.update(success=ok,error=error)
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        window.view.grab().save(str(out/'final.png'));window.closing=True;window.close();app.quit()
    def received(ok):
        nonlocal index,pending,dialog_path
        pending=False
        if not ok:return
        print('Passed UI stage',index,flush=True)
        if index in (4,8):window.view.grab().save(str(out/f'stage-{index}.png'))
        report['stages'].append(index);action=stages[index][1];dialog_path=stages[index][2];index+=1
        if action:window.page.runJavaScript(action)
        else:finish(True)
    def tick():
        nonlocal pending
        if time.monotonic()-started>180:finish(False,'Timeout stage '+str(index));return
        if pending or index>=len(stages):return
        pending=True;window.page.runJavaScript(stages[index][0],received)
    def dialog_tick():
        nonlocal dialog_path
        if not dialog_path:return
        dialog=app.activeModalWidget()
        if isinstance(dialog,QFileDialog):
            edit=dialog.findChild(QLineEdit,'fileNameEdit')
            if edit:
                dialog.setDirectory(str(Path(dialog_path).parent));edit.setText(dialog_path)
                if dialog.selectedFiles() and Path(dialog.selectedFiles()[0]).absolute()==Path(dialog_path).absolute():dialog_path=None;dialog.accept()
    timer=QTimer();timer.timeout.connect(tick);timer.start(300)
    dialogs=QTimer();dialogs.timeout.connect(dialog_tick);dialogs.start(200)
    app.exec()
    if report['success']:
        import soundfile as sf
        metadata=json.loads((out/'reconstruction.ptb.json').read_text('utf-8'));audio,sr=sf.read(out/'reconstructed.wav')
        assert len(audio)==metadata['samples'] and sr==metadata['sample_rate']
        with closing(sqlite3.connect(database)) as connection:
            jobs=connection.execute('SELECT state,result_manifest,snapshot FROM jobs').fetchall()
        assert any(state=='succeeded' and json.loads(snapshot)['operation']=='acoustic_analysis' for state,_,snapshot in jobs)
        assert all(state=='succeeded' for state,_,_ in jobs)
        report['jobs_succeeded']=len(jobs);report['wav_readback']=True
    assert originals == {name:hashlib.sha256((inputs/name).read_bytes()).hexdigest() for name in originals}
    report['added_outputs']=sorted(p.name for p in inputs.iterdir() if p.name not in originals)
    report['inputs_unchanged']=True;report['normal_close']=window.closing
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(out/'report.json');window.page.deleteLater();app.processEvents()
    return 0 if report['success'] else 1
