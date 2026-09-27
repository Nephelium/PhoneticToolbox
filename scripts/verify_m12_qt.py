"""M12 installed wheels and actual Qt scheme/QWebChannel/native folder picker."""
import json
import sys
from pathlib import Path
import time
from uuid import uuid4
from PyQt6.QtCore import QPoint, QPointF, QTimer, Qt
from PyQt6.QtGui import QWheelEvent
from PyQt6.QtWidgets import QApplication, QFileDialog, QLineEdit
from PyQt6.QtTest import QTest
from ptb_desktop.host import register_scheme, Workbench
from ptb_worker.local_acoustic_files import initialize_local_files
from phonetic_core.annotation import parse_document
from m12_ui_bridge import fixtures

ROOT=Path(__file__).resolve().parents[1]


def main(*, bundle=ROOT, out=None, database=None, cache=None, reaper=None, r1=False, r2=False, r3=False):
    r2=r2 or r3;r1=r1 or r2
    out=(out or ROOT/'output/validation/m12-qt'/uuid4().hex).absolute();out.mkdir(parents=True)
    inputs=fixtures(out)
    if r1:
        (inputs/'r1 空白.wav').write_bytes((inputs/'audio_recording.wav').read_bytes())
        (inputs/'r1 空白.TextGrid').write_text('',encoding='utf-8')
    if cache is None:
        cache=out/'cache';cache.mkdir();initialize_local_files(cache)
    register_scheme();app=QApplication(['M12 Qt verification'])
    window=Workbench(bundle/'frontend/dist',test=True,jobs_path=database or ROOT/'output/validation/p06/local-state.sqlite3',local_files_root=cache,
                     reaper_binary=reaper,vocal_resources=bundle/'resources/vocal_tract/native')
    window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen,True);window.resize(1500,1050);window.show()
    click=lambda label:"[...document.querySelectorAll('button')].find(b=>b.offsetParent&&b.textContent.trim()==="+json.dumps(label,ensure_ascii=False)+")?.click()"
    stages=[
      ("document.querySelector('.host-badge')?.textContent==='本地桌面'",click('语音标注对齐'),False),
      ("!!document.querySelector('.annotation-page')",click('选择语料文件夹'),inputs),
      ("document.querySelectorAll('.annotation-file-list button').length==="+str(3 if r1 else 2),"document.querySelector('.annotation-file-list button').click()",False),
      ("!!document.querySelector('.annotation-grid')&&document.querySelector('.annotation-page').getAttribute('aria-busy')==='false'", "(()=>{const c=document.querySelector('.annotation-grid'),b=c.getBoundingClientRect();c.dispatchEvent(new MouseEvent('dblclick',{clientX:b.x+b.width*.15,clientY:b.y+b.height*.2,bubbles:true}));})()",False),
      ("!document.querySelector('[aria-label=\"编辑选中标注文本\"]').disabled", "(()=>{const e=document.querySelector('[aria-label=\"编辑选中标注文本\"]');e.value='Qt 中文 æ';e.dispatchEvent(new Event('input',{bubbles:true}));e.dispatchEvent(new Event('change',{bubbles:true}));})()",False),
      ("document.querySelector('.annotation-page').textContent.includes('保存 TextGrid *')",click('保存 TextGrid *'),False),
      ("document.querySelector('.notice')?.textContent.includes('已保存：')", "(()=>{const e=document.querySelector('[aria-label=\"唇形共同偏移毫秒\"]');e.value='-21';e.dispatchEvent(new Event('change',{bubbles:true}));})()",False),
      ("document.querySelector('.annotation-page').textContent.includes('保存唇偏 *')",click('保存唇偏 *'),False),
      ("document.querySelector('.notice')?.textContent.includes('唇偏已独立保存')",click('下载当前 TextGrid'),out/'qt-download.TextGrid'),
      ("document.querySelector('.notice')?.textContent.includes('已发起下载')",click('下载安全唇形 JSON'),out/'qt-download.lip.json'),
      ("document.querySelector('.notice')?.textContent.includes('已发起下载')",None,False),
    ]
    if r1:
        double=lambda at,ctrl=False:"(()=>{const e=document.querySelector('.annotation-plots .wave-track svg'),r=e.getBoundingClientRect();e.dispatchEvent(new MouseEvent('dblclick',{clientX:r.x+r.width*"+str(at/2)+",clientY:r.y+r.height*.5,ctrlKey:"+str(ctrl).lower()+",bubbles:true}));})()"
        stages[-1]=(stages[-1][0],"[...document.querySelectorAll('.annotation-file-list button')].find(b=>b.textContent.includes('r1 空白.wav')).click()",False)
        stages.extend([
          ("document.querySelector('.notice')?.textContent.includes('请点击')",click('创建标注层'),False),
          ("!!document.querySelector('[aria-label=\"新建音节层名\"]')","(()=>{for(const [label,text] of [['新建音节层名','音节'],['新建音素层名','音素']]){const e=document.querySelector('[aria-label=\"'+label+'\"]');e.value=text;e.dispatchEvent(new Event('input',{bubbles:true}));}})()",False),
          ("document.querySelector('[aria-label=\"新建音素层名\"]')?.value==='音素'",click('创建层'),False),
          ("document.querySelector('.notice')?.textContent.includes('已创建')",click('粘贴词表'),False),
          ("!!document.querySelector('[aria-label=\"拼音词表\"]')","(()=>{const e=document.querySelector('[aria-label=\"拼音词表\"]');e.value='zhe4 shi4 shang4';e.dispatchEvent(new Event('input',{bubbles:true}));})()",False),
          ("document.querySelector('[aria-label=\"拼音词表\"]')?.value==='zhe4 shi4 shang4'",click('应用词表'),False),
          ("document.querySelector('.notice')?.textContent.includes('已应用词表')","document.querySelector('.sequence-toggle input[type=checkbox]').click()",False),
          ("document.querySelector('.sequence-toggle input[type=checkbox]')?.checked",double(.2),False),
          ("!!document.querySelector('.sequence-hint strong')",double(.7),False),
          ("document.querySelector('[aria-label=\"下一音节\"]')?.value==='1'",double(.35),False),
          ("document.querySelector('.notice')?.textContent.includes('已按')",double(1.1,True),False),
          ("document.querySelector('[aria-label=\"下一音节\"]')?.value==='2'","document.body.dispatchEvent(new KeyboardEvent('keydown',{key:'ArrowRight',bubbles:true}))",False),
          ("document.querySelector('.notice')?.textContent.includes('右移 1 ms')",click('保存 TextGrid *'),False),
          ("document.querySelector('.notice')?.textContent.includes('已保存：r1 空白_自动保存.TextGrid')",None,False),
        ])
    if r2:
        stages[-1]=(stages[-1][0],"document.querySelector('.annotation-grid').scrollIntoView({block:'center'})",False)
        stages.extend([
          ("document.querySelector('[aria-label=\"词层名\"]')?.tagName==='SELECT'&&document.querySelector('[aria-label=\"词层名\"]').value==='音节'&&document.querySelector('[aria-label=\"音素层名\"]').value==='音素'","document.querySelector('.annotation-plots').dataset.r2Names='verified'",False),
          ("(()=>{const q=s=>document.querySelector(s)?.getBoundingClientRect(),w=q('.wave-track svg'),o=q('.overview-controls'),a=q('.spectrum-wrap'),g=q('.annotation-grid');return w&&o&&a&&g&&o.bottom<=w.top&&Math.abs(w.x-a.x)<2&&Math.abs(w.width-a.width)<2&&Math.abs(w.x-g.x)<2&&document.querySelectorAll('main').length===1&&document.documentElement.scrollHeight<=innerHeight+1&&!document.querySelector('.annotation-plots .transport');})()","document.querySelector('.annotation-plots').dataset.r2Layout='verified'",False),
          ("(()=>{const e=document.querySelector('[aria-label=\"语谱图选区\"]'),label=document.querySelector('.label-editor .mono');if(!e||!label)return false;const [a,b]=label.textContent.split('–').map(parseFloat);return a>0&&b>a&&Number(e.dataset.start)===a&&Number(e.dataset.end)===b&&Number(document.querySelector('.amplitude-axis').dataset.limit)>0&&document.querySelectorAll('.annotation-boundary').length>0;})()",None,False),
        ])
    if r3:
        stages[-1]=(stages[-1][0],click('设置'),False)
        stages.extend([
          ("!!document.querySelector('[aria-label=\"放大页面\"]')","document.querySelector('[aria-label=\"放大页面\"]').click()",False),
          ("document.documentElement.style.zoom==='1.1'","window.__r3Wheel=0;window.addEventListener('wheel',e=>{window.__r3Wheel++;window.__r3Prevented=e.defaultPrevented;window.__r3Event={dx:e.deltaX,dy:e.deltaY,shift:e.shiftKey,ctrl:e.ctrlKey};},{capture:true,passive:false})",False),
          ("window.__r3Wheel===0","__native_ctrl_wheel__",False),
          ("window.__r3Wheel>0&&window.__r3Prevented&&document.documentElement.style.zoom==='1.1'",click('恢复 100%'),False),
          ("document.documentElement.style.zoom==='1'","document.querySelector('[aria-label=\"关闭对话框\"]').click()",False),
          ("!document.querySelector('[role=dialog]')","document.querySelector('.sequence-toggle input[type=checkbox]').click()",False),
          ("!document.querySelector('.sequence-toggle input[type=checkbox]').checked",double(1.3),False),
          ("!!document.querySelector('.sequence-hint strong')",double(1.5),False),
          ("!document.querySelector('.sequence-hint strong')&&!document.querySelector('[aria-label=\"编辑选中标注文本\"]').disabled","(()=>{const e=document.querySelector('[aria-label=\"编辑选中标注文本\"]');e.value='R3 新增';e.dispatchEvent(new Event('input',{bubbles:true}));e.dispatchEvent(new Event('change',{bubbles:true}));})()",False),
          ("document.querySelector('.annotation-page').textContent.includes('保存 TextGrid *')",click('保存 TextGrid *'),False),
          ("document.querySelector('.notice')?.textContent.includes('已保存：r1 空白_自动保存.TextGrid')&&!([...document.querySelectorAll('button')].some(b=>['清空文本','删除边界'].includes(b.textContent.trim())))",None,False),
        ])
        stages[-1]=(stages[-1][0],"(()=>{const e=document.querySelector('[aria-label=\"标注可视时长\"]');e.value='.08';e.dispatchEvent(new Event('input',{bubbles:true}));e.dispatchEvent(new Event('change',{bubbles:true}));})()",False)
        stages.extend([
          ("document.querySelector('.wave-line')?.dataset.mode==='samples'","(()=>{const p=document.querySelector('[aria-label=\"平移波形时间窗\"]');p.value='.5';p.dispatchEvent(new Event('input',{bubbles:true}));document.querySelector('.wave-track svg').scrollIntoView({block:'center'});window.__r3PanStart=.5;})()",False),
          ("Number(document.querySelector('.wave-track svg').dataset.start)===window.__r3PanStart","__native_shift_wheel__",False),
          ("(()=>{const w=document.querySelector('.wave-track svg'),t=document.querySelector('.annotation-tracks'),c=document.querySelector('[aria-label=\"标注语谱图与唇形曲线\"]'),e=window.__r3Event;return e?.shift&&Number(w.dataset.start)!==window.__r3PanStart&&Math.abs(Number(w.dataset.start)-window.__r3PanStart-.08*.0015*(e.dy||e.dx))<1e-8&&w.dataset.start===t.dataset.start&&w.dataset.end===t.dataset.end&&Math.abs(w.getBoundingClientRect().height-c.getBoundingClientRect().height)<1;})()",None,False),
        ])
        phone_double="(()=>{const c=document.querySelector('.annotation-grid'),r=c.getBoundingClientRect();c.dispatchEvent(new MouseEvent('dblclick',{clientX:r.x+r.width*.78/2,clientY:r.y+r.height*.73,bubbles:true}));})()"
        stages[-1]=(stages[-1][0],"(()=>{const e=document.querySelector('[aria-label=\"标注可视时长\"]');e.value='2';e.dispatchEvent(new Event('input',{bubbles:true}));e.dispatchEvent(new Event('change',{bubbles:true}));})()",False)
        stages.extend([
          ("document.querySelector('[aria-label=\"标注可视时长\"]').value==='2.000'",phone_double,False),
          ("document.querySelector('.notice')?.textContent.includes('首个边界 0.78')||document.querySelector('.notice')?.textContent.includes('首个边界 0.77')",click('保存 TextGrid *'),False),
          ("document.querySelector('.notice')?.textContent.includes('已保存：')",click('撤销'),False),
          ("document.querySelector('.annotation-page').textContent.includes('保存 TextGrid *')","(()=>{const e=document.querySelector('[aria-label=\"音素首个切分点\"]');e.value='equal';e.dispatchEvent(new Event('change',{bubbles:true}));})();"+phone_double,False),
          ("document.querySelector('.notice')?.textContent.includes('已按等分切分')",click('保存 TextGrid *'),False),
          ("document.querySelector('.notice')?.textContent.includes('已保存：')","document.querySelector('[aria-label=\"标注语谱图与唇形曲线\"]').scrollIntoView({block:'center'})",False),
          ("document.querySelector('[aria-label=\"标注语谱图与唇形曲线\"]').getBoundingClientRect().top>=0","__spectrum_drag__",False),
          ("(()=>{const a=document.querySelector('[aria-label=\"语谱图选区\"]'),b=document.querySelector('[aria-label=\"标注层选区\"]');return a&&b&&Math.abs(Number(a.dataset.start)-.5)<.01&&Math.abs(Number(a.dataset.end)-1.2)<.01&&a.dataset.start===b.dataset.start&&a.dataset.end===b.dataset.end&&document.querySelector('[aria-label=\"编辑选中标注文本\"]').disabled;})()",None,False),
        ])
    index=0;pending=False;dialog_pending=False;started=time.monotonic();report={'success':False,'stages':[],
        'frozen':bool(getattr(sys,'frozen',False)),'executable':sys.executable,'working_directory':str(Path.cwd())}
    def finish(success,error=None):
        timer.stop();dialog_timer.stop();report.update(success=success,error=error)
        if success:
            try:
                doc=parse_document((inputs/'audio_recording_自动保存.TextGrid').read_text('utf-8'))
                assert any(i['text']=='Qt 中文 æ' for i in doc['tiers'][0]['intervals'])
                import pickle
                assert pickle.loads((inputs/'audio_recording.pkl').read_bytes())['metadata']['lip_manual_offset']==-.021
                report['actual_output_readback']=True
                assert (out/'qt-download.TextGrid').read_bytes()==(inputs/'audio_recording_自动保存.TextGrid').read_bytes()
                assert json.loads((out/'qt-download.lip.json').read_text('utf-8'))['data']['metadata']['lip_manual_offset']==-.021
                report['native_download_readback']=True
                if r1:
                    sequence=parse_document((inputs/'r1 空白_自动保存.TextGrid').read_text('utf-8'))
                    words=[i for i in sequence['tiers'][0]['intervals'] if i['text']]
                    phones=[i for i in sequence['tiers'][1]['intervals'] if i['text']]
                    assert [i['text'] for i in words]==(['zhe4','shi4','R3 新增'] if r3 else ['zhe4','shi4'])
                    assert [i['text'] for i in phones]==(['zh','e4','sh','ii4','r3 新增'] if r3 else ['zh','e4','shi4'])
                    assert phones[0]['xmin']==words[0]['xmin'] and phones[1]['xmax']==words[0]['xmax']
                    assert abs(words[1]['xmin']-words[0]['xmax']-.001)<1.1e-6
                    assert (inputs/'r1 空白.TextGrid').read_bytes()==b''
                    report['r1_sequence_and_1ms_save_readback']=True
                    assert [t['name'] for t in sequence['tiers']]==['音节','音素']
                    if r2:report['r2_named_layers_layout_and_linked_selection']=True
                    if r3:
                        assert window.view.zoomFactor()==1
                        assert 1.29<words[2]['xmin']<1.31 and 1.49<words[2]['xmax']<1.51
                        assert (words[2]['xmin'],words[2]['xmax'])==(phones[4]['xmin'],phones[4]['xmax'])
                        assert phones[2]['xmax']==round((words[1]['xmin']+words[1]['xmax'])/2,6)
                        report['r3_native_ctrl_wheel_blocked_and_settings_scale']=True
                        report['r3_manual_annotation_and_chinese_filename']=True
                        report['r3_native_shift_pan_equal_height_and_continuous_wave']=True
                        report['r3_native_spectrum_selection_and_phone_split_modes']=True
            except Exception as exc:
                report.update(success=False,error=repr(exc))
        def dark(_):
            QTimer.singleShot(400,close)
        def light(_):
            QTimer.singleShot(400,take_light)
        def take_light():
            window.view.grab().save(str(out/'qt-light.png'))
            window.page.runJavaScript("document.documentElement.dataset.theme='dark'",dark)
        def close():
            window.view.grab().save(str(out/'qt-dark.png'));window.closing=True;window.close()
        def diagnostics(value):
            report['view_diagnostics']=value
            window.page.runJavaScript("document.documentElement.dataset.theme='light'",light)
        window.page.runJavaScript("(()=>{const w=document.querySelector('.wave-track svg'),c=document.querySelector('[aria-label=\"标注语谱图与唇形曲线\"]');return {wheel:window.__r3Event,start:w?.dataset.start,end:w?.dataset.end,waveHeight:w?.getBoundingClientRect().height,specHeight:c?.getBoundingClientRect().height};})()",diagnostics)
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
                dialog_pending=pick
                if action=='__native_ctrl_wheel__':
                    target=window.view.focusProxy() or window.view;point=QPoint(400,30)
                    QApplication.sendEvent(target,QWheelEvent(QPointF(point),QPointF(target.mapToGlobal(point)),QPoint(),QPoint(0,120),Qt.MouseButton.NoButton,Qt.KeyboardModifier.ControlModifier,Qt.ScrollPhase.NoScrollPhase,False))
                elif action=='__native_shift_wheel__':
                    def pan_at(coords):
                        target=window.view.focusProxy() or window.view;point=QPoint(round(coords[0]),round(coords[1]))
                        QApplication.sendEvent(target,QWheelEvent(QPointF(point),QPointF(target.mapToGlobal(point)),QPoint(),QPoint(0,-120),Qt.MouseButton.NoButton,Qt.KeyboardModifier.ShiftModifier,Qt.ScrollPhase.NoScrollPhase,False))
                    window.page.runJavaScript("(()=>{const r=document.querySelector('.wave-track svg').getBoundingClientRect();return [r.x+r.width/2,r.y+r.height/2];})()",pan_at)
                elif action=='__spectrum_drag__':
                    def select_at(coords):
                        target=window.view.focusProxy() or window.view;a=QPoint(round(coords[0]),round(coords[2]));b=QPoint(round(coords[1]),round(coords[2]))
                        QTest.mousePress(target,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,a)
                        QTest.mouseMove(target,b,30);QTest.mouseRelease(target,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,b)
                    window.page.runJavaScript("(()=>{const r=document.querySelector('[aria-label=\"标注语谱图与唇形曲线\"]').getBoundingClientRect();return [r.x+r.width*.25,r.x+r.width*.6,r.y+r.height*.5];})()",select_at)
                else:window.page.runJavaScript(action)
            if index==len(stages):finish(True)
        window.page.runJavaScript(stages[index][0],ready)
    timer=QTimer();timer.timeout.connect(tick);timer.start(160)
    dialog_timer=QTimer();dialog_timer.timeout.connect(dialog_tick);dialog_timer.start(120)
    app.exec();(out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(out/'report.json')
    if not report['success']:raise SystemExit(1)


if __name__=='__main__':main()
