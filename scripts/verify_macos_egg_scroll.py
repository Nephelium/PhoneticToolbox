"""Visible macOS EGG scroll benchmark through the real desktop and local service.

The only injected input is the owned directory picker; scientific computation,
QWebChannel payloads and UI wheel handlers are real. No device claim is made.
"""
from pathlib import Path
import os,sys,json,time,statistics

def main(output):
    S=Path(getattr(sys,'_MEIPASS',Path(__file__).resolve().parents[1]))
    from PyQt6.QtCore import QTimer,Qt
    from PyQt6.QtWidgets import QApplication,QFileDialog
    from ptb_desktop.web_assets import register_scheme
    from ptb_desktop.host import Workbench
    from ptb_worker.local_workspace import prepare_workspace
    import numpy as np,soundfile as sf
    out=Path(output);out.mkdir(parents=True,exist_ok=False);inputs=out/'scratch';inputs.mkdir()
    sr=44100;n=round(77.249*sr);t=np.arange(n)/sr
    path=inputs/'synthetic-77.249s.wav'
    sf.write(path,np.column_stack([.18*np.sin(2*np.pi*150*t)+.04*np.sin(2*np.pi*300*t),.4+.15*np.sin(2*np.pi*150*t+.5)]),sr,subtype='FLOAT')
    db,cache=prepare_workspace(out/'scratch'/'workspace',S/'backend/migrations')
    register_scheme();QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    app=QApplication(sys.argv[:1]);app.setApplicationName('PTB-EGG-Scroll-QA');app.setQuitOnLastWindowClosed(False)
    w=Workbench(S/'frontend/dist',test=True,vocal_profile=out/'scratch'/'vocal-profile',jobs_path=db,local_files_root=cache,reaper_binary=S/'resources/research/reaper',vocal_resources=S/'resources/vocal_tract/native')
    original_dialog=QFileDialog.getExistingDirectory
    QFileDialog.getExistingDirectory=lambda *args,**kw:str(inputs)
    requests=[];gestures=[];report={'success':False,'frozen':bool(getattr(sys,'frozen',False)),'scope':'visible macOS Qt WebEngine; actual desktop task bridge and HTTP child; test directory picker; dispatched wheel events; two animation frames after ready','requests':requests,'gestures':gestures,'input':{'duration_s':n/sr,'sample_rate':sr,'channels':2,'subtype':'FLOAT'}}
    invoke=w.bridge.tasks.invoke
    def observed(body):
        started=time.perf_counter();value=invoke(body)
        if body.get('op','').startswith('egg_preview'):
            requests.append({'op':body['op'],'seconds':time.perf_counter()-started,'response_bytes':len(json.dumps(value).encode()),'audio_base64_bytes':len(value.get('audio_base64') or '')})
        return value
    w.bridge.tasks.invoke=observed
    w.show();w.raise_();state={'step':'navigate','started':time.monotonic(),'request_count':0,'deadline':time.monotonic()+150,'gesture':0}
    def save():(out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    def finish(error=None):
        timer.stop();QFileDialog.getExistingDirectory=original_dialog
        if error:report['error']=error
        save();w.grab().save(str(out/'egg-scroll.png'))
        w.accept_close();w.page.deleteLater();w.view.deleteLater();QTimer.singleShot(200,app.quit)
        print(json.dumps({'success':report['success'],'gestures':gestures,'error':error}),flush=True)
    def poll():
        if time.monotonic()>state['deadline']:finish('UI benchmark timed out at '+state['step']);return
        def got(v):
            if not v:return
            step=state['step']
            if v.get('errors'):finish(str(v['errors']));return
            if step=='navigate' and v.get('nav'):
                state['step']='module';w.page.runJavaScript("[...document.querySelectorAll('.nav-item')].find(e=>e.title==='EGG 信号分析').click()")
            elif step=='module' and v.get('egg'):
                state['step']='files';w.page.runJavaScript("[...document.querySelectorAll('.egg-page button')].find(e=>e.textContent.trim()==='打开音频目录').click()")
            elif step=='files' and v.get('files',0)>1:
                state['step']='initial';state['started']=time.monotonic()
                w.page.runJavaScript("(()=>{let e=document.querySelector('select[aria-label=\"EGG 音频文件\"]');e.value=e.options[1].value;e.dispatchEvent(new Event('change',{bubbles:true}));})()")
            elif step=='initial' and v.get('ready'):
                report['initial_ui_seconds']=time.monotonic()-state['started'];state['step']='gesture'
                report['initial_science_child_pids']=[p.pid for p in __import__('psutil').Process(w.service.process.pid).children(recursive=True)]
            elif step=='gesture':
                if state['gesture']>=11:
                    zoom_requests=[r for r in requests[state['initial_request_count']:] if r['op']=='egg_preview_update']
                    report['summary']={'zoom_updates':len(zoom_requests),'zoom_audio_base64_bytes':sum(r['audio_base64_bytes'] for r in zoom_requests),'median_ui_seconds':statistics.median(g['ui_seconds'] for g in gestures),'max_ui_seconds':max(g['ui_seconds'] for g in gestures)}
                    report['final_science_child_pids']=[p.pid for p in __import__('psutil').Process(w.service.process.pid).children(recursive=True)]
                    report['success']=bool(len(gestures)==11 and zoom_requests and report['summary']['zoom_audio_base64_bytes']==0 and report['initial_science_child_pids']==report['final_science_child_pids'])
                    finish();return
                if not state['gesture']:state['initial_request_count']=len(requests)
                state['step']='wait';state['request_count']=len(requests);state['started']=time.monotonic()
                w.page.runJavaScript("(()=>{window.__qaPaintDone=false;const e=document.querySelector('.cq-pane svg');const r=e.getBoundingClientRect();e.dispatchEvent(new WheelEvent('wheel',{deltaY:-120,bubbles:true,cancelable:true,clientX:r.left+r.width/2,clientY:r.top+r.height/2}));})()")
            elif step=='wait' and len(requests)>state['request_count'] and v.get('ready'):
                state['step']='paint';w.page.runJavaScript('requestAnimationFrame(()=>requestAnimationFrame(()=>window.__qaPaintDone=true))')
            elif step=='paint' and v.get('painted'):
                gestures.append({'index':state['gesture'],'ui_seconds':time.monotonic()-state['started'],'updates':len(requests)-state['request_count']})
                state['gesture']+=1;state['step']='gesture';save()
        w.page.runJavaScript("({nav:[...document.querySelectorAll('.nav-item')].some(e=>e.title==='EGG 信号分析'),egg:!!document.querySelector('.egg-page'),files:document.querySelector('select[aria-label=\"EGG 音频文件\"]')?.options.length||0,ready:document.querySelector('.egg-live-status')?.textContent.trim()==='实时预览',errors:[...document.querySelectorAll('.egg-page [role=alert]')].map(e=>e.textContent).join(''),painted:!!window.__qaPaintDone})",got)
    timer=QTimer();timer.timeout.connect(poll);timer.start(50)
    app.exec()
    report['input']['bytes']=path.stat().st_size;report['input']['sha256']=__import__('hashlib').sha256(path.read_bytes()).hexdigest()
    assert path.resolve().is_relative_to(inputs.resolve()) and not path.is_symlink()
    path.unlink();report['scratch_cleanup']={'audio_removed':True,'workspace':'retained under owned scratch'};save()
    return 0 if report['success'] else 1
