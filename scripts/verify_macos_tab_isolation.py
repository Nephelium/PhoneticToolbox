"""Actual Qt module-tab visibility and persistent iframe regression."""
from pathlib import Path
import json,sys,os,time

def main(output):
    from PyQt6.QtCore import QTimer,Qt
    from PyQt6.QtWidgets import QApplication
    from ptb_desktop.web_assets import register_scheme
    from ptb_desktop.host import Workbench
    from ptb_worker.local_workspace import prepare_workspace
    root=Path(getattr(sys,'_MEIPASS',Path(__file__).resolve().parents[1]))
    # Outside-source diagnostic invocation supplies the actual source explicitly.
    if not (root/'frontend/dist').is_dir():root=Path(os.environ['PTB_TAB_QA_SOURCE'])
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    db,cache=prepare_workspace(output/'scratch'/'workspace',root/'backend/migrations')
    register_scheme();QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    app=QApplication(sys.argv[:1]);app.setApplicationName('PTB-Tab-Isolation-QA');app.setQuitOnLastWindowClosed(False)
    window=Workbench(root/'frontend/dist',test=True,vocal_profile=output/'scratch'/'vocal-profile',jobs_path=db,local_files_root=cache,reaper_binary=root/'resources/research/reaper',vocal_resources=root/'resources/vocal_tract/native')
    window.show();window.raise_()
    targets=['音系归纳','国际音标表Plus','感知实验','MFA 自动标注']
    report={'success':False,'frozen':bool(getattr(sys,'frozen',False)),'scope':'visible Qt; actual native M10 iframe; tab changes and scrolling to container ends; preserved iframe instance and section selection','cases':[]}
    state={'phase':'open','index':0,'deadline':time.monotonic()+150,'after':0.,'waiting':False}
    code=r'''(()=>{const f=[...document.querySelectorAll('iframe')].find(f=>f.title==='生理参数合成');let quality=null;try{quality=[...(f?.contentDocument?.querySelectorAll('select')??[])].find(s=>[...s.options].some(o=>o.value==='129'))?.value??null;}catch{}return {nav:!![...document.querySelectorAll('.nav-item')].find(e=>e.title==='生理参数合成'),iframe:!!f,visible:!!f&&f.getClientRects().length>0&&f.offsetWidth>0&&f.offsetHeight>0,rect:f?{top:f.getBoundingClientRect().top,bottom:f.getBoundingClientRect().bottom,height:f.getBoundingClientRect().height}:null,ready:f?.contentDocument?.body?.dataset.engineState,rendered:f?.contentDocument?.getElementById('loadIndicator')?.hidden===true,canvases:[...(f?.contentDocument?.querySelectorAll('canvas')??[])].map(c=>({width:c.width,height:c.height})),token:f?.contentWindow?.__ptbTabSentinel,section_count:quality,active:document.querySelector('.tab-wrap.active')?.textContent.trim(),main_height:document.querySelector('#main-content')?.clientHeight,main_scroll_height:document.querySelector('#main-content')?.scrollHeight}})()'''
    def save():(output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    def finish(error=None):
        timer.stop()
        if error:report['error']=error
        report['success']=not error and len(report['cases'])==len(targets) and all(c['inactive_iframe_hidden'] and c['returned_iframe_preserved'] for c in report['cases'])
        save();window.accept_close();window.page.deleteLater();window.view.deleteLater();QTimer.singleShot(200,app.quit)
        print(json.dumps(report,ensure_ascii=False),flush=True)
    def click(title):
        window.page.runJavaScript('(()=>{const e=[...document.querySelectorAll(".nav-item")].find(e=>e.title==='+json.dumps(title)+');if(e)e.click();return !!e;})()')
    def next_target():
        if state['index']>=len(targets):finish();return
        click(targets[state['index']]);state['phase']='target';state['after']=time.monotonic()+.7
    def tick():
        if state['waiting']:return
        if time.monotonic()>state['deadline']:finish('Timeout at '+state['phase']);return
        if time.monotonic()<state['after']:return
        state['waiting']=True
        def inspected(value):
            state['waiting']=False
            if not value:return
            phase=state['phase']
            if phase=='open' and value.get('nav'):
                click('生理参数合成');state['phase']='initial'
            elif phase=='initial' and value.get('ready')=='ready' and value.get('visible') and value.get('rendered'):
                report['initial']=value
                window.grab().save(str(output/'m10-ready.png'))
                window.page.runJavaScript("""document.querySelector('iframe[title="生理参数合成"]').contentWindow.__ptbTabSentinel='preserve-native-view'""")
                next_target()
            elif phase=='target':
                state['phase']='scrolled';state['after']=time.monotonic()+.2
                window.page.runJavaScript('(()=>{for(const e of document.querySelectorAll("#main-content,#main-content *")){if(e.clientHeight&&e.scrollHeight>e.clientHeight&&["auto","scroll"].includes(getComputedStyle(e).overflowY))e.scrollTop=e.scrollHeight;}window.scrollTo(0,document.body.scrollHeight);})()')
            elif phase=='scrolled':
                row={'title':targets[state['index']],'inactive_iframe_hidden':not value.get('visible'),'inactive':value};report['cases'].append(row)
                window.grab().save(str(output/('inactive-'+str(state['index'])+'.png')))
                click('生理参数合成');state['phase']='returned';state['after']=time.monotonic()+.35
            elif phase=='returned' and value.get('ready')=='ready' and value.get('visible'):
                report['cases'][-1]['returned']=value
                report['cases'][-1]['returned_iframe_preserved']=value.get('token')=='preserve-native-view' and value.get('section_count')==report['initial'].get('section_count')
                window.grab().save(str(output/('returned-'+str(state['index'])+'.png')))
                state['index']+=1;save();next_target()
        window.page.runJavaScript(code,inspected)
    timer=QTimer();timer.timeout.connect(tick);timer.start(100)
    app.exec();return 0 if report['success'] else 1

