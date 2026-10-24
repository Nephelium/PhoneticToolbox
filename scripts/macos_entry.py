"""macOS entry for the shared desktop host and its fixed child protocols."""
import json
import os
from pathlib import Path
import sys

ROOT=Path(getattr(sys,'_MEIPASS',Path(__file__).resolve().parents[1]))
for relative in ('scripts','packages/phonetic_core/src','backend/src','desktop/src'):
    sys.path.insert(0,str(ROOT/relative))
os.environ['PYTHONNOUSERSITE']='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['PYTHONPATH']=os.pathsep.join(str(ROOT/relative) for relative in ('backend/src','desktop/src','packages/phonetic_core/src'))

def main():
    if sys.platform!='darwin':raise RuntimeError('macOS entry requires macOS')
    if len(sys.argv)>1 and sys.argv[1]=='--darwin-child':
        from macos_science_entry import main as child
        return child(sys.argv[2:])
    if len(sys.argv)>1 and sys.argv[1]=='--macos-smoke':
        return smoke(Path(sys.argv[2]))
    if len(sys.argv)>1 and sys.argv[1]=='--macos-verify':
        from verify_macos_runtime import main as verify
        return verify(*sys.argv[2:])
    if len(sys.argv)>1 and sys.argv[1]=='--macos-media-verify':
        from verify_macos_media import main as verify
        return verify(*sys.argv[2:])
    if len(sys.argv)>1 and sys.argv[1]=='--macos-egg-scroll':
        from verify_macos_egg_scroll import main as verify
        return verify(*sys.argv[2:])
    if len(sys.argv)>1 and sys.argv[1]=='--macos-tab-isolation':
        from verify_macos_tab_isolation import main as verify
        return verify(*sys.argv[2:])
    if len(sys.argv)>1 and sys.argv[1]=='--macos-performance':
        from verify_macos_performance import main as verify
        return verify(*sys.argv[2:])
    if len(sys.argv)>1 and sys.argv[1]=='--macos-ui-hotspots':
        from verify_macos_ui_hotspots import main as verify
        return verify(*sys.argv[2:])
    if len(sys.argv)>1 and sys.argv[1]=='--macos-icon-size':
        from verify_macos_icon_size import main as verify
        return verify(*sys.argv[2:])
    if len(sys.argv)>1 and sys.argv[1]=='--macos-geolinguistics':
        from verify_geolinguistics_qt import verify
        return verify(ROOT,Path(sys.argv[2]))
    if len(sys.argv)>1 and sys.argv[1]=='--macos-jobs':
        from verify_macos_jobs import main as verify
        return verify(*sys.argv[2:])
    from research_entry import main as research
    return research()

def smoke(output):
    from PyQt6.QtCore import QTimer,Qt
    from PyQt6.QtWidgets import QApplication
    from ptb_desktop.web_assets import register_scheme
    from ptb_desktop.host import Workbench
    from ptb_worker.local_workspace import prepare_workspace
    output.mkdir(parents=True,exist_ok=True)
    database,files=prepare_workspace(output/'scratch'/'workspace',ROOT/'backend/migrations')
    register_scheme();QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    app=QApplication(sys.argv[:1]);app.setApplicationName('PhoneticToolbox-macOS-QA');app.setQuitOnLastWindowClosed(False)
    window=Workbench(ROOT/'frontend/dist',test=True,vocal_profile=output/'scratch'/'vocal-profile',jobs_path=database,local_files_root=files,
                     reaper_binary=ROOT/'resources/research/reaper',vocal_resources=ROOT/'resources/vocal_tract/native')
    window.show();window.raise_()
    result={'success':False,'platform':sys.platform,'scope':'visible Qt WebEngine desktop; default GPU; test profile; no device equivalence'}
    attempts=[0]
    def inspect():
        attempts[0]+=1
        def ready(value):
            if attempts[0]>=5 and value and len(value.get('text',''))>200:
                result.update(success=True,document=value,window=[window.width(),window.height()])
                window.grab().save(str(output/'workbench.png'))
                timer.stop()
                window.page.runJavaScript('[...document.querySelectorAll(".nav-item")].map(e=>e.title).filter(t=>t&&t!=="首页")',navigate)
            elif attempts[0]>45:finish()
        window.page.runJavaScript('({text:document.body.innerText,url:location.href,title:document.title,ready:document.readyState})',ready)
    timer=QTimer();timer.timeout.connect(inspect);timer.start(1000)
    def navigate(titles):
        titles=list(titles or [])+['使用说明'];result['navigation']=[]
        def next_page():
            if not titles:finish();return
            title=titles.pop(0)
            code='(()=>{const title='+json.dumps(title)+';const e=[...document.querySelectorAll(".nav-item,.sidebar-bottom button")].find(e=>e.title===title||e.textContent.trim()===title);if(!e)return false;e.click();return true})()'
            window.page.runJavaScript(code,lambda clicked:QTimer.singleShot(2000,lambda:inspect_page(title,clicked)))
        def inspect_page(title,clicked):
            def recorded(value):
                result['navigation'].append(dict(title=title,clicked=clicked,document=value))
                text=(value or {}).get('text','')+' '.join(f.get('text','') for f in (value or {}).get('frames',[]))
                if not clicked or len(text)<10:result['success']=False
                if title=='生理参数合成' and not any(f.get('engine')=='ready' and f.get('section_count')=='65' for f in (value or {}).get('frames',[])):result['success']=False
                if title=='使用说明' and not (value or {}).get('manual'):result['success']=False
                if title in ('生理参数合成','EGG 信号分析','使用说明','录音'):
                    window.grab().save(str(output/('module-'+str(len(result['navigation']))+'.png')))
                next_page()
            window.page.runJavaScript('({text:document.querySelector("#main-content")?.innerText||"",manual:!!document.querySelector(".manual-reader"),frames:[...document.querySelectorAll("iframe")].filter(f=>f.offsetWidth&&f.offsetHeight).map(f=>({src:f.src,ready:f.contentDocument?.readyState,engine:f.contentDocument?.body?.dataset.engineState,section_count:[...(f.contentDocument?.querySelectorAll("select")??[])].find(s=>[...s.options].some(o=>o.value==="129"))?.value,text:f.contentDocument?.body?.innerText?.slice(0,1000)}))})',recorded)
        next_page()
    def finish():
        timer.stop();(output/'qt-smoke.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        window.accept_close();window.page.deleteLater();window.view.deleteLater();QTimer.singleShot(200,app.quit)
    code=app.exec()
    return 0 if result['success'] else 1

if __name__=='__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    raise SystemExit(main())
