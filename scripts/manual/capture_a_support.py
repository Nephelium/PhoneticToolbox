"""Owned Qt capture support for the M01/M04/M06/M09/M10 manual recipes."""
from __future__ import annotations
import hashlib,json,os,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from manual.capture_locale import CaptureLocale
from workbench_source import configure

class Capture:
    def __init__(self,args,out,scratch,chapter,namespace,*,reaper_binary=None):
        configure()
        os.environ['QT_QPA_PLATFORM']='windows';os.environ['QTWEBENGINE_CHROMIUM_FLAGS']='--mute-audio'
        for key in ('QT_ENABLE_HIGHDPI_SCALING','QT_SCALE_FACTOR','QT_SCREEN_SCALE_FACTORS'):os.environ.pop(key,None)
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QApplication
        from ptb_desktop.host import Workbench,register_scheme
        from ptb_worker.local_workspace import prepare_workspace
        self.args,self.out,self.scratch,self.chapter=args,out,scratch,chapter
        self.ui=CaptureLocale(args.locale,(namespace,'workbench','components','common'))
        files=[Path(sys.argv[0]),Path(__file__),ROOT/'scripts/manual/capture_locale.py',ROOT/f'manual/chapters/{chapter}.json',args.frontend_dist/'index.html',*sorted((ROOT/'frontend/src/modules'/namespace).rglob('*.vue')),*sorted((ROOT/'frontend/src/modules'/namespace).rglob('*.ts'))]
        files.extend(ROOT/f'frontend/src/i18n/locales/{code}/{namespace}.json'for code in ('zh-CN',args.locale))
        self.sources={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest()for p in files if p.is_file()}
        register_scheme();self.app=QApplication(['Owned manual '+chapter+' '+args.locale]);db,cache=prepare_workspace(scratch/'state',ROOT/'backend/migrations')
        self.window=Workbench(args.frontend_dist.resolve(),test=True,jobs_path=db,local_files_root=cache,reaper_binary=reaper_binary,vocal_profile=scratch/'vocal',vocal_resources=ROOT/'resources/vocal_tract/native',start_module=chapter.upper())
        self.window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen);self.window.resize(self.window.screen().geometry().width(),self.window.screen().geometry().height());self.window.showMaximized();self.window.view.setZoomFactor(1);self.window.page.setAudioMuted(True)
        self.report={'success':False,'sourceChapter':chapter,'locale':args.locale,'captures':[],'checks':[],'physicalCompositorVerified':False,'frontend':str(args.frontend_dist.resolve())}
    def pause(self,ms=120):
        from PyQt6.QtCore import QEventLoop,QTimer
        loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
    def js(self,code):
        from PyQt6.QtCore import QEventLoop,QTimer
        loop=QEventLoop();values=[]
        wrapped='(()=>{try{return {ok:true,value:eval('+json.dumps(code)+')??null};}catch(e){return {ok:false,error:String(e)}}})()'
        self.window.page.runJavaScript(wrapped,lambda v:(values.append(v),loop.quit()));QTimer.singleShot(15000,loop.quit);loop.exec()
        if not values or not values[0]or not values[0].get('ok'):raise RuntimeError(str(values)+' '+code)
        return values[0].get('value')
    def until(self,code,seconds=80):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            if self.js(code):return
            self.pause()
        raise AssertionError(code+'\n'+str(self.js('document.body.innerText.slice(-2500)')))
    def click(self,chinese):
        label=self.ui.label(chinese)
        assert self.js('(()=>{const e=[...document.querySelectorAll("button")].find(b=>b.offsetParent&&[b.textContent.trim(),b.getAttribute("aria-label"),b.title].includes('+json.dumps(label)+'));if(!e||e.disabled)return false;e.click();return true})()'),label
        self.pause()
    def prepare(self):
        self.ui.prepare_qt(self.js,self.until);self.click('设置');self.click('浅色');self.until('document.documentElement.dataset.theme==="light"')
        self.js('document.querySelector("#tab-'+self.chapter.upper()+'").click();document.querySelector("#tab-settings").closest(".tab-wrap").querySelector(".tab-close").click()')
        if not self.js('document.querySelector(".app-shell").classList.contains("collapsed")'):self.click('收起侧栏')
        self.pause(500)
    def fill(self,selector,value,event='input'):
        assert self.js('(()=>{const e=document.querySelector('+json.dumps(selector)+');if(!e)return false;e.value='+json.dumps(str(value))+';e.dispatchEvent(new Event('+json.dumps(event)+',{bubbles:true}));'+('e.dispatchEvent(new Event("change",{bubbles:true}));'if event=='input'else'')+'return true})()'),selector
        self.pause(180)
    def select(self,selector,index=1):
        self.until('document.querySelector('+json.dumps(selector)+')?.options.length>'+str(index))
        self.js('(()=>{const e=document.querySelector('+json.dumps(selector)+');e.value=e.options['+str(index)+'].value;e.dispatchEvent(new Event("change",{bubbles:true}));})()');self.pause(200)
    def frame(self,code):return self.js('(()=>{const f=document.querySelector(".vocal-page iframe"),d=f.contentDocument,win=f.contentWindow;'+code+'})()')
    def frame_click(self,selector):
        assert self.frame('const e=d.querySelector('+json.dumps(selector)+');if(!e||e.disabled)return false;e.click();return true;'),selector
        self.pause()
    def frame_ready(self):self.until('document.querySelector(".vocal-page iframe")?.contentDocument?.body?.dataset.posePending==="false"')
    def frame_fill(self,selector,value,event='input'):
        assert self.frame('const e=d.querySelector('+json.dumps(selector)+');if(!e)return false;e.value='+json.dumps(str(value))+';e.dispatchEvent(new Event('+json.dumps(event)+',{bubbles:true}));return true;'),selector
        self.pause(180)
    def shot(self,name,regions=None):
        metrics=self.ui.metrics(self.js);self.pause(550);self.window.view.repaint();self.pause(100)
        assert self.js('document.querySelector(".app-shell").classList.contains("collapsed")')
        image=self.window.view.grab().toImage();raw=[image.width(),image.height()];dpr=image.devicePixelRatio()
        assert raw in ([2560,1600],[2561,1601])and abs(dpr-1.5)<.01,(raw,dpr)
        assert self.window.isMaximized()and self.window.view.zoomFactor()==1
        if raw!=[2560,1600]:image=image.copy(0,0,2560,1600)
        path=self.out/(name+'.png');assert image.save(str(path))
        boxes={}
        for key,selector in (regions or {}).items():
            framed=selector.startswith('frame:');query=selector[6:]if framed else selector
            script='const e=d.querySelector('+json.dumps(query)+');if(!e)return null;const r=e.getBoundingClientRect(),p=f.getBoundingClientRect();return [p.x+r.x,p.y+r.y,r.width,r.height];'if framed else'(()=>{const r=document.querySelector('+json.dumps(query)+')?.getBoundingClientRect();return r?[r.x,r.y,r.width,r.height]:null})()'
            box=self.frame(script)if framed else self.js(script)
            if box and box[2]>0 and box[3]>0:boxes[key]={'selector':selector,'box':box,'units':'css-px'}
        row={'name':name,'file':path.name,'image':[2560,1600],'raw_image':raw,'dpr':dpr,'pageZoom':1,'maximized':True,'theme':'light','navigationCollapsed':True,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'locale':metrics['locale'],'fontLocale':metrics['fontLocale'],'fontFamily':metrics['fontFamily'],'regions':boxes,'visibleText':self.js('document.body.innerText')+(self.frame('return d.body.innerText;')if self.chapter=='m10'else'')}
        self.report['captures'].append(row);print('Captured '+name,flush=True)
    def close(self):
        from PyQt6 import sip
        self.report['sources']={p:{'path':p,'sha256':sha}for p,sha in self.sources.items()}
        self.report['sourceChangesDuringRun']=[p for p,sha in self.sources.items()if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=sha]
        (self.out/'report.json').write_text(json.dumps(self.report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        self.window.closing=True;self.window.close();self.app.processEvents();sip.delete(self.window);self.app.processEvents();sip.delete(self.app)
        print(json.dumps({'success':self.report['success'],'report':str(self.out/'report.json'),'captures':len(self.report['captures'])},ensure_ascii=False),flush=True)
