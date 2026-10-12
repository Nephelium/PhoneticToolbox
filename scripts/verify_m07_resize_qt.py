"""Isolated native Qt M07 layout stability; only synthetic input and owned state."""
from pathlib import Path
import json,os,sqlite3,sys,time,wave,math,struct
ROOT=Path(__file__).resolve().parents[1]
for part in ('scripts','desktop/src','backend/src','packages/phonetic_core/src'):
    sys.path.insert(0,str(ROOT/part))
os.environ['PYTHONPATH']=os.pathsep.join(str(ROOT/p) for p in ('scripts','desktop/src','backend/src','packages/phonetic_core/src'))
os.environ['QT_QPA_PLATFORM']='windows'
# QT_SCALE_FACTOR multiplies Windows' native screen scale, already 150% here.
os.environ['QT_SCALE_FACTOR']='1'
os.environ['QTWEBENGINE_CHROMIUM_FLAGS']='--mute-audio'
from PyQt6.QtCore import Qt,QEventLoop,QTimer
from PyQt6.QtWidgets import QApplication,QFileDialog
from ptb_desktop.host import Workbench,register_scheme
from ptb_worker.local_acoustic_files import initialize_local_files
from verification_artifacts import verification_run

def main():
    scan='--scan' in sys.argv
    with verification_run('m07','resize-qt',{'synthetic':'11025 Hz mono 0.6 s, 140/95 Hz sine with envelope','mode':'before scan' if scan else 'regression'}) as (out,scratch):
        db=scratch/'jobs.sqlite3'
        with sqlite3.connect((ROOT/'output/validation/p06/local-state.sqlite3').as_uri()+'?mode=ro',uri=True) as src,sqlite3.connect(db) as dst:src.backup(dst)
        cache=scratch/'cache';cache.mkdir();initialize_local_files(cache)
        inputs=scratch/'inputs';inputs.mkdir()
        for index,hz in enumerate((140,95)):
            with wave.open(str(inputs/f'input{index}.wav'),'wb') as f:
                f.setparams((1,2,11025,0,'NONE','not compressed'))
                f.writeframes(b''.join(struct.pack('<h',round(16000*math.sin(2*math.pi*hz*i/11025)*math.sin(math.pi*i/6615))) for i in range(6615)))
        register_scheme();app=QApplication(['M07 resize owned QA'])
        w=Workbench(ROOT/'frontend/dist',test=True,jobs_path=db,local_files_root=cache,vocal_profile=scratch/'vocal')
        w.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen);w.resize(1706,1018);w.show();w.page.setAudioMuted(True)
        QFileDialog.getExistingDirectory=lambda *a,**k:str(inputs)
        def pause(ms=100):
            loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
        def js(code):
            loop=QEventLoop();values=[];w.page.runJavaScript(code,lambda v:(values.append(v),loop.quit()));QTimer.singleShot(8000,loop.quit);loop.exec()
            if not values:raise RuntimeError('JS timeout')
            return values[0]
        def until(code,seconds=90):
            end=time.monotonic()+seconds
            while time.monotonic()<end:
                if js(code):return
                pause()
            raise AssertionError(code+' '+str(js('document.body.innerText.slice(-1500)')))
        def click(text):
            print('click '+text,flush=True)
            selector='[...document.querySelectorAll("button")].find(e=>e.offsetParent&&(e.getAttribute("aria-label")||e.textContent.trim())==='+json.dumps(text)+')'
            until('!!('+selector+')&&!('+selector+').disabled');js(selector+'.click()');pause(150)
        def select(label,index):
            print('select '+label,flush=True)
            selector='document.querySelector('+json.dumps('select[aria-label="'+label+'"]')+')'
            until(selector+'?.options.length>'+str(index));js(selector+'.selectedIndex='+str(index)+';'+selector+'.dispatchEvent(new Event("change",{bubbles:true}))');pause(300)
        report={'success':False,'layouts':[],'scope':'Windows native Qt, synthetic audio, hidden owned window'}
        print(out,flush=True)
        try:
            until('!!document.querySelector(".nav-item")')
            if '--kaiti' in sys.argv:
                js('document.documentElement.style.setProperty("--font",getComputedStyle(document.documentElement).getPropertyValue("--font").replace("SimSun","KaiTi"));document.documentElement.style.setProperty("--body-size","15px");localStorage.setItem("ptb.v3.layout.panels.local:M07.workbench",JSON.stringify({"--panel-left":294,"--panel-right":357}));document.querySelector("[aria-label=工具导航宽度]").dispatchEvent(new KeyboardEvent("keydown",{key:"Home",bubbles:true}))')
            click('发声类型合成');click('打开音频目录');select('源音频',1);select('目标音频',2)
            until('document.querySelectorAll(".source-plot .wave-line,.target-plot .wave-line").length===2')
            select('连续统类型',2)
            js('(()=>{const count=document.querySelector("[aria-label=\"F0 控制点数量\"]");count.value="51";count.dispatchEvent(new Event("input",{bubbles:true}));count.dispatchEvent(new Event("change",{bubbles:true}))})()');pause(200)
            click('提取 F0');until('document.querySelector(".m07-page").getAttribute("aria-busy")==="false"');click('生成当前');until('!!document.querySelector(".result-audios")');click('step09');pause(1000)
            js('window.gutterDone=false;(async()=>{const workspace=document.querySelector(".m07-workspace"),workbench=workspace.querySelector(".module-workbench"),center=workbench.querySelector(".workbench-center"),plots=center.querySelector(".m07-plots");const settle=async()=>{for(let i=0;i<8;i++)await new Promise(requestAnimationFrame)};const geometry=()=>[workbench,center,workbench.querySelector(".workbench-right")].map(e=>{const r=e.getBoundingClientRect();return [r.x,r.width]});await settle();const outerBefore=geometry();workbench.style.minHeight=(workspace.clientHeight+20)+"px";await settle();const outerAfter=geometry();workbench.style.minHeight="";await settle();const plotBefore=plots.getBoundingClientRect().width;plots.style.minHeight=(center.clientHeight+20)+"px";await settle();const plotAfter=plots.getBoundingClientRect().width;plots.style.minHeight="";await settle();window.gutterResult={outerBefore,outerAfter,plotBefore,plotAfter};window.gutterDone=true})()')
            until('window.gutterDone',10);report['gutter']=js('window.gutterResult')
            report['device_pixel_ratio']=js('window.devicePixelRatio')
            sizes=[(1706,1018),(1920,1080),(1440,900),(1280,800),(960,720),(800,500),(2559,1527)]
            if scan:sizes=[(1400+i*8,900+i//10*20) for i in range(60)]
            if '--dense' in sys.argv:sizes=[(1650+i*2,1018) for i in range(100)]
            if '--dpr15' in sys.argv:sizes=[(1706+i,1018) for i in range(-20,21)]
            fonts=[None]
            if '--font-matrix' in sys.argv:
                fonts=[(name,size) for name in ('SimSun','Microsoft YaHei','KaiTi') for size in (10,12,14,15,18,20,24)]
                sizes=[(1440,900),(1706,1018),(1920,1080)]
            for font in fonts:
              if font:
                js('document.documentElement.style.setProperty("--font",'+json.dumps('"Times New Roman","'+font[0]+'"')+');document.documentElement.style.setProperty("--body-size",'+json.dumps(str(font[1])+'px')+')');pause(200)
              for width,height in sizes:
                w.resize(width,height);pause(350)
                js('window.resizeFrames=[];window.resizeDone=false;(async()=>{for(let i=0;i<36;i++){await new Promise(requestAnimationFrame);window.resizeFrames.push([".m07-workspace",".module-workbench",".workbench-center",".workbench-right",".m07-plots",".source-plot",".wave-viewport",".wave-track svg"].map(s=>{const e=document.querySelector(".m07-page "+s),r=e.getBoundingClientRect();return [r.x,r.y,r.width,r.height,e.clientWidth,e.clientHeight,e.scrollHeight].map(n=>Math.round(n*100)/100)}));}window.resizeDone=true})()')
                until('window.resizeDone',10);frames=js('window.resizeFrames');variants=len({json.dumps(f) for f in frames[12:]})
                report['layouts'].append(dict(width=width,height=height,font=font,variants=variants,frames=frames));print(font,width,height,variants,flush=True)
                if variants>1:
                    w.view.grab().save(str(out/'unstable.png'))
                    if scan:break
            report['success']=all(r['variants']==1 for r in report['layouts'])
            if not scan:
                gutter=report['gutter'];assert gutter['outerBefore']==gutter['outerAfter'],'workspace scrollbar must not move columns'
                assert gutter['plotBefore']==gutter['plotAfter'],'center scrollbar must not resize plots'
            w.view.grab().save(str(out/'final.png'))
            if not scan:assert report['success'],'M07 layout must settle after resize'
        finally:
            (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
            w.closing=True;w.close();pause(900);app.quit()

if __name__=='__main__':main()
