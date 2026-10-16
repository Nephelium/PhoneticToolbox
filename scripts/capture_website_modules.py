"""Website screenshots in a hidden, isolated native workbench. No camera capture."""
from pathlib import Path
import hashlib,json,os,shutil,sqlite3,sys,time
ROOT=Path(__file__).resolve().parents[1]
for part in ('scripts','desktop/src','backend/src','packages/phonetic_core/src'):
    sys.path.insert(0,str(ROOT/part))
os.environ['PYTHONPATH']=os.pathsep.join(str(ROOT/p) for p in ('scripts','desktop/src','backend/src','packages/phonetic_core/src'))
os.environ['QT_QPA_PLATFORM']='windows'
os.environ['QT_ENABLE_HIGHDPI_SCALING']='0'
os.environ['QT_SCALE_FACTOR']='1'
os.environ['QTWEBENGINE_CHROMIUM_FLAGS']='--mute-audio'
os.environ['PTB_EGG_PYTHON']=str(ROOT/'.venv/m03-compatible/python.exe')
from PyQt6.QtCore import Qt,QEventLoop,QTimer
from PyQt6.QtWidgets import QApplication,QFileDialog
from ptb_desktop.host import Workbench,register_scheme
from ptb_worker.local_acoustic_files import initialize_local_files
from verification_artifacts import verification_run

class Capture:
    def __init__(self,out,scratch):
        self.out=out;self.report={'success':False,'captures':[],'errors':[]}
        db=scratch/'jobs.sqlite3'
        with sqlite3.connect((ROOT/'output/validation/p06/local-state.sqlite3').as_uri()+'?mode=ro',uri=True) as src,sqlite3.connect(db) as dst:src.backup(dst)
        cache=scratch/'cache';cache.mkdir();initialize_local_files(cache)
        register_scheme();self.app=QApplication(['PhoneticToolbox Website Capture'])
        self.w=Workbench(ROOT/'frontend/dist',test=True,jobs_path=db,local_files_root=cache,vocal_profile=scratch/'vocal',reaper_binary=ROOT/'phonetic_toolbox/core/acoustic/reaper.exe')
        self.w.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen);self.w.resize(2560,1440);self.w.show();self.w.page.setAudioMuted(True)
        self.w.page.renderProcessTerminated.connect(lambda s,c:self.report['errors'].append([s.name,c]))
        self.folder=scratch
        QFileDialog.getExistingDirectory=lambda *a,**k:str(self.folder)
    def pause(self,ms=100):
        loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
    def js(self,code):
        loop=QEventLoop();values=[];self.w.page.runJavaScript(code,lambda v:(values.append(v),loop.quit()));QTimer.singleShot(8000,loop.quit);loop.exec()
        if not values:raise RuntimeError('JS timeout')
        return values[0]
    def until(self,code,seconds=90):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            if self.js(code):return
            self.pause()
        raise AssertionError(code+' '+str(self.js('document.body.innerText.slice(-1500)')))
    def click(self,label):
        sel='[...document.querySelectorAll("button")].find(b=>b.offsetParent&&(b.getAttribute("aria-label")||b.textContent.trim())==='+json.dumps(label)+')'
        self.until('!!('+sel+')&&!('+sel+').disabled');self.js(sel+'.click()');self.pause(150)
    def module(self,title):
        sel='[...document.querySelectorAll(".nav-item")].find(b=>b.title==='+json.dumps(title)+')'
        self.until('!!('+sel+')');self.js(sel+'.click()');self.pause(450)
    def select(self,label,index=1):
        sel=json.dumps('select[aria-label="'+label+'"]');self.until('document.querySelector('+sel+')?.options.length>'+str(index))
        self.js('(()=>{const e=document.querySelector('+sel+');e.value=e.options['+str(index)+'].value;e.dispatchEvent(new Event("change",{bubbles:true}))})()');self.pause(400)
    def palette(self,palette,mode):
        self.click('设置');self.click('外观');self.click('浅色' if mode=='light' else '深色');self.click('配色方案')
        self.until('!!document.querySelector("[data-palette-option='+palette+']")');self.js('document.querySelector("[data-palette-option='+palette+']").click()')
        self.until('document.documentElement.dataset.palette==='+json.dumps(palette)+'&&document.documentElement.dataset.theme==='+json.dumps(mode))
        self.current_palette=palette;self.current_mode=mode
        for _ in range(10):
            scale=self.js('document.querySelector("[aria-label=页面缩放比例]")?.textContent')
            if scale=='150%':break
            self.click('放大页面' if int(scale.strip('%'))<150 else '缩小页面')
        assert self.js('document.querySelector("[aria-label=页面缩放比例]").textContent')=='150%'
    def snapshot(self,name,module,description):
        self.until("document.fonts.status==='loaded'");self.pause(1600)
        body=self.js('document.body.innerText');assert 'C:\\Users\\13680' not in body
        file=ROOT/'temporary-site/assets/screens'/(name+'.png');pix=self.w.view.grab();assert (pix.width(),pix.height())==(2560,1440);assert pix.save(str(file))
        self.report['captures'].append({'file':file.name,'module':module,'palette':self.current_palette,'mode':self.current_mode,'width':pix.width(),'height':pix.height(),'source':'Native Windows Qt built workbench, complete unedited view','description':description,'sha256':hashlib.sha256(file.read_bytes()).hexdigest()})
        self.report['captures'][-1]['page_scale_percent']=150
        print('Captured '+name,flush=True);self.save()
    def save(self):
        (self.out/'report.json').write_text(json.dumps(self.report,ensure_ascii=False,indent=2)+'\n','utf-8')
    def close(self):
        self.save();self.w.closing=True;self.w.close();self.pause(900);self.app.quit()

def main():
    with verification_run('website','native-modules','Five native views: M10, M03, M09, M08 and icons; existing demonstration speech/EGG; no camera/microphone') as (out,scratch):
        speech=scratch/'语句示例';speech.mkdir();egg=scratch/'EGG示例';egg.mkdir()
        # Inputs previously used for project documentation; no raw media published.
        originals=[(Path(r'C:\Users\13680\Desktop\project\音频数据\线下录音\女1\原声_平静.wav'),speech/'语句示例.wav'),(Path(r'C:\Users\13680\Desktop\project\音频数据\EGG测试\3.wav'),egg/'EGG示例.wav')]
        for src,dst in originals:shutil.copyfile(src,dst)
        c=Capture(out,scratch)
        if len(sys.argv)>1:
            prior=json.loads(Path(sys.argv[1]).read_text('utf-8'))
            for row in prior['captures']:
                assert hashlib.sha256((ROOT/'temporary-site/assets/screens'/row['file']).read_bytes()).hexdigest()==row['sha256']
            c.report['captures']=prior['captures']
        try:
            c.until("!!document.querySelector('.home-page')")
            if not c.report['captures']:
                c.palette('solarized','light');c.module('生理参数合成')
                c.until('document.querySelector(".vocal-page iframe")?.contentDocument?.querySelector("canvas")?.width>0',180)
                c.pause(3000);c.snapshot('vocal-solarized-light','M10','Native articulatory geometry and physiological parameter controls')
                c.palette('gruvbox','light');c.module('EGG 信号分析');c.folder=egg;c.click('打开音频目录');c.select('EGG 音频文件')
                c.until('document.querySelector(".egg-live-status")?.textContent.includes("实时预览")',180)
                c.js("(()=>{const a=document.querySelectorAll('.egg-bottom-bar .selection-controls input');for(const [i,v] of [[1,41],[0,40]]){a[i].value=v;a[i].dispatchEvent(new Event('input',{bubbles:true}));a[i].dispatchEvent(new Event('change',{bubbles:true}))}})()")
                c.pause(1000);c.until('document.querySelector(".egg-live-status")?.textContent.includes("实时预览")',180)
                c.snapshot('egg-gruvbox-light','M03','Actual EGG preview, existing demonstration recording, voiced interval 40–41 s')
            c.palette('ayu','dark');c.module('语谱图转音频');c.click('音频藏信息');c.folder=speech;c.click('打开音频目录');c.select('藏信息源音频')
            c.until('document.querySelector(".spectral-editor canvas")?.width>0',120);c.until('document.body.innerText.includes("画布已载入")',120)
            c.snapshot('spectrum-ayu-dark','M09','Actual spectrogram loaded from existing demonstration speech')
            c.palette('one','light');c.module('变速变调');c.folder=speech;c.click('打开音频目录')
            c.js('(()=>{const e=document.querySelector(".m08-page select");e.value=e.options[1].value;e.dispatchEvent(new Event("change",{bubbles:true}));})()')
            c.until('!!document.querySelector("svg.m08-curve")');c.until('document.querySelector(".m08-page")?.getAttribute("aria-busy")==="false"',180)
            c.snapshot('pitch-one-light','M08','Actual waveform and pitch track from existing demonstration speech')
            c.palette('dracula','dark');c.click('图标');c.until('document.querySelectorAll(".icon-choice").length>=20')
            c.snapshot('icons-dracula-dark','settings-icons','Current application icon selection page')
            assert not c.report['errors'];c.report['success']=True
            p=ROOT/'temporary-site/assets/screens/provenance.json';data=json.loads(p.read_text('utf-8'))
            names={r['file'] for r in c.report['captures']};data['captures']=[r for r in data['captures'] if r['file'] not in names]+c.report['captures']
            data['scope']='Current Chinese application views from Vue/Chrome and native Qt. Actual existing example inputs for analysis. No raw media, camera capture, cropping or recoloring; web CSS perspective only.'
            p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n','utf-8')
        except Exception as e:
            c.report['failure']=str(e);c.w.view.grab().save(str(out/'failure.png'));raise
        finally:c.close();print(json.dumps({'out':str(out),'success':c.report['success']}),flush=True)

if __name__=='__main__':main()
