"""M03-R7 actual hidden Windows Qt, public synthetic source, owned test data."""
import argparse,json,os,sqlite3,time
from pathlib import Path
from uuid import uuid4
os.environ.setdefault('QT_QPA_PLATFORM','windows')
from PyQt6.QtCore import QEventLoop,QTimer,Qt,QPoint
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication,QFileDialog
from PyQt6.QtGui import QImage
from ptb_desktop.host import Workbench,register_scheme
from ptb_worker.local_acoustic_files import initialize_local_files

ROOT=Path(__file__).resolve().parents[1]
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True,type=Path);args=parser.parse_args()
    out=ROOT/'output/validation/m03-r7'/('qt-'+uuid4().hex);out.mkdir(parents=True)
    db=out/'jobs.sqlite3'
    with sqlite3.connect((ROOT/'output/validation/p06/local-state.sqlite3').as_uri()+'?mode=ro',uri=True) as src,sqlite3.connect(db) as dst:src.backup(dst)
    cache=out/'cache';cache.mkdir();initialize_local_files(cache);saved=out/'saved';saved.mkdir()
    os.environ['PTB_EGG_PYTHON']=str(ROOT/'.venv/m03-compatible/python.exe')
    register_scheme();app=QApplication(['M03-R7-owned-QA']);app.setApplicationName('M03-R7-owned-QA')
    w=Workbench(ROOT/'frontend/dist',test=True,jobs_path=db,local_files_root=cache,vocal_profile=out/'vocal',reaper_binary=ROOT/'phonetic_toolbox/core/acoustic/reaper.exe')
    w.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen);w.showMaximized();w.page.setAudioMuted(True)
    QFileDialog.getExistingDirectory=lambda *a,**k:str(saved if '结果' in str(a) else args.source.resolve().parent)
    report=dict(success=False,checks=[],schema_applied=[])
    def js(code):
        loop=QEventLoop();box=[];w.page.runJavaScript(code,lambda v:(box.append(v),loop.quit()));QTimer.singleShot(5000,loop.quit);loop.exec();return box[0] if box else None
    def pause(ms=100):loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
    def until(code,seconds=90):
        deadline=time.monotonic()+seconds
        while time.monotonic()<deadline:
            if js(code):return
            pause()
        raise AssertionError('UI timeout '+code+' '+str(js('document.querySelector(".egg-page")?.innerText.slice(0,2000)')))
    def button(text):return '[...document.querySelectorAll("button")].find(b=>b.offsetParent&&b.textContent.trim()==='+json.dumps(text,ensure_ascii=False)+')'
    def click(text):
        rect=js('(()=>{const b='+button(text)+';if(!b)return null;const r=b.getBoundingClientRect();return [r.x+r.width/2,r.y+r.height/2]})()');assert rect,text
        QTest.mouseClick(w.view.focusProxy(),Qt.MouseButton.LeftButton,pos=QPoint(round(rect[0]),round(rect[1])));pause(100)
    def fill(label,value):
        selector=json.dumps('input[aria-label="'+label+'"]')
        js('(()=>{const e=document.querySelector('+selector+');e.value='+json.dumps(value)+';e.dispatchEvent(new Event("input",{bubbles:true}));e.dispatchEvent(new Event("change",{bubbles:true}));})()')
    def idle():until(button('保存 CSV / 三图')+'&&!'+button('保存 CSV / 三图')+'.disabled')
    def range(start,end):
        js('(()=>{const a=document.querySelectorAll(".egg-bottom-bar .selection-controls input");for(const [i,v] of [[1,'+str(end)+'],[0,'+str(start)+']]){a[i].value=v;a[i].dispatchEvent(new Event("input",{bubbles:true}));a[i].dispatchEvent(new Event("change",{bubbles:true}));}})()');pause(150);idle()
    try:
        until('!!document.querySelector(".nav-item")');click('EGG 信号分析');until('!!document.querySelector(".egg-page")')
        click('打开音频目录');until('document.querySelectorAll(".egg-source option").length>=2')
        js('(()=>{const e=document.querySelector(".egg-source select");e.value=[...e.options].find(o=>o.textContent==='+json.dumps(args.source.name)+').value;e.dispatchEvent(new Event("change",{bubbles:true}));})()');idle()
        until('document.querySelectorAll(".egg-four-plots .scientific-plot>svg").length===4')
        assert js('document.querySelector(".egg-current").innerText.includes("48000 Hz · 1800.000 s")')
        assert js('document.querySelector(".egg-current").innerText.includes("原采样率选区试听")')
        report['checks'].append('original 48kHz/1800s metadata and four real scientific plots')
        range(1790,1800);fill('EGG 高通频率',30);pause(150);idle()
        js('document.querySelectorAll(".egg-checks input").forEach(e=>{if(!e.checked)e.click()})');pause(200);idle()
        until('document.querySelector(".spec-pane").innerText.includes("REAPER F0")')
        assert js('document.querySelectorAll(".spec-pane svg circle").length')>20
        click('播放选区');until('Number(document.querySelector(".egg-bottom-bar .playback-seek input").value)>1790');click('停止')
        report['checks'].append('muted native WebAudio starts at global tail coordinates')
        report['checks'].append('tail navigation, filter update and three F0 native viewport paths')
        for mode in ['light','dark']:
            js('document.documentElement.dataset.theme='+json.dumps(mode));pause(200)
            for width,height in [(1920,1080),(1440,900)]:
                w.showNormal();w.resize(width,height);pause(200);w.view.grab().save(str(out/f'{mode}-{width}.png'))
                assert js('document.querySelectorAll(".egg-four-plots .scientific-plot>svg").length')==4
        report['checks'].append('four plots in two native window sizes and light/dark themes')
        range(1789.999,1800);click('逆滤波 IF');until('document.querySelector(".egg-page").innerText.includes("960000")')
        w.showMaximized();pause(400);w.view.grab().save(str(out/'ten-second-limit.png'))
        report['checks'].append('10.001s inverse explicitly rejected in actual Qt')
        range(1790,1800);click('逆滤波 IF')
        until('!!document.querySelector(".inverse-grid")',240)
        until('!!document.querySelector("dialog")')
        until(button('选择目录保存完整结果')+'&&!'+button('选择目录保存完整结果')+'.disabled')
        until('document.querySelectorAll(".inverse-grid .scientific-plot>svg").length>=4');pause(500)
        w.view.grab().save(str(out/'inverse-ten-seconds.png'))
        click('选择目录保存完整结果')
        deadline=time.monotonic()+45
        while len(list(saved.glob('*.wav')))<2 and time.monotonic()<deadline:pause()
        assert len(list(saved.glob('*.wav')))==2
        from scipy.io import wavfile
        for path in saved.glob('*.wav'):
            fs,values=wavfile.read(path);assert fs==48000 and len(values)==480000
        report['checks'].append('native ten-second inverse job, result dialog and full WAV save/readback')
        report['success']=True
    except Exception as error:
        report['error']=str(error);w.view.grab().save(str(out/'failed.png'));raise
    finally:
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        js('window.onbeforeunload=null');w.closing=True;w.close();pause(500);w.page.deleteLater();app.processEvents();app.quit()
        report['service_exit_code']=w.service.exit_code
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(out,flush=True)
if __name__=='__main__':main()
