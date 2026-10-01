"""Owned Windows Qt font UI, installed font enumeration and module propagation."""
import json
import hashlib
import time
import argparse
from pathlib import Path
from uuid import uuid4
from PyQt6.QtCore import QEventLoop,QTimer,Qt
from PyQt6.QtWidgets import QApplication,QFileDialog
from ptb_desktop.host import Workbench,register_scheme

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--m10-only',action='store_true');args=parser.parse_args()
    out=ROOT/'output/validation/fonts'/('qt-'+('m10-' if args.m10_only else 'research-')+uuid4().hex);inputs=out/'inputs';inputs.mkdir(parents=True)
    if not args.m10_only:
        from verify_m02_m09_local import fixtures
        fixtures(inputs)
    originals={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.iterdir()}
    register_scheme();app=QApplication(['P04-FONT-owned-QA']);app.setApplicationName('P04-FONT-owned-QA')
    w=Workbench(ROOT/'frontend/dist',test=True,jobs_path=None if args.m10_only else ROOT/'output/validation/p06/local-state.sqlite3',vocal_profile=out/'vocal-profile',vocal_resources=ROOT/'resources/vocal_tract/native',start_module='M10' if args.m10_only else None)
    w.resize(1440,1000);w.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating);w.show()
    checks=[];report={'success':False,'checks':checks,'schema_applied':[]};target=out/'font-test.png'
    QFileDialog.getExistingDirectory=lambda *a,**k:str(inputs)
    QFileDialog.getSaveFileName=lambda *a,**k:(str(target),'')
    def js(code):
        loop=QEventLoop();box=[];w.page.runJavaScript(code,lambda value:(box.append(value),loop.quit()));QTimer.singleShot(5000,loop.quit);loop.exec();return box[0] if box else None
    def pause():loop=QEventLoop();QTimer.singleShot(120,loop.quit);loop.exec()
    def until(code,seconds=45):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            if js(code):return
            pause()
        raise RuntimeError('UI timeout: '+code)
    def click(text):js('[...document.querySelectorAll("button")].find(b=>b.offsetParent&&b.textContent.trim()==='+json.dumps(text,ensure_ascii=False)+')?.click()')
    def close_settings():js('document.querySelector('+json.dumps('button[aria-label="关闭 设置"]')+').click()')
    def fill(label,value):js('(()=>{const e=document.querySelector("input[aria-label=\\"'+label+'\\"]");e.value='+json.dumps(value)+';e.dispatchEvent(new Event("input",{bubbles:true}));})()')
    def settings(zh,latin,size=12):
        click('设置');until('!!document.querySelector(".font-settings")');fill('中文字体',zh);fill('英文与数字字体',latin);fill('图表基础字号',size);click('应用字体');until('document.querySelector(".font-settings [role=status]")?.textContent.includes("字体已应用")');
    try:
        until('document.documentElement.style.getPropertyValue("--font").length>0');settings('SimSun','Times New Roman')
        click('读取本机字体列表');until('document.querySelector(".font-settings [role=status]")?.textContent.includes("已读取")');assert js('[...document.querySelectorAll("#ptb-font-options option")].some(e=>e.value==="SimSun")')
        w.view.grab().save(str(out/'settings.png'));checks.append('real Qt family enumeration and settings apply')
        print('Qt settings verified',flush=True)
        if args.m10_only:
            close_settings()
            until('document.querySelector("iframe")?.contentDocument?.querySelector("#engineStatus")?.textContent.includes("已连接")',55)
            settings('KaiTi','Arial');close_settings();until('document.querySelector("iframe").contentDocument.documentElement.style.getPropertyValue("--sans").includes("KaiTi")')
            assert 'PTB-Doulos' in js('getComputedStyle(document.querySelector("iframe").contentDocument.querySelector("#presets button")).fontFamily');checks.append('M10 native page receives font change, phoneme presets fixed Doulos')
            w.view.grab().save(str(out/'m10.png'));print('M10 live font change verified',flush=True)
            def inner(selector):js('document.querySelector("iframe").contentDocument.querySelector('+json.dumps(selector)+').click()')
            inner('#tab-motion');inner('#captureFrame');until('document.querySelector("iframe").contentDocument.body.dataset.posePending==="false"');inner('#captureFrame')
            until('document.querySelector("iframe").contentDocument.querySelectorAll(".pose-card").length===2')
            w.bridge.test_vocal_picker=lambda op:str(out/'fonts.webm')
            inner('#exportVideo');inner('#videoStart')
            until('document.querySelector("iframe").contentDocument.querySelector("#videoStart").disabled')
            settings('SimSun','Times New Roman');close_settings()
            if js('document.querySelector("iframe").contentDocument.querySelector("#videoStart").disabled'):
                assert 'KaiTi' in js('document.querySelector("iframe").contentDocument.documentElement.style.getPropertyValue("--sans")');checks.append('in-flight video retains original font snapshot')
            until('document.querySelector("iframe").contentDocument.querySelector("#videoStatus").textContent.startsWith("已保存")',120)
            assert (out/'fonts.webm').stat().st_size>1000
            until('document.querySelector("iframe").contentDocument.documentElement.style.getPropertyValue("--sans").includes("SimSun")');checks.append('actual M10 video saved, queued font applies after export')
            report['success']=True
            return
        close_settings();click('参数估计');until('!!document.querySelector(".m01-page")');assert 'SimSun' in js('getComputedStyle(document.querySelector(".m01-page input")).fontFamily');checks.append('M01 controls inherit fonts')
        click('参数显示');until('!!document.querySelector(".m02-page")');click('选择音频目录');until('document.querySelectorAll(".m02-files button").length>0');click('tone.wav');until('!!document.querySelector(".empty-plot")&&document.querySelectorAll(".m02-parameters input").length===4');assert js('document.querySelectorAll(".m02-parameters input:checked").length===0');click('全选可见参数');until('document.querySelectorAll(".m02-parameters input:checked").length===3');click('将 3 项分配到图窗');until('document.querySelectorAll(".parameter-curve").length===3')
        assert 'PTB-Doulos' in js('getComputedStyle(document.querySelector(".m02-wave-annotations text")).fontFamily');click('保存整幅 PNG');until('!document.querySelector(".parameter-figure button:last-child").disabled');
        end=time.monotonic()+10
        while not target.exists() and time.monotonic()<end:pause()
        assert target.is_file();checks.append('M02 actual native PNG download and fixed Doulos annotation')
        click('语谱图转音频');until('!!document.querySelector(".m09-page")');assert 'SimSun' in js('getComputedStyle(document.querySelector(".m09-page input")).fontFamily');checks.append('M09 controls inherit fonts')
        settings('KaiTi','Arial',24);close_settings()
        click('参数显示');until('document.querySelector(".m02-page").offsetParent!==null');assert 'KaiTi' in js('getComputedStyle(document.querySelector(".parameter-chart")).fontFamily');assert js('document.querySelectorAll(".parameter-curve").length===3');checks.append('already open M02 keeps curves and receives new global fonts')
        target=out/'font-changed.png';click('保存整幅 PNG');until('!document.querySelector(".parameter-figure button:last-child").disabled');
        end=time.monotonic()+10
        while not target.exists() and time.monotonic()<end:pause()
        assert target.is_file();assert target.read_bytes()!=(out/'font-test.png').read_bytes()
        import cv2
        for name in ['font-test.png','font-changed.png']:
            import numpy as np
            image=cv2.imdecode(np.frombuffer((out/name).read_bytes(),dtype=np.uint8),cv2.IMREAD_COLOR);assert image is not None and image.shape[0]>1000
        assert originals=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.iterdir()}
        report['success']=True
    except Exception as error:
        report['error']=str(error);w.view.grab().save(str(out/'failed.png'));raise
    finally:
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(out,flush=True);w.closing=True;w.close();app.processEvents()

if __name__=='__main__':main()
