"""Actual built workbench/QWebChannel/local API with task-owned synthetic WAV."""
import hashlib
import json
import os
from pathlib import Path
import time
import wave
from workbench_source import configure
from verification_artifacts import verification_run, write_json


def main():
    root=Path(__file__).resolve().parents[1]
    configure(root)
    os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS','--mute-audio')
    import numpy as np
    from PyQt6.QtCore import QEventLoop,QTimer
    from PyQt6.QtWidgets import QApplication,QFileDialog
    from ptb_desktop.host import Workbench,register_scheme
    register_scheme()
    app=QApplication(['Desktop-preview-owned-QA'])
    with verification_run('desktop-preview-performance','qt','Visible default-GPU Windows source Workbench; M01 synthetic 1 s stereo WAV; repeated spectrogram in light/dark themes; no real device or database') as (out,scratch):
        inputs=scratch/'input';inputs.mkdir()
        t=np.arange(16000)/16000
        x=np.column_stack((.2*np.sin(2*np.pi*137*t),.3*np.sin(2*np.pi*211*t)))
        with wave.open(str(inputs/'synthetic.wav'),'wb') as stream:
            stream.setnchannels(2);stream.setsampwidth(2);stream.setframerate(16000)
            stream.writeframes((x*32767).astype('<i2').tobytes())
        original_dialog=QFileDialog.getExistingDirectory
        QFileDialog.getExistingDirectory=lambda *args,**kwargs:str(inputs)
        report=dict(success=False,requests=[],themes=[])
        window=None
        def pause(ms=80):
            loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
        def js(code):
            loop=QEventLoop();values=[]
            window.page.runJavaScript(code,lambda value:(values.append(value),loop.quit()))
            QTimer.singleShot(5000,loop.quit);loop.exec();assert values,code
            return values[0]
        def until(code):
            deadline=time.monotonic()+25
            while not js(code):
                assert time.monotonic()<deadline,code
                pause()
        def click(text):
            assert js('(()=>{const b=[...document.querySelectorAll("button")].find(b=>b.textContent.trim()==='+json.dumps(text)+');if(!b)return false;b.click();return true})()'),text
        try:
            window=Workbench(root/'frontend/dist',test=True,start_module='M01',vocal_profile=scratch/'vocal')
            original_preview=window.service.preview
            def preview(payload,view):
                started=time.perf_counter();value=original_preview(payload,view)
                report['requests'].append(dict(view=dict(view),milliseconds=(time.perf_counter()-started)*1000,
                    sha256=hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()))
                return value
            window.service.preview=preview
            window.resize(1440,1000);window.show()
            until('!!document.querySelector(".m01-page")')
            click('打开音频目录')
            until('!!document.querySelector(".m01-file-list button.file-row")')
            js('document.querySelector(".m01-file-list button.file-row").click()')
            until('!!document.querySelector(".wave-line")')
            for theme in ('light','dark'):
                js('document.documentElement.dataset.theme='+json.dumps(theme))
                for _ in range(2):
                    js('(()=>{const c=[...document.querySelectorAll("label")].find(e=>e.textContent.includes("显示语谱图"))?.querySelector("input");if(!c.checked)c.click()})()')
                    until('document.querySelector(".spectrogram-canvas")?.getAttribute("aria-busy")==="false"&&getComputedStyle(document.querySelector(".spectrogram-canvas canvas")).visibility==="visible"')
                    pixels=js('(()=>{const c=document.querySelector(".spectrogram-canvas canvas"),a=c.getContext("2d").getImageData(0,0,c.width,c.height).data;let dark=0;for(let i=0;i<a.length;i+=4)if(a[i]<200)dark++;return dark})()')
                    assert pixels>100,pixels
                    report['themes'].append(dict(theme=theme,dark_pixels=pixels))
                    window.view.grab().save(str(out/(theme+'.png')))
                    js('[...document.querySelectorAll("label")].find(e=>e.textContent.includes("显示语谱图")).querySelector("input").click()')
                    pause()
            groups={}
            for item in report['requests']:groups.setdefault(json.dumps(item['view'],sort_keys=True),[]).append(item)
            repeated=[items for items in groups.values() if len(items)>=2]
            assert repeated,report
            assert all(len({item['sha256'] for item in items})==1 for items in repeated)
            report['success']=True
        except Exception as error:
            report['error']=repr(error)
            if window:
                report['page_text']=js('document.body.innerText')
                window.view.grab().save(str(out/'failure.png'))
            raise
        finally:
            QFileDialog.getExistingDirectory=original_dialog
            if window:
                window.closing=True;window.close();window.page.deleteLater();pause(100)
                window.profile.deleteLater();pause(100);window.deleteLater();pause(100)
            write_json(out/'report.json',report)
            print(json.dumps(dict(output=str(out),**report),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
