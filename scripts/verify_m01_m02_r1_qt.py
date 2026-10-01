"""Hidden native Qt + built shared frontend; read-only optional user corpus, no DB."""
import argparse
import hashlib
import json
import time
from pathlib import Path
from uuid import uuid4
import numpy as np
import soundfile as sf
from PyQt6.QtCore import QEventLoop, QTimer, Qt
from PyQt6.QtWidgets import QApplication, QFileDialog
from ptb_desktop.host import Workbench, register_scheme

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--corpus',type=Path,required=True);args=parser.parse_args()
    out=ROOT/'output/validation/m01-m02-r1'/('qt-'+uuid4().hex);inputs=out/'inputs';inputs.mkdir(parents=True)
    long=inputs/'长录音-双声道.wav';rate=48000;duration=650
    # Constant-size chunks, separate tones make channel/order mistakes observable.
    t=np.arange(rate)/rate
    chunk=np.column_stack((.25*np.sin(2*np.pi*120*t),.4*np.sin(2*np.pi*250*t))).astype('float32')
    with sf.SoundFile(long,'w',samplerate=rate,channels=2,subtype='FLOAT',format='WAV') as audio:
        for _ in range(duration):audio.write(chunk)
    (inputs/'nested').mkdir();(inputs/'nested/short.wav').write_bytes(next(args.corpus.glob('*.wav')).read_bytes())
    original={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.corpus.iterdir() if p.suffix.lower() in ('.wav','.textgrid','.json','.xlsx','.sqlite')}
    long_sha=hashlib.sha256(long.read_bytes()).hexdigest()
    register_scheme();app=QApplication(['M01 M02 native verification'])
    window=Workbench(ROOT/'frontend/dist',test=True,vocal_profile=out/'vocal')
    window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen,True);window.resize(1600,1000);window.show()
    selected_directory=[args.corpus]
    QFileDialog.getExistingDirectory=lambda *a,**k:str(selected_directory[0])
    downloaded=[]
    def save_dialog(*a,**k):
        name=Path(a[2]).name;downloaded.append(name);return str(out/name),''
    QFileDialog.getSaveFileName=save_dialog
    checks=[];report={'success':False,'checks':checks,'source_bytes':long.stat().st_size,'duration':duration,'schema_applied':[]}
    def js(code):
        loop=QEventLoop();values=[];window.page.runJavaScript(code,lambda v:(values.append(v),loop.quit()));QTimer.singleShot(6000,loop.quit);loop.exec()
        return values[0] if values else None
    def pause(ms=100):
        loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
    def until(code,seconds=80):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            if js(code):return
            pause()
        raise AssertionError('UI timeout: '+code+' / '+str(js('document.querySelector(".error-banner,[role=alert]")?.textContent')))
    def click(text):
        until('[...document.querySelectorAll("button")].some(b=>b.offsetParent&&!b.disabled&&b.textContent.trim()==='+json.dumps(text)+')')
        js('[...document.querySelectorAll("button")].find(b=>b.offsetParent&&b.textContent.trim()==='+json.dumps(text)+')?.click()');pause()
    def check(selector):
        js('document.querySelector('+json.dumps(selector)+')?.click()');pause()
    try:
        until('document.querySelector(".host-badge")?.textContent==="本地桌面"')
        click('参数估计');click('选择音频目录')
        until('document.querySelectorAll(".m01-file-entry").length===48')
        js('document.querySelector(".m01-file-list .file-row").click()')
        until('document.querySelectorAll(".m01-tiers option").length===2')
        assert js('document.querySelectorAll(".m01-file-list .file-row").length')==48
        until('!!document.querySelector(".m01-page .wave-track")');pause(600)
        window.view.grab().save(str(out/'m01-natural.png'))
        checks.append('Actual Qt reads 48-file authorized corpus and renders real WAV/TextGrid')
        click('参数显示');click('选择音频目录')
        until('document.querySelectorAll(".m02-files button").length===48')
        js('document.querySelector(".m02-files button").click()')
        until('document.querySelectorAll(".m02-parameters input").length>10')
        check('.m02-parameters input');click('将 1 项分配到图窗');until('!!document.querySelector(".parameter-curve")')
        click('保存当前图');until('!document.querySelector(".parameter-figure button[disabled][title^=仅保存]")')
        end=time.monotonic()+15
        while not downloaded or not (out/downloaded[0]).exists():
            assert time.monotonic()<end;pause()
        assert downloaded[0].endswith('.png') and (out/downloaded[0]).read_bytes()[:8]==b'\x89PNG\r\n\x1a\n'
        click('清空选定图窗');until('document.querySelectorAll(".parameter-curve").length===0')
        click('删除选定图窗');until('document.querySelectorAll(".parameter-figure").length===0');click('新建图窗');until('!!document.querySelector(".empty-plot")')
        checks.append('Native parameter-table read, default PNG save dialog/output, clear/delete/new plot')
        selected_directory[0]=inputs
        click('选择音频目录');until('document.querySelectorAll(".m02-files button").length===1')
        start=time.monotonic();click('长录音-双声道.wav');until('!!document.querySelector(".m02-page .wave-track")&&document.querySelector(".m02-page [role=note]")?.textContent.includes("轻量预览")')
        report['m02_long_seconds']=time.monotonic()-start
        assert js('document.querySelector(".m02-page [role=note]").textContent.includes("2 声道")')
        check('.m02-page input[aria-label="显示两个声道"]')
        # Waveform's label wraps the checkbox rather than aria-label in some builds.
        js('[...document.querySelectorAll(".m02-page label")].find(e=>e.textContent.includes("显示两个声道"))?.querySelector("input")?.checked||[...document.querySelectorAll(".m02-page label")].find(e=>e.textContent.includes("显示两个声道"))?.querySelector("input")?.click()')
        until('document.querySelectorAll(".m02-page .wave-track").length===2')
        assert js('document.querySelector(".m02-toolbar").textContent.includes("650.000")')
        pause(600)
        window.view.grab().save(str(out/'m02-long.png'))
        js('(()=>{const inputs=document.querySelectorAll(".m02-toolbar input[type=number]");inputs[1].value="10";inputs[1].dispatchEvent(new Event("change",{bubbles:true}));})()');pause()
        js('(()=>{const input=document.querySelector(".m02-toolbar input[type=number]");input.value="640";input.dispatchEvent(new Event("change",{bubbles:true}));})()');pause()
        js('[...document.querySelectorAll(".m02-page label")].find(e=>e.textContent.includes("显示语谱图"))?.querySelector("input")?.click()')
        until('!!document.querySelector(".m02-page .spectrogram-canvas canvas")&&!document.querySelector(".m02-page .spectrogram-view [role=status]")')
        assert not js('document.querySelector(".spectrogram-view [role=alert]")?.textContent')
        assert js('Number(document.querySelector(".m02-page .wave-track svg").dataset.start)===640')
        js('document.querySelector(".m02-page .signal-panel").scrollTop=600')
        pause(600);window.view.grab().save(str(out/'m02-long-tail-spectrogram.png'))
        checks.append('Long stereo preview at 640–650 s produces actual Praat spectrogram through the shared local service')
        check('.m02-page .recursive-option input');until('document.querySelectorAll(".m02-files button").length===2')
        click('nested/short.wav');until('!document.querySelector(".m02-page [role=note]")&&!!document.querySelector(".m02-page .wave-track")')
        click('参数估计');click('选择音频目录')
        until('document.querySelectorAll(".m01-file-entry").length===1')
        js('document.querySelector(".m01-file-list .file-row").click()')
        until('document.querySelector(".m01-page [role=note]")?.textContent.includes("轻量预览")')
        assert js('document.querySelector(".signal-heading").textContent.includes("650.000")')
        check('.m01-page .recursive-option input');until('document.querySelectorAll(".m01-file-entry").length===2')
        # Decode is complete; inspect bounded preview metadata in the actual provider.
        _,(raw,sha,source_duration,note)=window.provider.preview_cache
        assert len(raw)<=64_000_000 and sha==long_sha and source_duration==duration
        report['preview_bytes']=len(raw);report['preview_note']=note
        assert hashlib.sha256(long.read_bytes()).hexdigest()==long_sha
        assert all(hashlib.sha256(p.read_bytes()).hexdigest()==sha for p,sha in original.items())
        pause(600)
        window.view.grab().save(str(out/'m01-long.png'))
        # The other module may replace the sole compact-byte cache. Refill the
        # explicitly granted preview when requesting this file's spectrogram.
        click('参数显示');until('!!document.querySelector(".m02-page .wave-track")')
        click('nested/short.wav');until('!document.querySelector(".m02-page [role=note]")')
        click('参数估计');until('!!document.querySelector(".m01-page .wave-track")')
        js('[...document.querySelectorAll(".m01-page label")].find(e=>e.textContent.includes("显示语谱图"))?.querySelector("input")?.click()')
        until('!!document.querySelector(".m01-page .spectrogram-canvas canvas")&&!document.querySelector(".m01-page .spectrogram-view [role=status]")')
        assert not js('document.querySelector(".m01-page .spectrogram-view [role=alert]")?.textContent')
        checks.append('M01 long spectrogram refills compact cache after M02 previews a different file')
        checks.append('249.6 MB / 650 s FLOAT32 stereo opens in both modules, bounded PCM preview, same source duration/hash, recursive scan and switch back')
        report['source_hashes_unchanged']=True;report['success']=True
    except Exception as exc:
        report['error']=str(exc);window.view.grab().save(str(out/'failed.png'));raise
    finally:
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf-8');print(out,flush=True)
        window.closing=True;window.close();app.processEvents()


if __name__=='__main__':main()
