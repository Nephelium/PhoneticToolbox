"""M02 only: actual local table/Praat reads and native PNG/SVG saves, no DDL."""
import hashlib
import json
from pathlib import Path
import time
from uuid import uuid4
import numpy as np
from scipy.io import wavfile
import cv2
import struct
import zlib
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QFileDialog, QLineEdit
from ptb_desktop.host import register_scheme, Workbench
from verify_m02_m09_local import fixtures, DB

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT/'output/validation/m02-png'/('qt-'+uuid4().hex)
    inputs = out/'inputs'; inputs.mkdir(parents=True); fixtures(inputs)
    rate, audio = wavfile.read(inputs/'tone.wav')
    wavfile.write(inputs/'tone.wav', rate, np.column_stack([audio, audio//3]))
    originals = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.iterdir()}
    register_scheme(); app = QApplication(['M02 PNG verification'])
    window = Workbench(ROOT/'frontend/dist', test=True, jobs_path=DB)
    window.show()
    click = lambda text: "[...document.querySelectorAll('button')].find(b=>b.offsetParent&&b.textContent.trim()==="+json.dumps(text,ensure_ascii=False)+")?.click()"
    label = lambda text: "[...document.querySelectorAll('.m02-page label')].find(e=>e.textContent.includes("+json.dumps(text,ensure_ascii=False)+"))?.querySelector('input')?.click()"
    stages = [
        ('desktop', 'document.querySelector(".host-badge")?.textContent==="本地桌面"', click('参数显示'), None),
        ('module', '!!document.querySelector(".m02-page")', click('选择音频目录'), str(inputs)),
        ('files', 'document.querySelectorAll(".m02-files button").length===1', click('tone.wav'), None),
        ('empty by default', '!!document.querySelector(".empty-plot")&&document.querySelectorAll(".m02-parameters input").length===4&&document.querySelectorAll(".m02-parameters input:checked").length===0', click('全选可见参数'), None),
        ('select explicitly', 'document.querySelectorAll(".m02-parameters input:checked").length===3', click('将 3 项分配到图窗'), None),
        ('table', 'document.querySelectorAll(".parameter-curve").length===3', "document.documentElement.dataset.theme='light';"+click('保存整幅 PNG'), str(out/'light.png')),
        ('light saved', None, "document.documentElement.dataset.theme='dark';"+label('显示两个声道'), None),
        ('stereo', 'document.querySelectorAll(".m02-page .wave-track").length===2', click('保存整幅 PNG'), str(out/'dark-stereo.png')),
        ('stereo saved', None, label('显示语谱图（Praat）'), None),
        ('Praat ready', '!!document.querySelector(".m02-page .spectrogram-canvas canvas")&&!document.querySelector(".m02-page .spectrogram-view [role=status]")', click('保存整幅 PNG'), str(out/'spectrogram.png')),
        ('spectrogram saved', None, 'document.querySelector(".m02-page button[aria-label=放大波形]").click()', None),
        ('zoom ready', '!!document.querySelector(".m02-page .spectrogram-canvas canvas")&&!document.querySelector(".m02-page .spectrogram-view [role=status]")', click('保存整幅 PNG'), str(out/'zoom.png')),
        ('zoom saved', None, "(()=>{const e=document.querySelector('.image-format');e.value='svg';e.dispatchEvent(new Event('change',{bubbles:true}));})();"+click('保存当前图'), str(out/'parameters.svg')),
        ('SVG saved', None, None, None),
    ]
    waits = {4:'light.png', 6:'dark-stereo.png', 8:'spectrogram.png', 10:'zoom.png', 11:'parameters.svg'}
    report = {'success':False, 'stages':[], 'schema_applied':[]}
    index = 0; pending = False; dialog_path = None; started = time.monotonic()
    def finish(ok, error=None):
        timer.stop(); dialogs.stop(); report.update(success=ok,error=error)
        window.view.grab().save(str(out/'page.png')); window.closing=True; window.close()
    def received(ok):
        nonlocal index,pending,dialog_path
        pending=False
        if not ok:return
        if index in waits and not (out/waits[index]).exists():return
        name,_,action,dialog_path=stages[index];report['stages'].append(name);index+=1
        print(name,flush=True)
        if action:window.page.runJavaScript(action)
        else:finish(True)
    def tick():
        nonlocal pending
        if time.monotonic()-started>120:finish(False,'timeout '+str(index));return
        if pending or index>=len(stages):return
        pending=True;window.page.runJavaScript(stages[index][1] or 'true',received)
    def dialog_tick():
        nonlocal dialog_path
        dialog=app.activeModalWidget()
        if not dialog_path or not isinstance(dialog,QFileDialog):return
        edit=dialog.findChild(QLineEdit,'fileNameEdit')
        if edit:
            dialog.setDirectory(str(Path(dialog_path).parent));edit.setText(dialog_path)
            selected=dialog.selectedFiles()
            if selected and Path(selected[0]).absolute()==Path(dialog_path).absolute():dialog_path=None;dialog.accept()
    timer=QTimer();timer.timeout.connect(tick);timer.start(400)
    dialogs=QTimer();dialogs.timeout.connect(dialog_tick);dialogs.start(200)
    app.exec()
    if report['success']:
        results={}
        for name in ('light.png','dark-stereo.png','spectrogram.png','zoom.png'):
            raw=(out/name).read_bytes();offset=8;dpi=None
            while offset<len(raw):
                n=struct.unpack_from('>I',raw,offset)[0];kind=raw[offset+4:offset+8];data=raw[offset+8:offset+8+n]
                assert zlib.crc32(kind+data)==struct.unpack_from('>I',raw,offset+8+n)[0]
                if kind==b'pHYs':
                    x,y,unit=struct.unpack('>IIB',data);assert unit==1;dpi=(x*.0254,y*.0254)
                offset+=12+n
            assert dpi and abs(dpi[0]-300)<.01
            rgb=cv2.imread(str(out/name));assert rgb is not None
            h,w=rgb.shape[:2];assert w>1000 and h>1800
            assert tuple(rgb[0,0])==(255,255,255)
            assert np.count_nonzero(np.min(rgb[:800],axis=2)<200)>1000,'Blank waveform'
            assert np.count_nonzero(np.min(rgb[-1000:],axis=2)<200)>1000,'Blank parameters'
            results[name]={'size':[w,h],'dpi':dpi}
        assert results['dark-stereo.png']['size'][1]>results['light.png']['size'][1]
        assert results['spectrogram.png']['size'][1]>results['dark-stereo.png']['size'][1]
        assert '<svg' in (out/'parameters.svg').read_text('utf-8')
        assert originals=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.iterdir()}
        report.update(images=results,originals_unchanged=True,normal_close=window.closing)
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(out/'report.json',flush=True)
    return 0 if report['success'] else 1


if __name__=='__main__':raise SystemExit(main())
