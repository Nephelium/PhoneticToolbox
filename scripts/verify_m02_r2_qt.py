"""M02-R2 built frontend, actual native XLSX/WAV/Praat on owned offscreen Qt, no DB."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS', '--disable-gpu --mute-audio')
import hashlib
import json
from pathlib import Path
import time
from uuid import uuid4
import numpy as np
from scipy.io import wavfile
from openpyxl import Workbook
from PyQt6.QtCore import QEventLoop, QTimer, Qt
from PyQt6.QtWidgets import QApplication, QFileDialog
from ptb_desktop.host import Workbench, register_scheme

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT/'output/validation/m02-r2'/('qt-'+uuid4().hex)
    inputs = out/'inputs'; inputs.mkdir(parents=True)
    rate = 8000
    wavfile.write(inputs/'display.wav', rate, (12000*np.sin(2*np.pi*200*np.arange(rate*4)/rate)).astype(np.int16))
    names = ['F0', 'small', 'Intensity'] + ['H'+str(i)+'-A3 (pF0)' for i in range(76)] + ['TextGrid']
    wb = Workbook(); sheet = wb.active; sheet.append(['Time_s', *names])
    for i in range(80):
        sheet.append([i/20, *[(('The photographer' if i < 40 else 'ɑ̃˥') if name == 'TextGrid' else 1 if name == 'small' else float(200+20*np.sin(i/20+j))) for j, name in enumerate(names)]])
    wb.save(inputs/'display.xlsx')
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.iterdir()}
    register_scheme(); app = QApplication(['M02-R2-owned-offscreen-QA'])
    window = Workbench(ROOT/'frontend/dist', test=True, vocal_profile=out/'vocal')
    window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    window.resize(1920, 1080); window.show()
    QFileDialog.getExistingDirectory = lambda *a, **k: str(inputs)
    QFileDialog.getSaveFileName = lambda *a, **k: (str(out/Path(a[2]).name), '')
    report = {'success': False, 'checks': [], 'layouts': [], 'schema_applied': [], 'scope': 'Windows actual Qt offscreen; public synthetic inputs and real native XLSX/WAV/Praat'}

    def pause(ms=150):
        loop = QEventLoop(); QTimer.singleShot(ms, loop.quit); loop.exec()

    def js(code):
        loop = QEventLoop(); result = []
        window.page.runJavaScript(code, lambda v: (result.append(v), loop.quit()))
        QTimer.singleShot(6000, loop.quit); loop.exec()
        assert result, 'JavaScript timeout'
        return result[0]

    def until(code, seconds=45):
        deadline = time.monotonic()+seconds
        while time.monotonic() < deadline:
            if js(code): return
            pause()
        raise AssertionError('UI timeout '+code+' / '+str(js('document.querySelector(".m02-page [role=alert]")?.textContent')))

    def click(text):
        js('[...document.querySelectorAll("button")].find(b=>b.offsetParent&&!b.disabled&&b.textContent.trim()==='+json.dumps(text, ensure_ascii=False)+')?.click()')
        pause()

    try:
        until('!!document.querySelector(".app-shell")')
        click('参数显示'); until('!!document.querySelector(".m02-page")')
        click('选择音频目录'); until('document.querySelectorAll(".m02-files button").length===1')
        click('display.wav'); until('document.querySelectorAll(".m02-parameters input").length===80')
        for width, columns in [(240, 2), (400, 3), (560, 4)]:
            js('(()=>{const h=document.querySelector(".m02-page [role=separator][aria-label=图窗与导出宽度]");h.dispatchEvent(new KeyboardEvent("keydown",{key:"Home",bubbles:true}));})()')
            for _ in range((width-240)//40):
                js('document.querySelector(".m02-page [role=separator][aria-label=图窗与导出宽度]").dispatchEvent(new KeyboardEvent("keydown",{key:"ArrowLeft",shiftKey:true,bubbles:true}))')
            pause(300)
            g = js('(()=>{const e=document.querySelector(".m02-parameters"),b=[...document.querySelectorAll(".m02-selection-actions button")].map(e=>e.getBoundingClientRect());return {columns:getComputedStyle(e).gridTemplateColumns.split(" ").length,height:e.clientHeight,total:e.scrollHeight,gap:b[1].top-b[0].bottom,aligned:[...e.querySelectorAll("label")].every(l=>{const a=l.querySelector("input").getBoundingClientRect(),b=l.querySelector("span").getBoundingClientRect();return Math.abs(a.top+a.bottom-b.top-b.bottom)<4})}})()')
            assert g['columns'] == columns and g['height'] <= 280 and g['total'] > g['height'] and g['aligned'] and 0 <= g['gap'] <= 10, g
            report['layouts'].append({'requested_width': width, **g})
        report['checks'].append('2/3/4 responsive columns, compact actions, aligned checkbox labels and independent list scrolling')
        for name in ['F0', 'Intensity', 'TextGrid']:
            js('[...document.querySelectorAll(".m02-parameters input")].find(e=>e.value==='+json.dumps(name)+')?.click()')
            pause()
        click('将 3 项分配到图窗'); until('document.querySelectorAll(".parameter-curve").length===2')
        js('[...document.querySelectorAll(".m02-page label")].find(e=>e.textContent.trim()==="显示语谱图（Praat）")?.querySelector("input")?.click()')
        until('!!document.querySelector(".m02-page .spectrogram-canvas canvas")&&!document.querySelector(".m02-page .spectrogram-view [role=status]")')
        geometry = '(()=>{const w=document.querySelector(".m02-page .wave-track>svg"),r=w.getBoundingClientRect(),c=document.querySelector(".parameter-chart"),b=c.getBoundingClientRect(),k=b.width/c.viewBox.baseVal.width,s=document.querySelector(".m02-page canvas").getBoundingClientRect(),m=w.querySelector("text").getScreenCTM();return {wave:[r.left,r.right],figure:[b.left+Number(c.dataset.plotLeft)*k,b.left+Number(c.dataset.plotRight)*k],spec:[s.left,s.right],glyph:[Math.hypot(m.a,m.b),Math.hypot(m.c,m.d)]}})()'
        for mode in ['light', 'dark']:
            click('设置'); click('浅色' if mode == 'light' else '深色'); click('参数显示')
            until('document.documentElement.dataset.theme==='+json.dumps(mode)); pause(200)
            g = js(geometry)
            assert all(abs(a-b) < 1.5 for pair in [g['figure'], g['spec']] for a, b in zip(g['wave'], pair)), g
            assert abs(g['glyph'][0]-g['glyph'][1]) < .01, g
            window.view.grab().save(str(out/(mode+'.png')))
            report['layouts'].append({'mode': mode, **g})
        report['checks'].append('real Praat preview and waveform/parameter time-axis alignment with isotropic English/IPA labels in both themes')
        # DOM pointer events on the real Qt-rendered surfaces supplement Chrome's native mouse tests.
        for selector, a, b in [('.parameter-chart', .7, .2), ('.m02-page canvas', .3, .6)]:
            js('(()=>{const e=document.querySelector('+json.dumps(selector)+');e.scrollIntoView();const r=e.getBoundingClientRect(),v=e.viewBox?.baseVal,left=v?Number(e.dataset.plotLeft)/v.width:0,right=v?Number(e.dataset.plotRight)/v.width:1,x=f=>r.left+r.width*(left+f*(right-left)),y=r.top+r.height*.4;e.setPointerCapture=()=>{};for(const [type,f] of [["pointerdown",'+str(a)+'],["pointermove",'+str(b)+'],["pointerup",'+str(b)+']])e.dispatchEvent(new PointerEvent(type,{button:0,pointerId:7,clientX:x(f),clientY:y,bubbles:true}));})()')
            pause(200)
            text = js('document.querySelector(".m02-toolbar>small").textContent')
            assert ('选区 '+f'{abs(b-a)*4:.3f}'+' s') in text, text
        report['checks'].append('parameter/real spectrogram pointer-event ranges update the shared seconds and selection overlay')
        click('保存整幅 PNG'); until('!document.querySelector(".parameter-figure button[title^=白底][disabled]")')
        deadline = time.monotonic()+15
        while not list(out.glob('*.png')) or not (out/'display-图窗 1.png').exists():
            assert time.monotonic() < deadline, 'PNG not saved'; pause()
        assert (out/'display-图窗 1.png').read_bytes()[:8] == b'\x89PNG\r\n\x1a\n'
        assert hashes == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.iterdir()}
        report['checks'].append('native save dialog/whole annotated PNG, input hashes preserved')
        report['success'] = True
    except Exception as error:
        report['error'] = str(error); window.view.grab().save(str(out/'failed.png')); raise
    finally:
        (out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(out, flush=True); window.closing = True; window.close(); app.processEvents()


if __name__ == '__main__': main()
