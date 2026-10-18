"""Owned Qt regression for M09 short-audio visibility and source-pixel brushes."""
import json
import sqlite3
import subprocess
import sys
from contextlib import contextmanager
import time
from pathlib import Path

from verification_artifacts import verification_run, write_json
from workbench_source import configure

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def child_paths():
    out = Path(sys.argv[2])
    yield out, out / 'scratch'


def main():
    configure(ROOT)
    import cv2
    import numpy as np
    import soundfile as sf
    from PyQt6.QtCore import QEvent, QEventLoop, QPoint, QTimer, Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication, QFileDialog
    from PyQt6 import sip
    from ptb_desktop.host import Workbench, register_scheme
    from ptb_worker.local_acoustic_files import initialize_local_files

    register_scheme()
    app = QApplication(['M09-canvas-owned-QA'])
    app.setQuitOnLastWindowClosed(False)
    with child_paths() as (out, scratch):
        inputs = scratch / 'inputs'; inputs.mkdir()
        saved = scratch / 'saved'; saved.mkdir()
        cache = scratch / 'cache'; cache.mkdir(); initialize_local_files(cache)
        db = scratch / 'jobs.sqlite3'
        with sqlite3.connect((ROOT / 'output/validation/p06/local-state.sqlite3').as_uri() + '?mode=ro', uri=True) as source:
            assert source.execute("SELECT count(*) FROM jobs WHERE state IN ('queued','running','cancel_requested')").fetchone()[0] == 0
            with sqlite3.connect(db) as target: source.backup(target)
        t = np.arange(9217) / 16000
        original = np.column_stack([.15*np.sin(2*np.pi*400*t) + .08*np.sin(2*np.pi*1600*t),
                                    .12*np.sin(2*np.pi*1700*t)])
        sf.write(inputs / 'short-stereo.wav', original, 16000, subtype='FLOAT')
        cv2.imwrite(str(inputs / 'white.png'), np.full((401, 401), 255, np.uint8))
        report = dict(success=False, cases=[], layouts=[], scope='Windows source, actual Qt default GPU, synthetic inputs')
        window = None

        def pause(ms=100):
            loop = QEventLoop(); QTimer.singleShot(ms, loop.quit); loop.exec()

        def js(code):
            loop = QEventLoop(); box = []
            window.page.runJavaScript(code, lambda value: (box.append(value), loop.quit()))
            QTimer.singleShot(10000, loop.quit); loop.exec()
            assert box, 'JavaScript timeout'
            return box[0]

        def until(code, seconds=90):
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                if js('Boolean(' + code + ')'): return
                pause()
            raise AssertionError(code + '\n' + str(js('document.body.innerText.slice(-2500)')))

        def click(text):
            assert js("(()=>{const e=[...document.querySelectorAll('button')].find(e=>e.offsetParent&&(e.getAttribute('aria-label')||e.textContent.trim())===" + json.dumps(text) + ");if(!e||e.disabled)return false;e.click();return true})()"), text
            pause()

        def select(label, value):
            js("(()=>{const e=document.querySelector('select[aria-label=\"" + label + "\"]');e.value=" + json.dumps(str(value)) + ";e.dispatchEvent(new Event('change',{bubbles:true}));})()")

        def fill(label, value):
            js("(()=>{const e=document.querySelector('input[aria-label=\"" + label + "\"]');e.value=" + json.dumps(str(value)) + ";e.dispatchEvent(new Event('input',{bubbles:true}));e.dispatchEvent(new Event('change',{bubbles:true}));})()")

        canvas = 'canvas[aria-label="可缩放频谱画布"]'

        def geometry():
            return js("(()=>{const c=document.querySelector('" + canvas + "'),v=document.querySelector('.paint-viewport'),r=c.getBoundingClientRect();const d=c.getContext('2d').getImageData(0,0,c.width,c.height).data;let dark=0;for(let i=0;i<d.length;i+=4)if(d[i]<220)dark++;return {width:c.width,height:c.height,sourceWidth:Number(c.dataset.sourceWidth),sourceHeight:Number(c.dataset.sourceHeight),displayWidth:r.width,displayHeight:r.height,viewportWidth:v.clientWidth,viewportHeight:v.clientHeight,dark}})()")

        def fitted():
            until("document.querySelector('" + canvas + "')?.width>0&&!document.querySelector('.m09-page[aria-busy=true]')")
            g = geometry()
            assert g['displayWidth'] <= g['viewportWidth'] and g['displayHeight'] <= g['viewportHeight'], g
            return g

        def dot(x=.5, y=.5, inspect=False):
            js("document.querySelector('"+canvas+"').scrollIntoView({block:'center',inline:'nearest',behavior:'instant'})")
            pause(150)
            r = js("(()=>{const r=document.querySelector('"+canvas+"').getBoundingClientRect();return {x:r.x,y:r.y,w:r.width,h:r.height}})()")
            p = QPoint(round(r['x'] + x*r['w']), round(r['y'] + y*r['h']))
            target = window.view.focusProxy() or window.view
            QTest.mouseMove(target, p); QTest.mousePress(target, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, p); pause(30)
            cursor_width=None
            if inspect:
                QTest.mouseMove(target,p+QPoint(5,0)); pause()
                report['cursor_probe']=dict(geometry=r,hit=js('document.elementFromPoint('+str(p.x())+','+str(p.y())+')?.outerHTML.slice(0,300)'))
                cursor_width=js("document.querySelector('.brush-cursor')?.getBoundingClientRect().width")
            QTest.mouseRelease(target, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, p); pause()
            QTest.mouseMove(target, p + QPoint(1, 0)); pause(30); QTest.mouseMove(target, p); pause()
            return cursor_width

        try:
            window = Workbench(ROOT / 'frontend/dist', test=True, jobs_path=db,
                               local_files_root=cache, vocal_profile=scratch / 'vocal-profile', start_module='M09')
            window.resize(1440, 1000); window.show(); window.page.setAudioMuted(True)
            QFileDialog.getExistingDirectory = lambda *a, **k: str(saved if '结果' in str(a) else inputs)
            until("!!document.querySelector('.m09-page')")
            report['device_pixel_ratio']=js('window.devicePixelRatio')
            click('音频藏信息'); click('打开音频目录')
            g = fitted(); assert g['sourceWidth'] == 38 and g['sourceHeight'] == 513 and g['dark'] > 100, g
            initial_height=g['displayHeight']
            assert abs(initial_height-314)<2, g
            report['cases'].append(dict(name='automatic short-audio full spectrum', geometry=g))
            for fft in (512, 2048, 1024):
                js("(()=>{const e=[...document.querySelectorAll('.audio-controls label')].find(e=>e.textContent.includes('频谱精度')).querySelector('select');e.value='" + str(fft) + "';e.dispatchEvent(new Event('change',{bubbles:true}));})()")
                until("document.querySelector('" + canvas + "')?.dataset.sourceHeight==='" + str(fft//2+1) + "'" + "&&!document.querySelector('.m09-page[aria-busy=true]')")
                measured=fitted(); assert measured['dark']>0 and abs(measured['displayHeight']-initial_height)<1, measured
            js("(()=>{const e=[...document.querySelectorAll('.audio-controls label')].find(e=>e.textContent.includes('编辑声道')).querySelector('select');e.value='1';e.dispatchEvent(new Event('change',{bubbles:true}));})()")
            until("!document.querySelector('.m09-page[aria-busy=true]')&&!!document.querySelector('" + canvas + "')")
            report['cases'].append(dict(name='FFT and channel automatically reload', geometry=fitted()))
            click('100%'); g = geometry()
            assert abs(g['displayWidth'] - g['width']) < .1 and abs(g['displayHeight'] - g['height']) < .1, g
            fill('画笔直径（像素）', 12); dot(.5, .78)
            until("document.querySelector('.axis-note').textContent.includes('1 笔')")
            click('生成藏信息音频')
            until("!!document.querySelector('.comparison img')&&!document.querySelector('.m09-page[aria-busy=true]')")
            click('保存重建结果'); until("document.body.innerText.includes('已保存 4 个结果文件')")
            wav = next(saved.glob('*.wav')); actual, sr = sf.read(wav, always_2d=True)
            source_audio, _ = sf.read(inputs / 'short-stereo.wav', always_2d=True)
            assert sr == 16000 and actual.shape == source_audio.shape
            np.testing.assert_array_equal(actual[:, 0], source_audio[:, 0])
            assert not np.array_equal(actual[:, 1], source_audio[:, 1])
            metadata = json.loads(next(saved.glob('*.json')).read_text('utf8'))
            assert metadata['phase_method'] == 'original-stft-phase' and metadata['config']['channel'] == 1
            assert metadata['config']['strokes'][0]['size'] == 12 and metadata['config']['strokes'][0]['size_unit'] == 'px'
            report['cases'].append(dict(name='actual edit/export preserves original phase and other channel', changed_bins=metadata['changed_bins']))
            click('适合窗口')
            fill('语谱图频率上限（Hz）',3000); pause()
            assert fitted()['displayHeight']==initial_height
            assert js("document.querySelector('.axis-note').textContent.includes('1 笔')")
            fill('语谱图频率上限（Hz）',8000); pause()
            assert fitted()['displayHeight']==initial_height
            fill('语谱图频率上限（Hz）',5000); fill('语谱图高度（像素）',400); pause(250)
            assert abs(fitted()['displayHeight']-394)<2
            report['cases'].append(dict(name='frequency limit preserves strokes and frame; adjustable height'))
            for width, height in ((1440, 1000), (1100, 800)):
                window.resize(width, height); pause(250)
                for theme in ('light', 'dark'):
                    js('document.documentElement.dataset.theme=' + json.dumps(theme)); pause(200)
                    g = fitted(); assert g['dark'] > 0
                    name = f'audio-{width}-{theme}'
                    window.view.grab().save(str(out / (name + '.png')))
                    report['layouts'].append(dict(name=name, geometry=g))
            click('图片涂鸦重建'); click('选择图片目录')
            until("document.querySelector('select[aria-label=语谱图图片]').options.length>1")
            file_id = js("document.querySelector('select[aria-label=语谱图图片]').options[1].value")
            select('语谱图图片', file_id); until("!!document.querySelector('.image-frame img')")
            click('应用校正并绘图'); fitted(); click('100%')
            for diameter in (1, 12, 22, 200):
                fill('画笔直径（像素）', diameter)
                dot()
                until("document.querySelector('.axis-note').textContent.includes('1 笔')")
                bounds = js("(()=>{const c=document.querySelector('" + canvas + "'),d=c.getContext('2d').getImageData(0,0,c.width,c.height).data;let x0=c.width,x1=-1,y0=c.height,y1=-1;for(let y=0;y<c.height;y++)for(let x=0;x<c.width;x++)if(d[(y*c.width+x)*4]<255){x0=Math.min(x0,x);x1=Math.max(x1,x);y0=Math.min(y0,y);y1=Math.max(y1,y)}return {width:x1-x0+1,height:y1-y0+1}})()")
                assert bounds == dict(width=diameter, height=diameter), (diameter, bounds)
                report['cases'].append(dict(name='image source-pixel diameter', diameter=diameter, bounds=bounds))
                click('撤销笔迹')
            fill('画笔直径（像素）', 12); click('−'); cursor_width=dot(inspect=True)
            assert abs(cursor_width - 9.6) < .1, cursor_width
            click('撤销笔迹'); click('重做笔迹')
            report['cases'].append(dict(name='80 percent zoom cursor and undo/redo', cursor_width=cursor_width))
            click('音频藏信息'); assert fitted()['sourceWidth'] == 38
            report['cases'].append(dict(name='mode switch retains independent audio strokes'))
            report['success'] = True
        finally:
            write_json(out / 'report.json', report)
            if window is not None:
                js('window.onbeforeunload=null'); window.closing = True; window.close(); window.service.close(); pause(500); window.deleteLater()
                app.sendPostedEvents(None, QEvent.Type.DeferredDelete); pause(300)
            app.quit()
            sip.delete(app)
            print(out, flush=True)


if __name__ == '__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--child':
        main()
    else:
        with verification_run('m09-canvas','qt','16000 Hz / 9217-frame stereo harmonics, white 401x401 PNG; real Qt, fixed frame, frequency limit, pixel brush and export') as (out,scratch):
            result=subprocess.run([sys.executable,'-B','-X','utf8',str(Path(__file__).resolve()),'--child',str(out)],capture_output=True,text=True,encoding='utf-8',errors='replace',creationflags=subprocess.CREATE_NO_WINDOW)
            (out/'qt.log').write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
            print(out,flush=True)
            assert result.returncode==0, 'See qt.log'
            assert json.loads((out/'report.json').read_text('utf-8'))['success']
