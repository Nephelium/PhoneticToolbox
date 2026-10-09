"""P18 final built UI checks in an owned offscreen Qt host.

Natural input paths are supplied explicitly. No task database, microphones,
scientific jobs, existing browser profiles or original-file writes are used.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from uuid import uuid4

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS', '--disable-gpu --mute-audio')
from PyQt6.QtCore import QEventLoop, QTimer
from PyQt6.QtWidgets import QApplication, QFileDialog
from ptb_desktop.host import Workbench, register_scheme

ROOT = Path(__file__).resolve().parents[1]
MODULES = [
    ('M01', '参数估计'), ('M02', '参数显示'), ('M03', 'EGG 信号分析'),
    ('M04', 'LPC 谱图'), ('M05', '唇形提取'), ('M06', '语音合成'),
    ('M07', '发声类型合成'), ('M08', '变速变调'), ('M09', '语谱图转音频'),
    ('M11', 'MFA 自动标注'), ('M12', 'TextGrid标注'), ('M13', '汉字转国际音标'),
    ('M14', '音系归纳'), ('M15', '感知实验'), ('M16', '录音'),
]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--audio', required=True)
    args = parser.parse_args()
    corpus = args.corpus.resolve(strict=True)
    audio = (corpus / args.audio).resolve(strict=True)
    assert audio.parent == corpus and audio.suffix.lower() == '.wav'
    originals = {p: digest(p) for p in corpus.iterdir() if p.is_file()}
    out = ROOT / 'output/validation/p18' / ('qt-' + uuid4().hex)
    out.mkdir(parents=True)
    register_scheme()
    app = QApplication(['P18-owned-visual-QA'])
    window = Workbench(ROOT / 'frontend/dist', test=True, vocal_profile=out / 'vocal')
    QFileDialog.getExistingDirectory = lambda *a, **k: str(corpus)
    window.show()
    window.resize(1920, 1080)
    report = {'success': False, 'platform': 'Windows actual Qt offscreen',
              'scope': 'built UI, natural WAV previews; no physical DPI/devices or scientific task validation',
              'layouts': [], 'checks': [], 'schema_applied': []}

    def pause(ms=120):
        loop = QEventLoop()
        QTimer.singleShot(ms, loop.quit)
        loop.exec()

    def js(code):
        loop = QEventLoop()
        values = []
        window.page.runJavaScript(code, lambda value: (values.append(value), loop.quit()))
        QTimer.singleShot(8000, loop.quit)
        loop.exec()
        return values[0] if values else None

    def until(code, seconds=45):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if js(code):
                return
            pause()
        raise AssertionError(code + '\n' + str(js('document.body.innerText.slice(-3000)')))

    def click(label, selector='button'):
        expr = '[...document.querySelectorAll(' + json.dumps(selector) + ')].find(e=>e.offsetParent && e.textContent.trim()===' + json.dumps(label) + ')'
        until('!!(' + expr + ') && !(' + expr + ').disabled')
        js(expr + '.click()')
        pause()

    def nav(mid, title):
        expr = '[...document.querySelectorAll(".nav-item")].find(e=>e.title===' + json.dumps(title) + ')'
        until('!!(' + expr + ')')
        js(expr + '.click()')
        until('document.querySelector("#tab-' + mid + '")?.getAttribute("aria-selected")==="true"')
        until('[...document.querySelectorAll(".module-frame")].some(e=>e.offsetParent)')
        pause(350)

    def shot(name):
        window.view.grab()
        pause(350)
        assert window.view.grab().save(str(out / (name + '.png')))

    def geometry():
        return js('''(()=>{
          const visible=e=>e.offsetParent!==null;
          const frame=[...document.querySelectorAll('.module-frame')].find(visible);
          const box=e=>{const r=e.getBoundingClientRect();return {x:r.x,y:r.y,width:r.width,height:r.height,bottom:r.bottom,scrollWidth:e.scrollWidth,clientWidth:e.clientWidth}};
          const panes={};for(const c of ['.workbench-left','.workbench-center','.workbench-right','.annotation-files','.annotation-editor','.annotation-settings','.recording-list','.recording-main']){const e=frame?.querySelector(c);if(e&&visible(e))panes[c]=box(e);}
          return {viewport:[innerWidth,innerHeight],documentWidth:document.documentElement.scrollWidth,frame:frame?box(frame):null,panes};
        })()''')

    try:
        until('document.querySelector(".host-badge")?.textContent==="本地桌面"')
        # Load every module before measuring, so async fonts and lazy components settle.
        for mid, title in MODULES:
            nav(mid, title)
        until('document.fonts.status==="loaded"')
        nav('M11', 'MFA 自动标注')
        assert js('[...document.querySelectorAll("label")].filter(e=>e.offsetParent && e.textContent.trim()==="Beam").length') == 1
        assert js('[...document.querySelectorAll("label")].filter(e=>e.offsetParent && e.textContent.trim()==="Retry beam").length') == 1
        nav('M03', 'EGG 信号分析')
        assert js('document.querySelector(".egg-page .workbench-left")!==null && document.querySelector(".egg-page .workbench-right")===null')
        report['checks'].append('MFA alignment parameters visible once outside runtime details; EGG has a left operation pane')
        for width, height in [(1920, 1080), (1366, 768)]:
            window.resize(width, height)
            pause(350)
            for theme in ['light', 'dark']:
                js('document.documentElement.dataset.theme=' + json.dumps(theme))
                for mid, title in MODULES:
                    nav(mid, title)
                    g = geometry()
                    assert g and g['frame'], (mid, g)
                    assert g['documentWidth'] <= g['viewport'][0] + 1, (mid, theme, g)
                    assert g['frame']['width'] > 400 and g['frame']['height'] > 250, (mid, g)
                    if width == 1920 and '.workbench-left' in g['panes']:
                        left = g['panes']['.workbench-left']
                        center = g['panes']['.workbench-center']
                        assert left['x'] < center['x'], (mid, g)
                        assert abs(left['width'] - 300) <= 1, (mid, g)
                        if '.workbench-right' in g['panes']:
                            right = g['panes']['.workbench-right']
                            assert abs(right['width'] - 300) <= 1, (mid, g)
                            assert abs(left['bottom'] - right['bottom']) <= 1, (mid, g)
                    report['layouts'].append({'module': mid, 'theme': theme, **g})
                    if width == 1920 and mid in ['M01', 'M03', 'M06', 'M12', 'M16']:
                        shot(mid + '-empty-' + theme)
        report['checks'].append('15 pages x 2 themes x 2 sizes: built Qt frame geometry and unified-column checks')
        window.resize(1920, 1080)
        js('document.documentElement.dataset.theme="dark"')
        nav('M01', '参数估计')
        click('选择音频目录')
        until('document.querySelectorAll(".m01-file-list .file-row").length>0')
        expr = '[...document.querySelectorAll(".m01-file-list .file-row")].find(e=>e.textContent.includes(' + json.dumps(args.audio) + '))'
        until('!!(' + expr + ')')
        js(expr + '.click()')
        until('!!document.querySelector(".m01-page .wave-track svg")')
        assert '44100 Hz' in js('document.querySelector(".m01-page .signal-heading").innerText')
        shot('M01-natural-dark')
        report['checks'].append('M01 native directory grant and natural WAV waveform preview')
        nav('M02', '参数显示')
        click('选择音频目录')
        expr = '[...document.querySelectorAll(".m02-files button")].find(e=>e.textContent.trim()===' + json.dumps(args.audio) + ')'
        until('!!(' + expr + ')')
        js(expr + '.click()')
        until('!!document.querySelector(".m02-page .wave-track svg")')
        shot('M02-natural-dark')
        report['checks'].append('M02 same real WAV preview, no fabricated parameter curve')
        nav('M04', 'LPC 谱图')
        click('打开 WAV 目录')
        expr = 'document.querySelector("select[aria-label=\\"LPC 音频文件\\"]")'
        until('[...(' + expr + ').options].some(e=>e.textContent===' + json.dumps(args.audio) + ')')
        js('(()=>{const e=' + expr + ';e.value=[...e.options].find(o=>o.textContent===' + json.dumps(args.audio) + ').value;e.dispatchEvent(new Event("change",{bubbles:true}));})()')
        until('!!document.querySelector(".lpc-page .wave-track svg")')
        shot('M04-natural-dark')
        report['checks'].append('M04 native WAV load and source waveform preview without running analysis')
        assert {p: digest(p) for p in originals} == originals
        report['originals'] = {'files_checked': len(originals), 'unchanged': True,
                               'audio_name': audio.name, 'audio_sha256': originals[audio]}
        report['success'] = True
    finally:
        (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), 'utf8')
        window.closing = True
        window.close()
        window.page.deleteLater()
        app.processEvents()
        print(json.dumps({'output': str(out), 'success': report['success'], 'layouts': len(report['layouts']), 'checks': len(report['checks'])}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
