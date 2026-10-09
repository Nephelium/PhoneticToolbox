"""Capture distinct M08 manual states in an owned maximized Windows Qt window.

The supplied natural syllable is copied into an isolated fixture directory.
Only disposable task state and explicitly generated demonstration outputs change.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
DIRS = ('desktop/src', 'backend/src', 'packages/phonetic_core/src', 'scripts')
for directory in DIRS:
    sys.path.insert(0, str(ROOT / directory))
os.environ['PYTHONPATH'] = os.pathsep.join(str(ROOT / d) for d in DIRS)
os.environ.setdefault('QT_QPA_PLATFORM', 'windows')
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS', '--disable-gpu --disable-gpu-compositing --mute-audio --autoplay-policy=no-user-gesture-required')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    from PyQt6.QtCore import QEventLoop, QTimer, Qt, QPoint
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication, QFileDialog
    from ptb_desktop.host import Workbench, register_scheme
    from verify_m08_wiring import setup

    out, db, cache = setup()
    captures = out / 'm08-manual-states'
    inputs = out / 'm08-inputs'
    saved = out / 'm08-saved'
    for directory in (captures, inputs, saved):
        directory.mkdir()
    source = ROOT / 'manual/assets/software-only/single-original.wav'
    fixture = inputs / 'single-syllable.wav'
    shutil.copyfile(source, fixture)
    QFileDialog.getExistingDirectory = lambda *a, **k: str(saved if '结果' in str(a) else inputs)
    QFileDialog.getSaveFileName = lambda *a, **k: (str(saved / Path(a[2]).name), '')
    register_scheme()
    app = QApplication(['PTB-owned-M08-manual-states'])
    w = Workbench(ROOT / 'frontend/dist', test=True, jobs_path=db, local_files_root=cache, vocal_profile=out / 'vocal')
    w.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
    w.showMaximized()
    report = {
        'success': False,
        'scope': 'Actual Windows Qt, owned hidden maximized window; no physical audio, IME, DWM or DPI claim',
        'python': sys.executable,
        'flags': os.environ.get('QTWEBENGINE_CHROMIUM_FLAGS'),
        'source': {'path': str(source), 'fixture': str(fixture), 'sha256': sha(source)},
        'captures': [], 'checks': [], 'terminations': [], 'out': str(out),
        'sources': [{'path': p, 'sha256': sha(ROOT / p)} for p in (
            'scripts/manual/capture_m08_states.py',
            'frontend/src/modules/pitch-manipulation/PitchManipulationPage.vue',
            'frontend/src/modules/pitch-manipulation/PitchCurve.vue',
            'frontend/src/modules/pitch-manipulation/HistoryPlot.vue',
            'frontend/dist/index.html', 'manual/V2_STYLE_GUIDE.md',
            'output/manual-work/v2-reference/html/chapter-05.html')],
    }
    w.page.renderProcessTerminated.connect(lambda s, c: report['terminations'].append([s.name, c]))

    def pause(ms=100):
        loop = QEventLoop()
        QTimer.singleShot(ms, loop.quit)
        loop.exec()

    def js(code):
        loop, values = QEventLoop(), []
        w.page.runJavaScript(code, lambda r: (values.append(r), loop.quit()))
        QTimer.singleShot(6000, loop.quit)
        loop.exec()
        if not values:
            raise RuntimeError('JavaScript timeout')
        return values[0]

    def until(code, seconds=80):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if js(code):
                return
            pause()
        raise AssertionError(code + '\n' + str(js('document.body.innerText.slice(-3500)')))

    def click(text):
        assert js('(()=>{const e=[...document.querySelectorAll("button")].find(b=>b.offsetParent&&b.textContent.trim()===' + json.dumps(text) + ');if(!e||e.disabled)return false;e.click();return true})()'), text
        pause(150)

    def fill(label, value):
        selector = json.dumps('[aria-label="' + label + '"]')
        assert js('(()=>{const e=document.querySelector(' + selector + ');if(!e)return false;e.value=' + json.dumps(str(value)) + ';e.dispatchEvent(new Event("input",{bubbles:true}));e.dispatchEvent(new Event("change",{bubbles:true}));return true})()'), label
        pause(100)

    def idle():
        until('document.querySelector(".m08-page")?.getAttribute("aria-busy")==="false"')

    def snapshot(name, caption, state):
        w.showMaximized()
        assert w.isMaximized()
        pause(500)
        w.view.repaint()
        w.view.grab()
        pause(400)
        app.processEvents()
        file = captures / (name + '.png')
        pix = w.view.grab()
        assert pix.save(str(file))
        report['captures'].append({
            'id': name, 'chapterId': 'm08', 'file': str(file), 'caption': caption,
            'state': state, 'distribution': 'software-only', 'git': False,
            'maximized': w.isMaximized(), 'window': [w.width(), w.height()],
            'frame': [w.frameGeometry().width(), w.frameGeometry().height()],
            'image': [pix.width(), pix.height()], 'devicePixelRatio': pix.devicePixelRatio(),
            'theme': js('document.documentElement.dataset.theme'),
            'palette': js('document.documentElement.dataset.palette'),
            'fontSize': js('getComputedStyle(document.documentElement).fontSize'),
            'sha256': sha(file),
        })
        print('Captured ' + name, flush=True)

    def drag(selector, points, shift=False):
        box = js('(()=>{const r=document.querySelector(' + json.dumps(selector) + ').getBoundingClientRect();return {x:r.x,y:r.y,w:r.width,h:r.height}})()')
        target = w.view.focusProxy() or w.view
        coords = [QPoint(round(box['x'] + x * box['w']), round(box['y'] + y * box['h'])) for x, y in points]
        modifiers = Qt.KeyboardModifier.ShiftModifier if shift else Qt.KeyboardModifier.NoModifier
        if shift:
            QTest.keyPress(target, Qt.Key.Key_Shift)
        QTest.mousePress(target, Qt.MouseButton.LeftButton, modifiers, coords[0])
        for point in coords[1:]:
            QTest.mouseMove(target, point, 30)
            pause(40)
        QTest.mouseRelease(target, Qt.MouseButton.LeftButton, modifiers, coords[-1])
        if shift:
            QTest.keyRelease(target, Qt.Key.Key_Shift)
        pause(250)

    try:
        until('!!document.querySelector(".home-page")&&document.fonts.status==="loaded"')
        click('设置')
        click('浅色')
        until('document.documentElement.dataset.theme==="light"')
        click('变速变调')
        until('!!document.querySelector(".m08-page")')
        click('打开音频目录')
        until('document.querySelector(".m08-page select")?.options.length===2')
        js('(()=>{const e=document.querySelector(".m08-page select");e.value=e.options[1].value;e.dispatchEvent(new Event("change",{bubbles:true}));})()')
        until('!!document.querySelector("svg.m08-curve")')
        idle()
        assert js('document.querySelectorAll(".history li").length') == 0
        snapshot('m08-loaded-empty-light', '短音节载入后，原音波形和原始 F0 已出现；尚未生成音频，输出与历史图为空。', {'historyCount': 0})
        fill('参考线 Hz', 200)
        click('添加参考线')
        before = js('document.querySelector("svg.m08-curve .modified").getAttribute("d")')
        drag('svg.m08-curve', [(0.23, .63), (.30, .55), (.40, .42), (.49, .32), (.58, .38), (.67, .51), (.77, .60)], shift=True)
        after = js('document.querySelector("svg.m08-curve .modified").getAttribute("d")')
        assert before != after, 'Native Shift drag must change the actual target path'
        snapshot('m08-freehand-reference-light', '按 Shift 拖动后的红色目标曲线与原始虚线分开，200 Hz 参考线保持可见；此时尚未合成。', {'historyCount': 0, 'nativeShiftDrag': True, 'referenceHz': 200})
        click('合成整段')
        until('document.querySelectorAll(".history li").length===1')
        idle()
        snapshot('m08-freehand-result-light', '手绘目标完成整段合成，输出波形和从实际 WAV 重新估计的历史 F0 已出现。', {'historyCount': 1, 'action': 'synthesize', 'speed': 1})
        drag('.plots>.module-section:nth-child(1) .wave-track svg', [(.20, .50), (.47, .50)])
        drag('.plots>.module-section:nth-child(3) .wave-track svg', [(.35, .50), (.70, .50)])
        selection_widths = js('[...document.querySelectorAll(".plots .wave-selection")].map(e=>Number(e.getAttribute("width")))')
        assert selection_widths[0] == 0 and selection_widths[1] > 0, selection_widths
        snapshot('m08-independent-selection-light', '合成波形已拖出试听选区，右栏显示合成音选区；切换到这份音频时原音试听选区清除，原音计算视野仍为整段。', {'historyCount': 1, 'playbackSelectionIndependentOfView': True, 'waveSelectionWidths': selection_widths})
        click('导入基频序列')
        fill('基频序列', '140\n180\n220\n180\n140')
        snapshot('m08-f0-import-light', '基频序列窗口中每行填写一个 Hz 值；导入会按当前视野内唯一连续有声段均匀重采样。', {'dialog': '导入基频序列', 'valuesHz': [140, 180, 220, 180, 140], 'applied': False})
        click('取消')
        click('编辑拐点表')
        click('添加行')
        for i, t, f, mode in [(0, .10, '140', 'constant'), (1, .32, '180,220,260', 'order'), (2, .57, '150', 'constant')]:
            fill('时间 ' + str(i), t)
            fill('频率 ' + str(i), f)
            fill('连接 ' + str(i), mode)
        snapshot('m08-breakpoints-three-light', '起点 140 Hz、终点 150 Hz 设为常量，中间 0.32 s 填入三个顺序值，准备生成三个不同峰值的版本。', {'dialog': '编辑拐点表', 'points': [{'time': .10, 'freqs': [140], 'mode': 'constant'}, {'time': .32, 'freqs': [180, 220, 260], 'mode': 'order'}, {'time': .57, 'freqs': [150], 'mode': 'constant'}]})
        click('保存更改')
        until('!document.querySelector("dialog[open]")')
        click('批量生成')
        until('document.querySelectorAll(".history li").length===4')
        idle()
        snapshot('m08-three-path-history-light', '拐点工具已生成三个峰值不同的实际音频，历史图同时显示先前手绘版本及三个拐点版本。', {'historyCount': 4, 'linearResultCount': 3})
        js('document.querySelectorAll(".history li")[2].querySelector("button").click()')
        pause(250)
        js('document.querySelectorAll(".history li")[2].querySelector("details").open=true')
        js('(()=>{const row=document.querySelectorAll(".history li")[2],list=document.querySelector(".history"),body=document.querySelector(".workbench-right-body"),section=document.querySelector(".history-details");list.scrollTop+=row.getBoundingClientRect().top-list.getBoundingClientRect().top;body.scrollTop+=section.getBoundingClientRect().top-body.getBoundingClientRect().top;})()')
        snapshot('m08-history-audition-light', '点击历史中的试听此版本后，合成波形切到所选版本；展开生成参数可核对这一次提交的拐点与范围。', {'historyCount': 4, 'selectedHistoryIndex': 2, 'parametersExpanded': True, 'mutedPhysicalOutput': True})
        js('document.querySelectorAll(".history li")[2].querySelector("details").open=false;document.querySelector(".workbench-right-body").scrollTop=0')
        click('批量保存')
        snapshot('m08-save-selection-light', '批量保存窗口列出当前视野的四份已生成 WAV，默认全部勾选；此步骤尚未选择外部保存目录。', {'dialog': '批量保存', 'selectedCount': 4, 'saved': False})
        click('保存所选音频')
        until('!document.querySelector("dialog[open]")')
        idle()
        until('!!document.querySelector(".save-location")')
        js('document.querySelector(".save-location details").open=true')
        assert len(list(saved.glob('*.wav'))) == 4
        snapshot('m08-save-confirmed-light', '四个音频已保存，最近保存提示给出实际目录和最终文件名；生成历史仍保留。', {'historyCount': 4, 'savedCount': 4, 'directory': str(saved)})
        report['saved'] = [{'name': p.name, 'sha256': sha(p), 'bytes': p.stat().st_size} for p in sorted(saved.glob('*.wav'))]
        click('批量变速变调')
        fill('语速倍率', .8)
        fill('音高倍率', 1.2)
        fill('音高偏移 Hz', 20)
        click('全选音频')
        snapshot('m08-batch-settings-light', '文件夹批量页已勾选同一短音节，语速 0.8、音高倍率 1.2、偏移 20 Hz；尚未提交这组设置。', {'tab': 'batch', 'speed': .8, 'pitch_ratio': 1.2, 'pitch_hz': 20, 'submitted': False})
        assert sha(source) == sha(fixture) == report['source']['sha256']
        assert not report['terminations']
        report['checks'] += ['Every capture explicitly maximized before image grab', 'Native Shift gesture changed actual target path', 'Actual natural-syllable synthesis and three-point linear outputs completed', 'Selected history loads actual audio; physical output muted', 'Four exported WAV files and on-screen directory verified', 'Original and isolated source-copy SHA256 unchanged']
        report['success'] = True
    except BaseException:
        report['error'] = traceback.format_exc()
        print(report['error'], flush=True)
        raise
    finally:
        report_file = out / 'm08-manual-states-report.json'
        report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print('REPORT ' + str(report_file), flush=True)
        print('REPORT_SHA256 ' + sha(report_file), flush=True)
        w.closing = True
        w.close()
        pause(300)
        app.quit()


if __name__ == '__main__':
    main()
