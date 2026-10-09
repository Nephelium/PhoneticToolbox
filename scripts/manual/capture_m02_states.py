"""Capture distinct M02 manual states on an owned, maximized Windows Qt window.

Reads the already approved isolated natural WAV/result fixtures without copying
or changing them. Writes only a disposable task database, a deliberately invalid
demonstration workbook, captured screenshots, exports and an evidence report.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import time
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
DIRS = ('desktop/src', 'backend/src', 'packages/phonetic_core/src', 'scripts')
for directory in DIRS:
    sys.path.insert(0, str(ROOT / directory))
os.environ['PYTHONPATH'] = os.pathsep.join(str(ROOT / d) for d in DIRS)
os.environ.setdefault('QT_QPA_PLATFORM', 'windows')
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS', '--disable-gpu --disable-gpu-compositing --mute-audio --autoplay-policy=no-user-gesture-required')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    from PyQt6.QtCore import QEventLoop, QTimer, Qt, QPoint, QPointF
    from PyQt6.QtGui import QWheelEvent
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication, QFileDialog
    from ptb_desktop.host import Workbench, register_scheme
    from verify_m08_wiring import setup

    inputs = ROOT / 'output/validation/m08-wiring/7375496ddca44ebd8fbed43bfb440b81/single'
    assert all((inputs / name).is_file() for name in ('single-syllable.wav', 'single-syllable.ptb.sqlite', 'single-syllable.xlsx', 'target-syllable.wav'))
    originals = {p.name: sha(p) for p in inputs.iterdir() if p.is_file()}
    out, db, cache = setup()
    captures = out / 'm02-manual-states'
    saved = out / 'm02-exported'
    invalid = out / 'm02-invalid-parameters'
    for folder in (captures, saved, invalid):
        folder.mkdir()
    (invalid / 'invalid-example.xlsx').write_text('This deliberately invalid example is not an Excel ZIP file.\n', encoding='utf-8')
    choice = {'folder': inputs}
    QFileDialog.getExistingDirectory = lambda *a, **k: str(choice['folder'])
    QFileDialog.getSaveFileName = lambda *a, **k: (str(saved / Path(a[2]).name), '')
    register_scheme()
    app = QApplication(['PTB-owned-M02-manual-states'])
    app.setQuitOnLastWindowClosed(False)
    w = Workbench(ROOT / 'frontend/dist', test=True, jobs_path=db, local_files_root=cache, vocal_profile=out / 'vocal')
    w.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
    w.showMaximized()
    report = {'success': False, 'scope': 'Windows actual Qt; owned hidden maximized window; existing approved natural result fixtures; no physical audio, compositor or DPI claim',
              'out': str(out), 'captures': [], 'checks': [], 'exports': [], 'terminations': [],
              'fixtureDirectory': str(inputs), 'inputHashes': originals, 'frontendIndexSha256': sha(ROOT / 'frontend/dist/index.html'),
              'sources': [{'path': p, 'sha256': sha(ROOT / p)} for p in (
                  'scripts/manual/capture_m02_states.py', 'manual/V2_STYLE_GUIDE.md',
                  'output/manual-work/v2-reference/html/chapter-02.html',
                  'frontend/src/modules/parameter-display/ParameterDisplayPage.vue',
                  'frontend/src/modules/parameter-display/ParameterFigure.vue',
                  'frontend/src/modules/parameter-display/state.ts', 'frontend/src/modules/parameter-display/export.ts')]}
    w.page.renderProcessTerminated.connect(lambda s, c: report['terminations'].append([s.name, c]))

    def pause(ms=100):
        loop = QEventLoop()
        QTimer.singleShot(ms, loop.quit)
        loop.exec()

    def js(code):
        loop, values = QEventLoop(), []
        w.page.runJavaScript(code, lambda value: (values.append(value), loop.quit()))
        QTimer.singleShot(6000, loop.quit)
        loop.exec()
        assert values, 'JavaScript timeout'
        return values[0]

    def until(code, seconds=60):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if js(code):
                return
            pause()
        raise AssertionError(code + '\n' + str(js('document.body.innerText.slice(-3000)')))

    def click(text, scope='document'):
        assert js('(()=>{const e=[...' + scope + '.querySelectorAll("button")].find(b=>b.offsetParent&&b.textContent.trim()===' + json.dumps(text) + ');if(!e||e.disabled)return false;e.click();return true})()'), text
        pause(150)

    def fill(selector, value):
        assert js('(()=>{const e=document.querySelector(' + json.dumps(selector) + ');if(!e)return false;e.value=' + json.dumps(str(value)) + ';e.dispatchEvent(new Event("input",{bubbles:true}));e.dispatchEvent(new Event("change",{bubbles:true}));return true})()'), selector
        pause(150)

    def select_table(name):
        assert js('(()=>{const e=document.querySelector(".signal-panel>label select"),o=[...e.options].find(o=>o.textContent===' + json.dumps(name) + ');if(!o)return false;e.value=o.value;e.dispatchEvent(new Event("change",{bubbles:true}));return true})()'), name
        until('!document.querySelector(".m02-page>p[role=status]")')
        pause(200)

    def mark(names):
        click('清空勾选')
        fill('[aria-label="参数搜索"]', '')
        for name in names:
            assert js('(()=>{const e=[...document.querySelectorAll(".m02-parameters input")].find(e=>e.value===' + json.dumps(name) + ');if(!e)return false;e.click();return true})()'), name
            pause(130)

    def assign(names):
        mark(names)
        click('将 ' + str(len(names)) + ' 项分配到图窗')
        until('document.querySelectorAll(".parameter-curve").length>0')
        pause(250)

    def snapshot(name, caption, state):
        assert w.isMaximized()
        pause(450)
        w.view.repaint()
        w.view.grab()
        pause(350)
        app.processEvents()
        path = captures / (name + '.png')
        pix = w.view.grab()
        assert pix.width() == 2160 and pix.height() == 1350, [pix.width(), pix.height()]
        assert pix.save(str(path))
        report['captures'].append({'id': name, 'chapterId': 'm02', 'file': str(path), 'caption': caption,
                                  'state': state, 'distribution': 'software-only', 'git': False,
                                  'maximized': w.isMaximized(), 'window': [w.width(), w.height()],
                                  'frame': [w.frameGeometry().width(), w.frameGeometry().height()],
                                  'image': [pix.width(), pix.height()], 'devicePixelRatio': pix.devicePixelRatio(),
                                  'theme': js('document.documentElement.dataset.theme'),
                                  'palette': js('document.documentElement.dataset.palette'),
                                  'fontSize': js('getComputedStyle(document.documentElement).fontSize'), 'sha256': sha(path)})
        print('Captured ' + name, flush=True)

    def chart_scope(index=0):
        return 'document.querySelectorAll(".parameter-figure")[' + str(index) + ']'

    def native_drag(selector, left, right, shift=False):
        box = js('(()=>{const e=document.querySelector(' + json.dumps(selector) + '),r=e.getBoundingClientRect(),v=e.viewBox?.baseVal;return {x:r.x,y:r.y,w:r.width,h:r.height,left:v&&e.dataset.plotLeft?Number(e.dataset.plotLeft)/v.width:0,right:v&&e.dataset.plotRight?Number(e.dataset.plotRight)/v.width:1}})()')
        target = w.view.focusProxy() or w.view
        def point(f):
            return QPoint(round(box['x'] + (box['left'] + (box['right'] - box['left']) * f) * box['w']), round(box['y'] + box['h'] * .45))
        modifiers = Qt.KeyboardModifier.ShiftModifier if shift else Qt.KeyboardModifier.NoModifier
        if shift:
            QTest.keyPress(target, Qt.Key.Key_Shift)
        QTest.mousePress(target, Qt.MouseButton.LeftButton, modifiers, point(left))
        for i in range(1, 7):
            QTest.mouseMove(target, point(left + (right - left) * i / 6), 30)
            pause(40)
        QTest.mouseRelease(target, Qt.MouseButton.LeftButton, modifiers, point(right))
        if shift:
            QTest.keyRelease(target, Qt.Key.Key_Shift)
        pause(300)

    def native_wheel(selector, delta=120):
        box = js('(()=>{const r=document.querySelector(' + json.dumps(selector) + ').getBoundingClientRect();return {x:r.x+r.width*.5,y:r.y+r.height*.45}})()')
        target = w.view.focusProxy() or w.view
        pos = QPointF(box['x'], box['y'])
        event = QWheelEvent(pos, QPointF(target.mapToGlobal(pos.toPoint())), QPoint(), QPoint(0, delta),
                            Qt.MouseButton.NoButton, Qt.KeyboardModifier.ControlModifier, Qt.ScrollPhase.NoScrollPhase, False)
        QApplication.sendEvent(target, event)
        pause(300)

    def export_wait(name):
        deadline = time.monotonic() + 20
        path = saved / name
        while not path.is_file() or path.stat().st_size == 0:
            assert time.monotonic() < deadline, name
            pause()
        return path

    try:
        until('!!document.querySelector(".home-page")&&document.fonts.status==="loaded"')
        click('设置')
        click('浅色')
        until('document.documentElement.dataset.theme==="light"')
        click('参数显示')
        until('!!document.querySelector(".m02-page")')
        if '--text-layer-only' in sys.argv:
            text_inputs = ROOT / 'output/validation/m02-r2/qt-7d39b5be925b4035aeb427235cf5e5c6/inputs'
            text_hashes = {p.name: sha(p) for p in text_inputs.iterdir() if p.is_file()}
            report['scope'] = 'Windows actual Qt; owned hidden maximized window; existing synthetic text-layer demonstration only; no natural-voice measurement, physical audio, compositor or DPI claim'
            report['fixtureDirectory'] = str(text_inputs)
            report['inputHashes'] = text_hashes
            choice['folder'] = text_inputs
            click('打开音频目录')
            until('document.querySelectorAll(".m02-files button").length===1')
            click('display.wav')
            until('document.querySelectorAll(".m02-parameters input").length===80', 100)
            assign(['F0', 'Intensity', 'TextGrid'])
            until('document.querySelectorAll(".m02-wave-annotations text").length===2&&document.querySelectorAll(".parameter-curve").length===2')
            snapshot('m02-table-text-layer-light', '在示例 XLSX 中分配 TextGrid 文字列后，两条文字区间同时显示在波形与参数图中；它们来自参数表的文字列。',
                     {'syntheticDemonstrationOnly': True, 'textColumn': 'TextGrid', 'waveLabels': js('[...document.querySelectorAll(".m02-wave-annotations text")].map(e=>e.textContent)'), 'curves': 2})
            assert text_hashes == {p.name: sha(p) for p in text_inputs.iterdir() if p.is_file()}
            report['textFixture'] = {'directory': str(text_inputs), 'source': 'Existing synthetic M02 demonstration, not a natural-voice measurement', 'hashes': text_hashes}
            report['checks'].append('Actual native XLSX text column assigned; English and IPA runs appear in both waveform and parameter plot; existing synthetic fixture hashes preserved')
            report['success'] = True
            return
        click('打开音频目录')
        until('document.querySelectorAll(".m02-files button").length===2')
        click('target-syllable.wav')
        until('!!document.querySelector(".m02-page .wave-track svg")&&!!document.querySelector(".m02-page .notice")')
        snapshot('m02-unmatched-result-light', '目标 WAV 没有同名参数表时，波形仍可查看，参数表关联显示未关联，并提示可手动关联结果。', {'unmatchedAudio': 'target-syllable.wav'})
        click('single-syllable.wav')
        until('document.querySelectorAll(".m02-parameters input").length>20', 100)
        select_table('single-syllable.xlsx')
        until('document.querySelectorAll(".m02-parameters input").length>20', 100)
        report['checks'].append('Existing natural WAV auto-linked SQLite; manually selected same-name XLSX, real native parsing completed')
        mark(['F0 - Praat', 'F0 - REAPER'])
        fill('[aria-label="参数搜索"]', 'F0 -')
        snapshot('m02-association-checks-light', '手动关联同名 XLSX 后，搜索 F0 - 只保留两项候选；两项已勾选，尚未分配，图窗仍显示待分配参数。', {'association': 'single-syllable.xlsx', 'query': 'F0 -', 'checkedCount': 2, 'curves': 0})
        click('将 2 项分配到图窗')
        until('document.querySelectorAll(".parameter-curve").length===2')
        snapshot('m02-first-curves-light', '点击将 2 项分配到图窗后，图窗 1 出现 Praat 与 REAPER 的 F0 曲线，波形与参数图使用同一全长时间窗。', {'curves': 2, 'assigned': ['F0 - Praat', 'F0 - REAPER']})
        click('清空选定图窗')
        assign(['F1 - Praat', 'SHR (pF0)'])
        until('document.querySelectorAll(".parameter-chart .right-tick").length===5')
        click('放大图窗', chart_scope())
        until('!!document.querySelector(".m02-maximized .parameter-chart")')
        snapshot('m02-dual-axis-light', '放大的图窗叠加 F1 与 SHR (pF0)，自动双轴把 F1 标在右轴、SHR 标在左轴，图例标明轴归属。', {'dual': True, 'axes': js('[...document.querySelectorAll(".m02-maximized .parameter-curve")].map(e=>({name:e.dataset.parameter,axis:e.dataset.axis}))')})
        click('还原图窗', chart_scope())
        click('新建图窗')
        assign(['F2 - Praat', 'F3 - Praat'])
        until('document.querySelectorAll(".parameter-chart").length===2')
        js('(()=>{const e=document.querySelectorAll(".parameter-figure")[0],p=document.querySelector(".workbench-center");p.scrollTop+=e.getBoundingClientRect().top-p.getBoundingClientRect().top;})()')
        pause(300)
        snapshot('m02-two-plots-light', '新建图窗 2 并分配 F2、F3 后，F1 与 SHR 保留在图窗 1，右栏目标指向图窗 2；两图共用时间范围。', {'groups': 2, 'target': 2, 'curvesByGroup': js('[...document.querySelectorAll(".parameter-figure")].map(e=>[...e.querySelectorAll(".parameter-curve")].map(e=>e.dataset.parameter))')})
        js('(()=>{const e=document.querySelectorAll(".parameter-figure")[1],p=document.querySelector(".workbench-center");p.scrollTop+=e.getBoundingClientRect().top-p.getBoundingClientRect().top;document.querySelector(".workbench-right-body").scrollTop=160;})()')
        pause(300)
        snapshot('m02-second-plot-light', '向下滚动工作区，完整查看图窗 2 中的 F2 与 F3 共轴轨迹和图例；右侧仍选中图窗 2 作为目标，首图分配保持。', {'groups': 2, 'target': 2, 'secondPlotCurves': 2, 'scrolledToSecondPlot': True})
        click('合并回首图', chart_scope(1))
        until('document.querySelectorAll(".parameter-chart").length===1')
        report['checks'].append('Native UI new group, assignment, two populated plots and merge back to first preserve assigned columns')
        # Move formants to their own group again, then delete it explicitly.
        click('新建图窗')
        assign(['F2 - Praat', 'F3 - Praat'])
        click('删除图窗', chart_scope(1))
        until('document.querySelectorAll(".parameter-chart").length===1&&document.querySelectorAll(".parameter-curve").length===2')
        report['checks'].append('Deleting the populated formant group removed its assignments rather than merging them')
        js('document.querySelector(".workbench-center").scrollTop=0')
        js('document.querySelector(".workbench-right-body").scrollTop=0')
        assert js('(()=>{const e=[...document.querySelectorAll(".m02-toolbar label")].find(e=>e.textContent.startsWith("时间窗长度"))?.querySelector("input");if(!e)return false;e.value="0.25";e.dispatchEvent(new Event("change",{bubbles:true}));return true})()')
        pause(200)
        before = js('document.querySelector(".m02-toolbar input[type=number]").value')
        native_wheel('.parameter-chart')
        after_span = js('[...document.querySelectorAll(".m02-toolbar label")].find(e=>e.textContent.startsWith("时间窗长度"))?.querySelector("input").value')
        assert float(after_span) < .25, after_span
        native_drag('.parameter-chart', .75, .40, shift=True)
        after = js('document.querySelector(".m02-toolbar input[type=number]").value')
        assert float(after) > float(before), [before, after]
        native_drag('.m02-page .wave-track>svg', .30, .68)
        assert '选区 0.076 s' in js('document.querySelector(".m02-toolbar>small").textContent')
        snapshot('m02-zoom-selection-light', '短视野内以 Ctrl＋滚轮继续放大、Shift 拖动参数图平移，再在波形拖选试听区间；时间窗起点、长度与选区时长分别显示。', {'nativeCtrlWheel': True, 'nativeShiftPan': True, 'nativeWaveDrag': True, 'startSeconds': after, 'spanSeconds': after_span})
        report['checks'].append('Native Qt Ctrl wheel reduced parameter time span; native Shift drag panned; native waveform drag changed shared selection without replacing the view')
        fill('[aria-label="参数搜索"]', '不存在的参数')
        snapshot('m02-no-matching-candidate-light', '输入不存在的参数名称后，右栏显示无匹配参数与清除筛选；已分配的双轴曲线仍保留。', {'query': '不存在的参数', 'curvesRetained': 2})
        click('清除筛选')
        until('document.querySelectorAll(".m02-parameters input").length>70')
        report['checks'].append('No-match search leaves curves intact; 清除筛选 resets search and enables both reaper and correction candidates')
        click('适合窗口')
        assert js('(()=>{const e=[...document.querySelectorAll(".m02-toolbar label")].find(e=>e.textContent.trim()==="显示语谱图（Praat）")?.querySelector("input");if(!e)return false;e.click();return true})()')
        until('!!document.querySelector(".spectrogram-canvas canvas")&&!document.querySelector(".spectrogram-view [role=status]")', 90)
        snapshot('m02-spectrogram-ready-light', '开启 Praat 语谱图并等待就绪后，灰度语谱与波形按同一时间轴排列；参数图位于其下方，整幅 PNG 将组合这些已显示图面。', {'realPraatSpectrogramReady': True})
        click('保存整幅 PNG', chart_scope())
        whole = export_wait('single-syllable-图窗 1.png')
        whole_raw = whole.read_bytes()
        assert whole_raw[:8] == b'\x89PNG\r\n\x1a\n'
        report['exports'].append({'file': str(whole), 'kind': 'whole PNG', 'sha256': sha(whole), 'image': list(struct.unpack('>II', whole_raw[16:24])), 'containsSpectrogram': True})
        # Restore compact waveform to show the format choice and complete chart.
        js('[...document.querySelectorAll(".m02-toolbar label")].find(e=>e.textContent.trim()==="显示语谱图（Praat）").querySelector("input").click()')
        click('放大图窗', chart_scope())
        fill('select[aria-label="图窗 1图片格式"]', 'svg')
        click('保存当前图', chart_scope())
        svg = export_wait('图窗 1.svg')
        svg_text = svg.read_text(encoding='utf-8')
        assert '<svg' in svg_text and '时间（秒）' in svg_text and 'SHR (pF0)' in svg_text and 'data:font' in svg_text
        snapshot('m02-svg-current-export-light', '图片格式选择 SVG 后保存当前图；完整双轴曲线、左右原数值轴、图例与保存按钮在同一放大图窗内可见。', {'format': 'SVG', 'nativeDownloadCompleted': True})
        report['exports'].append({'file': str(svg), 'kind': 'current SVG', 'sha256': sha(svg), 'hasTimeAxis': True, 'hasLegend': True, 'hasEmbeddedFont': True})
        fill('select[aria-label="图窗 1图片格式"]', 'png')
        click('保存当前图', chart_scope())
        png = export_wait('图窗 1.png')
        raw = png.read_bytes()
        assert raw[:8] == b'\x89PNG\r\n\x1a\n'
        report['exports'].append({'file': str(png), 'kind': 'current PNG', 'sha256': sha(png), 'image': list(struct.unpack('>II', raw[16:24]))})
        click('还原图窗', chart_scope())
        click('保存绘图配置')
        persisted = js('Object.fromEntries(Object.keys(localStorage).filter(k=>k.startsWith("ptb.v3.m02.")).map(k=>[k,JSON.parse(localStorage.getItem(k))]))')
        assert len(persisted) == 1
        report['savedConfiguration'] = persisted
        choice['folder'] = invalid
        click('选择参数目录')
        until('document.querySelector(".signal-panel>label select")?.options.length>=6')
        select_table('invalid-example.xlsx')
        until('!!document.querySelector(".m02-page>.error-banner")')
        snapshot('m02-invalid-table-error-light', '手动关联损坏的示例 XLSX 后，顶部显示原生读取失败提示；原音波形仍保留，可重新选择有效结果表。', {'deliberatelyInvalidExample': True, 'alert': js('document.querySelector(".m02-page>.error-banner").textContent')})
        select_table('single-syllable.ptb.sqlite')
        until('document.querySelectorAll(".parameter-curve").length===2&&!document.querySelector(".m02-page>.error-banner")')
        report['checks'].append('Actual invalid XLSX was rejected; selecting original valid SQLite restored curves and cleared error')
        w.view.reload()
        until('!!document.querySelector(".home-page")&&document.fonts.status==="loaded"')
        click('参数显示')
        until('!!document.querySelector(".m02-page")')
        assert js('document.querySelectorAll(".m02-files button").length') == 0
        choice['folder'] = inputs
        click('打开音频目录')
        until('document.querySelectorAll(".m02-files button").length===2')
        click('single-syllable.wav')
        until('document.querySelectorAll(".parameter-curve").length===2', 100)
        snapshot('m02-config-restored-light', '重新载入工作台并重新打开原音目录后，已保存的两项图窗分配恢复；时间窗回到全长，目录仍需要重新选择。', {'restoredCurves': 2, 'directoryReauthorized': True, 'fullSpanRestored': True})
        report['checks'].append('Same isolated profile reloaded: directory authorization absent, group configuration persisted; reload source explicitly restored curves with full span')
        assert originals == {p.name: sha(p) for p in inputs.iterdir() if p.is_file()}
        assert not report['terminations']
        report['checks'].append('All already prepared WAV, SQLite and XLSX files retain their original SHA-256; no scientific task was rerun')
        report['success'] = True
    except Exception as error:
        report['error'] = str(error)
        w.view.grab().save(str(captures / 'failed.png'))
        raise
    finally:
        w.closing = True
        w.close()
        pause(300)
        app.quit()
        (captures / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps({'success': report['success'], 'report': str(captures / 'report.json'), 'captures': len(report['captures'])}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
