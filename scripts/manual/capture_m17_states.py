"""Capture distinct M17 states using an owned maximized Windows Qt window.

Uses disposable workbench state and an off-record Qt profile. The real Qt
clipboard bridge writes to a process-local adapter, leaving user clipboard
contents untouched. TXT downloads go to this run's isolated output directory.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
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
    from PyQt6.QtCore import QEventLoop, QTimer, Qt, QPoint, QPointF, QEvent
    from PyQt6.QtGui import QMouseEvent
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication, QFileDialog
    from ptb_desktop.host import Workbench, register_scheme
    from verify_m08_wiring import setup

    out, db, cache = setup()
    captures, saved = out / 'm17-manual-states', out / 'm17-saved'
    captures.mkdir()
    saved.mkdir()
    clipboard_writes: list[str] = []

    class RecordingClipboard:
        value = ''

        def setText(self, value):
            self.value = value
            clipboard_writes.append(value)

        def text(self):
            return self.value

    clipboard = RecordingClipboard()
    QApplication.clipboard = lambda: clipboard
    QFileDialog.getSaveFileName = lambda *a, **k: (str(saved / Path(a[2]).name), '')
    register_scheme()
    app = QApplication(['PTB-owned-M17-manual-states'])
    w = Workbench(ROOT / 'frontend/dist', test=True, jobs_path=db,
                  local_files_root=cache, vocal_profile=out / 'vocal', start_module='M17')
    w.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
    w.showMaximized()
    report = {
        'success': False, 'chapterId': 'm17', 'out': str(out),
        'scope': 'Actual Windows Qt; owned hidden maximized window, off-record profile and isolated jobs/cache; no physical audio/IME/DWM/DPI claim',
        'clipboard': 'Original QWebChannel writeClipboard slot with process-local recording adapter; user clipboard untouched',
        'python': sys.executable, 'flags': os.environ.get('QTWEBENGINE_CHROMIUM_FLAGS'),
        'captures': [], 'checks': [], 'terminations': [],
        'sources': [{'path': p, 'sha256': sha(ROOT / p)} for p in (
            'scripts/manual/capture_m17_states.py', 'frontend/src/modules/ipa-plus/IpaPlusPage.vue',
            'frontend/src/modules/ipa-plus/ChartView.vue', 'frontend/src/modules/ipa-plus/editor.ts',
            'frontend/src/modules/ipa-plus/data/catalog.json', 'frontend/src/modules/ipa-plus/data/symbol-content.json',
            'frontend/dist/index.html', 'manual/V2_STYLE_GUIDE.md',
            'output/manual-work/v2-reference/html/chapter-01.html',
            'output/manual-work/v2-reference/html/chapter-10.html')],
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

    def until(code, seconds=40):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if js(code):
                return
            pause()
        raise AssertionError(code + '\n' + str(js('document.body.innerText.slice(-3500)')))

    def click(text):
        assert js('(()=>{const e=[...document.querySelectorAll("button")].find(b=>b.offsetParent&&b.textContent.trim()===' + json.dumps(text) + ');if(!e||e.disabled)return false;e.click();return true})()'), text
        pause(180)

    def fill(label, value, select=None):
        assert js('(()=>{const e=document.querySelector(' + json.dumps('[aria-label="' + label + '"]') + ');if(!e)return false;e.focus({preventScroll:true});e.value=' + json.dumps(value) + ';e.dispatchEvent(new Event("input",{bubbles:true}));e.dispatchEvent(new Event("change",{bubbles:true}));return true})()'), label
        pause(120)
        if select is not None:
            selection(*select)

    def selection(start, end):
        js('(()=>{const e=document.querySelector(".m17-editor");e.focus({preventScroll:true});e.setSelectionRange(' + str(start) + ',' + str(end) + ');e.dispatchEvent(new Event("select",{bubbles:true}));})()')
        pause(120)

    def text_state():
        return js('(()=>{const e=document.querySelector(".m17-editor");return {text:e.value,start:e.selectionStart,end:e.selectionEnd}})()')

    def native(selector, button=Qt.MouseButton.LeftButton):
        assert js('(()=>{const e=document.querySelector(' + json.dumps(selector) + ');if(!e)return false;e.scrollIntoView({block:"nearest",inline:"nearest"});return true})()'), selector
        pause(160)
        box = js('(()=>{const r=document.querySelector(' + json.dumps(selector) + ').getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2}})()')
        target = w.view.focusProxy() or w.view
        pos = target.mapFromGlobal(w.view.mapToGlobal(QPoint(round(box['x']), round(box['y']))))
        for point in (QPoint(4, 4), pos):
            event = QMouseEvent(QEvent.Type.MouseMove, QPointF(point),
                                QPointF(target.mapToGlobal(point)), Qt.MouseButton.NoButton,
                                Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier)
            QApplication.sendEvent(target, event)
            pause(40)
        QTest.mouseClick(target, button, Qt.KeyboardModifier.NoModifier, pos)
        pause(230)

    def symbol(identifier, inspect=False):
        selector = '[data-symbol-id="' + identifier + '"]'
        native(selector, Qt.MouseButton.RightButton if inspect else Qt.MouseButton.LeftButton)

    def close_detail():
        if js('!!document.querySelector(".m17-detail-panel")'):
            click('收起')
        js('document.querySelector(".m17-chart-viewport").scrollTop=0')

    def snapshot(name, caption, after_id):
        w.showMaximized()
        assert w.isMaximized()
        pause(500)
        w.view.repaint()
        w.view.grab()
        pause(350)
        app.processEvents()
        file = captures / (name + '.png')
        pix = w.view.grab()
        assert [pix.width(), pix.height()] == [2160, 1350], [pix.width(), pix.height()]
        assert pix.save(str(file))
        report['captures'].append({
            'id': name, 'chapterId': 'm17', 'file': str(file), 'caption': caption,
            'placementAfterId': after_id, 'state': text_state(),
            'distribution': 'software-only', 'git': False, 'maximized': w.isMaximized(),
            'window': [w.width(), w.height()], 'frame': [w.frameGeometry().width(), w.frameGeometry().height()],
            'image': [pix.width(), pix.height()], 'devicePixelRatio': pix.devicePixelRatio(),
            'theme': js('document.documentElement.dataset.theme'),
            'palette': js('document.documentElement.dataset.palette'),
            'fontSize': js('getComputedStyle(document.documentElement).fontSize'), 'sha256': sha(file),
        })
        print('Captured ' + name, flush=True)

    try:
        until('document.querySelector(".m17-body")?.dataset.loaded==="true"&&document.querySelector(".m17-body")?.dataset.fontReady==="true"&&document.fonts.status==="loaded"')
        click('设置')
        click('浅色')
        until('document.documentElement.dataset.theme==="light"')
        click('国际音标表Plus')
        until('document.querySelector(".ipa-plus-page")?.offsetParent!==null')
        assert text_state()['text'] == '', 'Off-record workbench must begin with empty text'
        symbol('ipa-vowels-025')
        assert text_state()['text'] == 'a'
        symbol('ipa-vowels-025', inspect=True)
        until('!!document.querySelector(".m17-detail-panel")')
        snapshot('m17-vowel-detail-r2-light', '点击元音图的 a 后，编辑框已输入 a。右键同一按钮展开开前不圆唇元音的实际输入、码位和介绍。', 'm17-chart-ipa-table')
        close_detail()

        click('附加符号与韵律')
        selection(1, 1)
        symbol('ipa-diacritics-071')
        assert text_state()['text'] == 'ã'
        selection(0, 2)
        until('document.querySelector(".m17-status code")?.textContent.includes("U+0303")')
        snapshot('m17-nasal-selection-r2-light', '光标置于 a 后点深色鼻化记号，正文成为 ã；选中完整字素后，底部显示 U+0061 与 U+0303，虚线圆没有写入。', 'm17-insertion-combining')
        fill('查找符号、名称、码位或 CIN', 'U+0062')
        symbol('ipa-pulmonic-002')
        assert text_state()['text'] == 'b'
        snapshot('m17-search-replace-r2-light', 'IPA 内检索 U+0062 后点浊双唇爆发音 b，原先选中的整个 ã 已替换为 b；搜索结果仍保留。', 'm17-workflow-search')
        click('撤销')
        assert text_state()['text'] == 'ã'
        click('重做')
        assert text_state()['text'] == 'b'
        report['checks'].append('Actual selected grapheme replacement and button undo/redo round trip')
        fill('查找符号、名称、码位或 CIN', '')

        click('基础图表')
        fill('国际音标文本', 'ts', (0, 2))
        symbol('ipa-other-011')
        assert text_state()['text'] == 't͡s'
        selection(0, 3)
        snapshot('m17-tie-combination-r2-light', '先选中 t 与 s 两个字素，再点其他符号中的上连音线，正文成为 t͡s；底部码位可核对连接字符 U+0361。', 'm17-insertion-bridge')

        click('extIPA')
        click('附加符号与发声')
        # This is a distinct extIPA page, not a second consonant overview.
        snapshot('m17-extipa-voicing-r2-light', 'extIPA 的附加符号与发声页并列展示附加符号、部分发声和构音记号；切表后此前的 t͡s 正文仍保留。', 'm17-chart-extipa-table')
        click('节奏与其他')
        fill('国际音标文本', 'pa ta', (0, 5))
        symbol('extipa-rhythm-006')
        assert text_state() == {'text': '{f pa ta f}', 'start': 3, 'end': 8}
        snapshot('m17-extipa-loud-range-r2-light', '选中 pa ta 后点较响范围，工具在两侧加入 {f 与 f}，范围内部的原文字保持选中，后续可继续替换。', 'm17-insertion-span')

        click('VoQS')
        fill('国际音标文本', 'aː', (0, 2))
        symbol('voqs-scope-007')
        assert text_state() == {'text': '{V! aː V!}', 'start': 4, 'end': 6}
        symbol('voqs-scope-007', inspect=True)
        snapshot('m17-voqs-range-detail-r2-light', '选中 aː 后点糙声范围，正文显示 {V! aː V!}；右键可核对范围工具说明，标签本身不生成声学测量值。', 'm17-chart-voqs-table')
        close_detail()

        before = text_state()
        native('.m17-toggle:nth-of-type(3) input')
        # Select by label if template layout changes; never modify component state.
        until('document.querySelector(".m17-chart-meta").textContent.includes("点击符号播放演示")')
        symbol('voqs-scope-007')
        until('document.querySelector(".m17-detail-panel")?.textContent.includes("此音标尚未添加演示内容")')
        assert text_state() == before
        snapshot('m17-playback-unavailable-r2-light', '开启点击播放后点糙声范围，详情显示此音标尚未添加演示内容；正文 {V! aː V!} 和内部选区保持。', 'm17-section-02')
        close_detail()
        native('.m17-toggle:nth-of-type(3) input')
        until('document.querySelector(".m17-chart-meta").textContent.includes("深色符号")')

        fill('国际音标文本', 'ã  t͡s\n{f pa ta f}\n{V! aː V!}', (0, 2))
        js('document.querySelector(".m17-divider").focus()')
        for _ in range(4):
            QTest.keyClick(w.view.focusProxy() or w.view, Qt.Key.Key_Up)
            pause(70)
        assert js('document.querySelector(".m17-divider").getAttribute("aria-valuenow")') == '184'
        selection(0, 2)
        report['checks'].append('Native divider ArrowUp adjusts editor height 120→184px; all three output lines visible')
        click('复制全部')
        until('document.querySelector(".m17-status").textContent.includes("已复制全部文字")')
        assert clipboard_writes[-1] == text_state()['text']
        snapshot('m17-copy-confirmed-r2-light', '点击复制全部后，底部显示已复制全部文字；复制内容包括三行正文，编辑框中的选区仍保留。', 'm17-section-04')
        click('保存文本')
        until('document.querySelector(".m17-status").textContent.includes("已发起 UTF-8 文本下载")')
        end = time.monotonic() + 15
        while time.monotonic() < end and not list(saved.glob('*.txt')):
            pause()
        files = list(saved.glob('*.txt'))
        assert len(files) == 1, files
        assert files[0].read_text(encoding='utf-8') == clipboard_writes[-1]
        snapshot('m17-save-request-r2-light', '点击保存文本后，底部提示已发起 UTF-8 文本下载。完成目标路径选择后，应重新打开 TXT 核对字符和换行。', 'm17-section-04')
        report['saved'] = [{'file': str(p), 'bytes': p.stat().st_size, 'encoding': 'UTF-8', 'sha256': sha(p), 'textEqualsEditorAndCopy': True} for p in files]
        report['checks'].append('Original clipboard bridge with recording adapter and actual UTF-8 Blob download round trip; three-line Unicode text identical')

        click('帮助')
        until('!!document.querySelector(".manual-reader")&&document.querySelector(".manual-breadcrumb")?.textContent.includes("国际音标表Plus")&&!!document.querySelector(".manual-document")')
        js('document.querySelector(".manual-reading-scroll").scrollTop=0')
        snapshot('m17-help-open-r2-light', '点击帮助进入国际音标表Plus 说明书，左侧提供目录和搜索；右上返回国际音标表Plus 可回到原编辑状态。', 'm17-controls-top-table')
        click('返回 国际音标表Plus')
        until('document.querySelector(".ipa-plus-page")?.offsetParent!==null')
        assert text_state()['text'] == clipboard_writes[-1]
        report['checks'].append('P19-R17 contextual help opens the actual M17 chapter and returns to retained text')
        click('关于')
        click('内置字体版权与许可')
        click('PTB IPA Plus')
        until('document.querySelector(".font-license-panel")?.textContent.includes("PTB IPA Plus 1.000")')
        snapshot('m17-font-license-r2-light', '从左下关于打开内置字体版权与许可，再选择 PTB IPA Plus 页签，可核对 1.000 版本、Doulos/Noto 来源与 OFL 原文。', 'm17-sources-license-table')
        assert not report['terminations'], report['terminations']
        report['checks'] += ['All PNG files captured from actual maximized Qt WebEngine view at 2160x1350 without resizing, cropping or annotation', 'Native pointer insertion/right-click detail, textarea input/selection events, actual toolbar button actions', 'No media created or played; unavailable-media prompt preserves text and selected range', 'No source assets, product code, central manifest or user files modified']
        report['success'] = True
    except BaseException:
        report['failureState'] = text_state()
        w.view.grab().save(str(out / 'm17-failure.png'))
        report['error'] = traceback.format_exc()
        print(report['error'], flush=True)
        raise
    finally:
        report_file = out / 'm17-manual-states-report.json'
        report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print('REPORT ' + str(report_file), flush=True)
        print('REPORT_SHA256 ' + sha(report_file), flush=True)
        w.closing = True
        w.close()
        pause(300)
        app.quit()


if __name__ == '__main__':
    main()
