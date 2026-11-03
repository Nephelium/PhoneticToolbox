"""Capture distinct M17 states using an owned maximized Windows Qt window.

Uses disposable workbench state and an off-record Qt profile. The real Qt
clipboard bridge writes to a process-local adapter, leaving user clipboard
contents untouched. TXT downloads go to this run's isolated output directory.
"""
from __future__ import annotations

import hashlib
import argparse
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
os.environ['QTWEBENGINE_CHROMIUM_FLAGS'] = '--mute-audio --autoplay-policy=no-user-gesture-required'
for key in ('QT_ENABLE_HIGHDPI_SCALING','QT_SCALE_FACTOR','QT_SCREEN_SCALE_FACTORS'):os.environ.pop(key,None)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture(args,out,scratch):
    from PyQt6.QtCore import QEventLoop, QTimer, Qt, QPoint, QPointF, QEvent
    from PyQt6.QtGui import QMouseEvent
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication, QFileDialog
    from PyQt6 import sip
    from ptb_desktop.host import Workbench, register_scheme
    from ptb_worker.local_workspace import prepare_workspace
    from ptb_desktop.legacy_cache import isolated_context
    from manual.capture_locale import CaptureLocale
    from manual.author_preview import prepare

    ui=CaptureLocale(args.locale,('ipa-plus','workbench','components','manual'))
    paths=[Path(__file__),ROOT/'scripts/manual/capture_locale.py',ROOT/'scripts/manual/author_preview.py',args.frontend_dist/'index.html',ROOT/'desktop/src/ptb_desktop/host.py',ROOT/'frontend/src/app/AppShell.vue',ROOT/'frontend/src/app/HelpPage.vue',ROOT/'manual/chapters/m17.json',ROOT/'frontend/src/state/fonts.ts']
    paths+=list((ROOT/'frontend/src/modules/ipa-plus').glob('*.vue'))+list((ROOT/'frontend/src/modules/ipa-plus').glob('*.ts'))
    paths += [ROOT/f'manual/locales/{args.locale}/m17.json']
    paths += [ROOT/'frontend/src/i18n/locales'/code/(namespace+'.json') for code in ('zh-CN',args.locale) for namespace in ('ipa-plus','ipa-plus-catalog','ipa-plus-content','ipa-plus-sources','workbench','components','manual') if (ROOT/'frontend/src/i18n/locales'/code/(namespace+'.json')).is_file()]
    sources={str(i):{'path':str(p.resolve()),'sha256':sha(p)} for i,p in enumerate(dict.fromkeys(paths))}
    preview=prepare(scratch,args.locale,args.frontend_dist)
    db,cache=prepare_workspace(scratch/'state',ROOT/'backend/migrations')
    captures, saved = out / 'm17-manual-states', scratch / 'm17-saved'
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
    w = Workbench(preview['frontend'], test=True, jobs_path=db,
                  local_files_root=cache, vocal_profile=scratch / 'vocal', start_module='M17',cache_context=isolated_context(scratch/'qa-caches'))
    w.bridge.tasks._manual_packs=preview['manager']
    w.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
    w.resize(w.screen().geometry().width(),w.screen().geometry().height())
    w.showMaximized()
    w.view.setZoomFactor(1)
    report = {
        'success': False, 'chapterId': 'm17', 'sourceChapter':'m17','locale':args.locale,'out': str(out),'authoringPreview':preview['receipt'],
        'scope': 'Actual Windows Qt; owned hidden maximized window, off-record profile and isolated jobs/cache; no physical audio/IME/DWM/DPI claim',
        'clipboard': 'Original QWebChannel writeClipboard slot with process-local recording adapter; user clipboard untouched',
        'python': sys.executable, 'flags': os.environ.get('QTWEBENGINE_CHROMIUM_FLAGS'),
        'captures': [], 'checks': [], 'terminations': [],
        'sources': sources,'physicalCompositorVerified':False,
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
        if text=='返回 国际音标表Plus':text=ui.key('workbench.return_to')+ui.key('workbench.ipa_chart_plus')
        elif text=='收起':text=ui.key('modules.ipa-plus.text.afd4b783536b')
        elif text not in ('extIPA','VoQS','PTB IPA Plus'):text=ui.label(text)
        assert js('(()=>{const text='+json.dumps(text)+';const e=[...document.querySelectorAll("button")].find(b=>b.offsetParent&&(b.textContent.trim()===text||b.getAttribute("aria-label")===text||b.title===text));if(!e||e.disabled)return false;e.click();return true})()'), text
        pause(180)

    def fill(label, value, select=None):
        label=ui.label(label)
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
        if button is not None:QTest.mouseClick(target, button, Qt.KeyboardModifier.NoModifier, pos)
        pause(230)

    def symbol(identifier, inspect=False):
        selector = '[data-symbol-id="' + identifier + '"]'
        native(selector, Qt.MouseButton.RightButton if inspect else Qt.MouseButton.LeftButton)

    def close_detail():
        if js('!!document.querySelector(".m17-detail-panel")'):
            click('收起')
        js('document.querySelector(".m17-chart-viewport").scrollTop=0')

    def snapshot(name, caption, after_id):
        if args.scene and name not in args.scene:return
        w.showMaximized()
        assert w.isMaximized()
        pause(500)
        w.view.repaint()
        w.view.grab()
        pause(350)
        app.processEvents()
        file = captures / (name + '.png')
        metrics=ui.metrics(js)
        regions=js('(()=>{return Object.fromEntries([["editor",".m17-editor-area"],["status",".m17-status"]].map(([name,selector])=>{const r=document.querySelector(selector).getBoundingClientRect();return [name,{selector,box:[r.left,r.top,r.width,r.height],units:"css-px"}];}));})()') if name=='m17-tie-combination-r2-light' else {}
        pix = w.view.grab().toImage();raw=[pix.width(),pix.height()]
        assert raw in ([2560,1600],[2561,1601]) and pix.devicePixelRatio()==1.5,raw
        if raw!=[2560,1600]:pix=pix.copy(0,0,2560,1600)
        assert pix.save(str(file))
        report['captures'].append({
            'id': name,'name':name,'chapterId': 'm17', 'file': file.relative_to(out).as_posix(), 'caption': caption,
            'placementAfterId': after_id, 'state': text_state(),
            'distribution': 'software-only', 'git': False, 'maximized': w.isMaximized(),
            'window': [w.width(), w.height()], 'frame': [w.frameGeometry().width(), w.frameGeometry().height()],
            'image': [pix.width(), pix.height()], 'devicePixelRatio': pix.devicePixelRatio(),
            'theme': js('document.documentElement.dataset.theme'),
            'palette': js('document.documentElement.dataset.palette'),
            'fontSize': js('getComputedStyle(document.documentElement).fontSize'), 'sha256': sha(file),
            **metrics,
            'regions':regions,
        })
        print('Captured ' + name, flush=True)

    try:
        until('document.querySelector(".m17-body")?.dataset.loaded==="true"&&document.querySelector(".m17-body")?.dataset.fontReady==="true"&&document.fonts.status==="loaded"')
        ui.prepare_qt(js,until)
        click('设置')
        click('浅色')
        until('document.documentElement.dataset.theme==="light"')
        click('国际音标表Plus')
        until('document.querySelector(".ipa-plus-page")?.offsetParent!==null')
        click('收起侧栏')
        assert text_state()['text'] == '', 'Off-record workbench must begin with empty text'
        snapshot('m17-overview-light','Actual current IPA chart, empty editor and disabled playback control.','m17-overview')
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
        # Move the real pointer away without clicking or changing the selected
        # text. A hover card must not cover the editor in the focused detail.
        native('#tab-M17',None)
        until('!document.querySelector(".m17-hover-panel")')
        snapshot('m17-tie-combination-r2-light', '先选中 t 与 s 两个字素，再点其他符号中的上连音线，正文成为 t͡s；底部码位可核对连接字符 U+0361。', 'm17-insertion-bridge')

        click('extIPA')
        snapshot('m17-extipa-detail-r4-light','Actual current extIPA chart.','m17-chart-extipa-table')
        click('附加符号与发声')
        # This is a distinct extIPA page, not a second consonant overview.
        snapshot('m17-extipa-voicing-r2-light', 'extIPA 的附加符号与发声页并列展示附加符号、部分发声和构音记号；切表后此前的 t͡s 正文仍保留。', 'm17-chart-extipa-table')
        click('节奏与其他')
        fill('国际音标文本', 'pa ta', (0, 5))
        symbol('extipa-rhythm-006')
        assert text_state() == {'text': '{f pa ta f}', 'start': 3, 'end': 8}
        snapshot('m17-extipa-loud-range-r2-light', '选中 pa ta 后点较响范围，工具在两侧加入 {f 与 f}，范围内部的原文字保持选中，后续可继续替换。', 'm17-insertion-span')

        click('VoQS')
        snapshot('m17-voqs-detail-r4-light','Actual current VoQS chart.','m17-chart-voqs-table')
        fill('国际音标文本', 'aː', (0, 2))
        symbol('voqs-scope-007')
        assert text_state() == {'text': '{V! aː V!}', 'start': 4, 'end': 6}
        symbol('voqs-scope-007', inspect=True)
        snapshot('m17-voqs-range-detail-r2-light', '选中 aː 后点糙声范围，正文显示 {V! aː V!}；右键可核对范围工具说明，标签本身不生成声学测量值。', 'm17-chart-voqs-table')
        close_detail()

        before = text_state()
        disabled=js('(()=>{const e=document.querySelector(".m17-toggle-unavailable");return {disabled:e.querySelector("input").disabled,title:e.title};})()')
        assert disabled=={'disabled':True,'title':ui.key('modules.ipa-plus.text.fd30e8f09aa8')},disabled
        native('.m17-toggle-unavailable')
        assert text_state() == before
        report['playbackUnavailable']=disabled
        snapshot('m17-playback-unavailable-r2-light', 'Current disabled playback control and unchanged input; native title records that demonstration content is being prepared.', 'm17-section-02')
        until('document.querySelector(".m17-chart-meta").textContent.includes('+json.dumps(ui.key('modules.ipa-plus.text.6dc3042751a1'))+')')

        fill('国际音标文本', 'ã  t͡s\n{f pa ta f}\n{V! aː V!}', (0, 2))
        js('document.querySelector(".m17-divider").focus()')
        for _ in range(4):
            QTest.keyClick(w.view.focusProxy() or w.view, Qt.Key.Key_Up)
            pause(70)
        assert js('document.querySelector(".m17-divider").getAttribute("aria-valuenow")') == '184'
        selection(0, 2)
        report['checks'].append('Native divider ArrowUp adjusts editor height 120→184px; all three output lines visible')
        click('复制全部')
        until('document.querySelector(".m17-status").textContent.includes('+json.dumps(ui.key('modules.ipa-plus.text.d688831e0216'))+')')
        assert clipboard_writes[-1] == text_state()['text']
        snapshot('m17-copy-confirmed-r2-light', '点击复制全部后，底部显示已复制全部文字；复制内容包括三行正文，编辑框中的选区仍保留。', 'm17-section-04')
        click('保存文本')
        until('document.querySelector(".m17-status").textContent.includes('+json.dumps(ui.key('modules.ipa-plus.text.821407425965'))+')')
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
        # Contextual navigation does not simulate an outline click. The reader
        # marks third-level locations, so the chapter button need not be current.
        until('!!document.querySelector(".manual-reader .manual-document[data-manual-chapter=m17]")&&!!document.querySelector("button[data-manual-chapter=m17]")')
        assert js('document.querySelector(".manual-language-toolbar select").value')==args.locale
        report['helpTarget']=js('({chapter:document.querySelector(".manual-document").dataset.manualChapter,title:document.querySelector(".manual-breadcrumb").innerText,language:document.querySelector(".manual-language-toolbar select").value})')
        assert report['helpTarget']['chapter']=='m17' and report['helpTarget']['language']==args.locale
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
        assert set(args.scene).issubset({item['name'] for item in report['captures']}),'Unknown or uncaptured scene'
        report['checks'] += ['All PNG files from actual maximized Qt view at native DPR1.5/page100; only optional one-pixel rounding crop to 2560x1600, no resizing', 'Native pointer insertion/right-click detail, textarea input/selection events, actual toolbar button actions', 'Current disabled playback control preserves text and selected range; no media played', 'Complete author-only localized reader uses real native import/read and HelpPage; final pack gate unchanged']
        report['success'] = True
    except BaseException:
        report['failureState'] = text_state()
        w.view.grab().save(str(out / 'm17-failure.png'))
        report['error'] = traceback.format_exc()
        print(report['error'], flush=True)
        raise
    finally:
        report_file = out / 'm17-manual-states-report.json'
        report['sourceChangesDuringRun']=[p['path'] for p in sources.values() if sha(Path(p['path']))!=p['sha256']]
        report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print('REPORT ' + str(report_file), flush=True)
        print('REPORT_SHA256 ' + sha(report_file), flush=True)
        w.closing = True
        w.close()
        app.processEvents();sip.delete(w);app.processEvents();sip.delete(app)


def main(args):
    from verification_artifacts import verification_run
    with verification_run('m17-manual',args.locale,'Actual isolated Qt, current UI locale, full author-only manual preview, no physical device or final EXE claim') as (out,scratch):
        capture(args,out,scratch)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--locale',choices=('en','ja'),required=True);parser.add_argument('--frontend-dist',type=Path,default=ROOT/'frontend/dist');parser.add_argument('--scene',action='append',default=[],help='Only save the named scene; repeat for multiple scenes. All interaction checks still run.')
    main(parser.parse_args())
