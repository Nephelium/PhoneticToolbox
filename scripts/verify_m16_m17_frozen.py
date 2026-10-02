"""Limited M16/M17 smoke inside the caller's existing Qt/frozen workbench.

The caller owns QApplication, the window, its isolated profile and final report.
Only synthetic PortAudio is injected. No physical device, source catalogue,
existing database or new GUI process is opened by this helper.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path


def _process_image(process):
    """Read the real child image through its existing multiprocessing handle."""
    if os.name != 'nt':
        return str(Path(f'/proc/{process.pid}/exe').resolve(strict=True))
    import ctypes
    from ctypes import wintypes
    query = ctypes.WinDLL('kernel32', use_last_error=True).QueryFullProcessImageNameW
    query.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR,
                      ctypes.POINTER(wintypes.DWORD)]
    query.restype = wintypes.BOOL
    capacity = wintypes.DWORD(32768)
    buffer = ctypes.create_unicode_buffer(capacity.value)
    if not query(int(process._popen._handle), 0, buffer, ctypes.byref(capacity)):
        raise ctypes.WinError(ctypes.get_last_error())
    return buffer.value


def verify(window, out, report):
    """Append evidence to ``report``; raise on failure; never close the caller UI."""
    import numpy as np
    import soundfile as sf
    from PyQt6.QtCore import QEventLoop, QTimer
    from PyQt6.QtWidgets import QApplication
    from m16_test_backend import Backend
    from phonetic_core.recording.edits import frames
    from ptb_desktop.recording.storage import read_range

    app = QApplication.instance()
    assert app is not None, 'verify requires the caller-owned QApplication'
    evidence = Path(out) / 'm16-m17'
    evidence.mkdir(parents=True, exist_ok=False)
    project = evidence / 'project'
    exports = evidence / 'exports'
    project.mkdir()
    exports.mkdir()
    checks = report['m16_m17_checks'] = []
    layouts = report['m17_layouts'] = []
    report['m16_m17_success'] = False
    report['m16_m17_scope'] = (
        'Actual existing Qt/QWebChannel window; explicitly synthetic dual audio; '
        'real spawn processing and WAV readback; packaged font and six chart '
        'layouts. No physical soundcard/EGG, physical DPI or cross-platform claim.'
    )
    report['m16_m17_output'] = str(evidence)
    bridge = window.bridge.recording_bridge()
    service = bridge.service
    assert not service.capture and not service.job and not service.player
    assert service.project is None, 'M16 smoke must start with no existing project'
    original_backend, original_choose = service.backend, bridge.choose
    original_dispatch = service.dispatch
    original_size = window.size()
    service.backend = Backend()
    bridge.choose = lambda purpose: service.grant(exports if purpose == 'export' else project, purpose)

    def pause(ms=100):
        loop = QEventLoop()
        timer = QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(loop.quit)
        timer.start(ms)
        loop.exec()

    def js(code):
        loop = QEventLoop()
        timer = QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(loop.quit)
        values = []
        window.page.runJavaScript(code, lambda value: (values.append(value), loop.quit()))
        timer.start(5000)
        loop.exec()
        timer.stop()
        assert values, 'JavaScript callback timed out: ' + code[:180]
        return values[0]

    def until(code, seconds=30):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if js(code):
                return
            pause()
        raise AssertionError(code + '\n' + str(js('document.body.innerText.slice(-5000)')))

    def native_until(predicate, seconds=15):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if predicate():
                return
            pause(10)
        raise AssertionError('Native state timed out\n' + str(js('document.body.innerText.slice(-4000)')))

    def click(text, selector='.recording-page button', wait_ms=80):
        expression = ('[...document.querySelectorAll(' + json.dumps(selector) +
                      ')].find(e=>e.offsetParent && e.textContent.trim()===' + json.dumps(text) + ')')
        until('!!(' + expression + ') && !(' + expression + ').disabled')
        js(expression + '.click()')
        if wait_ms:
            pause(wait_ms)

    def nav(title):
        expression = '[...document.querySelectorAll(".nav-item")].find(e=>e.title===' + json.dumps(title) + ')'
        until('!!(' + expression + ')')
        js(expression + '.click()')

    def fill(label, value):
        js('(()=>{const l=[...document.querySelectorAll(".recording-page label")].find(e=>e.textContent.trim().startsWith(' +
           json.dumps(label) + '));const e=l.querySelector("input,textarea");e.value=' + json.dumps(str(value)) +
           ';e.dispatchEvent(new Event("input",{bubbles:true}));e.dispatchEvent(new Event("change",{bubbles:true}));})()')
        pause(40)

    def snapshot(name):
        # Qt offscreen can return a prior composited frame on the first grab.
        pause(400)
        window.view.grab()
        pause(350)
        app.processEvents()
        assert window.view.grab().save(str(evidence / name)), name

    def take():
        return service.project.data['takes'][0]

    def head():
        item = take()
        return item['versions'][item['head']]

    def raw_hashes():
        return {span['file']: hashlib.sha256((project / span['file']).read_bytes()).hexdigest()
                for span in take()['versions'][0]['spans']}

    try:
        window.resize(1500, 1000)
        nav('录音')
        until('!!document.querySelector(".recording-page")')
        click('新建工程')
        until('document.body.innerText.includes("本地录音工程已建立")')
        click('设备与录前检测')
        device = service.dispatch({'op': 'devices'})[0]['id']
        js('(()=>{const e=document.querySelector(\'select[aria-label="输入设备"]\');e.value=' +
           json.dumps(device) + ';e.dispatchEvent(new Event("change",{bubbles:true}));})()')
        until('document.querySelectorAll("select[aria-label^=物理输入]").length===2')
        assert js('[...document.querySelectorAll("select[aria-label^=物理输入]")].map(e=>e.value)') == ['microphone', 'microphone']
        click('收起')
        click('● 开始录音')
        native_until(lambda: service.capture is not None and service.capture.received >= 40000)
        assert service.capture.config['roles'] == ['microphone', 'microphone']
        assert service.capture.task is None
        click('■ 停止录音 / 检测')
        until('document.body.innerText.includes("已存入本地工程，尚未导出")')
        assert not bridge.capturing and len(service.project.data['takes']) == 1
        assert take()['task_snapshot'] is None and take()['config']['roles'] == ['microphone', 'microphone']
        total = frames(head()['spans'])
        assert 40000 <= total < 1000000
        original = read_range(project, head()['spans'], 0, total)
        original_hashes = raw_hashes()
        checks.append('M16 real UI default dual-audio free recording starts/stops and durably saves synthetic PCM')

        fill('起点（帧）', 10)
        fill('终点（帧）', 20)
        click('删除选区')
        until('document.body.innerText.includes("编辑已存入工程")')
        assert frames(head()['spans']) == total - 10
        edited = read_range(project, head()['spans'], 0, total - 10)
        assert np.array_equal(edited, np.concatenate([original[:10], original[20:]]))
        click('恢复原始')
        native_until(lambda: frames(head()['spans']) == total)
        assert np.array_equal(read_range(project, head()['spans'], 0, total), original)
        assert raw_hashes() == original_hashes
        checks.append('M16 exact frame deletion and raw restore preserve both audio channels and raw SHA-256')

        fill('起点（帧）', 0)
        fill('终点（帧）', 10000)
        click('设为噪声样本')
        until('document.body.innerText.includes("噪声样本：")')
        click('一键降噪（完整版本）', wait_ms=0)
        native_until(lambda: service.job is not None and service.job['kind'] == 'denoise')
        process = service.job['owner']
        child_image = _process_image(process)
        assert process.pid != os.getpid() and process._start_method == 'spawn'
        expected_images = {Path(sys.executable).resolve()}
        if not getattr(sys, 'frozen', False):
            # Windows venv launchers redirect to the base interpreter image.
            expected_images.add(Path(getattr(sys, '_base_executable', sys.executable)).resolve())
        assert Path(child_image).resolve() in expected_images, (child_image, expected_images)
        report['m16_frozen_spawn'] = {
            'frozen': bool(getattr(sys, 'frozen', False)), 'pid': process.pid,
            'start_method': process._start_method, 'child_executable': child_image,
            'parent_executable': sys.executable,
        }
        until('document.body.innerText.includes("后台处理已结束")', 90)
        assert head()['kind'] == 'denoised' and frames(head()['spans']) == total
        derived = read_range(project, head()['spans'], 0, total)
        assert np.isfinite(derived).all() and not np.array_equal(derived, original)
        assert raw_hashes() == original_hashes
        assert process.exitcode == 0
        report['m16_frozen_spawn']['exitcode'] = process.exitcode
        checks.append('M16 noise selection runs actual spawn child from the same executable, publishes a finite derived version and preserves raw')

        click('保存 / 批量导出')
        click('选择目录并导出')
        until('document.body.innerText.includes("已导出 1 / 1")', 60)
        wavs = list(exports.rglob('*.wav'))
        assert len(wavs) == 1 and not list(exports.rglob('*.partial'))
        wav = wavs[0]
        actual, rate = sf.read(wav, dtype='float32', always_2d=True)
        info = sf.info(wav)
        assert actual.shape == (total, 2) and rate == take()['config']['sample_rate']
        assert info.subtype == 'FLOAT' and np.array_equal(actual, derived)
        manifest = json.loads((wav.parent / 'manifest.json').read_text('utf8'))
        entry = manifest['items'][0]
        assert len(manifest['items']) == 1 and entry['status'] == 'exported'
        assert entry['files'][0]['sha256'] == hashlib.sha256(wav.read_bytes()).hexdigest()
        assert (wav.parent / 'manifest.csv').is_file()
        report['m16_export_readback'] = {'file': str(wav), 'frames': total, 'channels': 2,
                                       'sample_rate': rate, 'subtype': info.subtype,
                                       'float32_exact': True, 'raw_hashes': original_hashes}
        checks.append('M16 FLOAT WAV reopens with exact derived samples, original frame count, two channels and verified manifest hash')
        snapshot('m16-export.png')

        click('● 重录 / 新 take')
        native_until(lambda: service.capture is not None and service.capture.received >= 4000)
        nav('国际音标 Plus')
        until('document.querySelector(".m17-body")?.dataset.loaded==="true" && document.querySelector(".m17-body")?.dataset.fontReady==="true"')
        assert bridge.capturing
        text = '中文 ḁ 𝼆 V𐞀 '
        js('(()=>{const e=document.querySelector(".m17-editor");e.focus();e.value=' + json.dumps(text) +
           ';e.setSelectionRange(e.value.length,e.value.length);e.dispatchEvent(new Event("input",{bubbles:true}));'
           'e.dispatchEvent(new KeyboardEvent("keydown",{key:" ",code:"Space",bubbles:true}));})()')
        pause(150)
        assert bridge.capturing and service.player is None
        js('document.querySelector(\'button[aria-label="关闭 录音"]\').click()')
        until('document.body.innerText.includes("停止并保存工程后关闭")')
        click('取消关闭', 'button')
        assert bridge.capturing
        status_entered = threading.Event()
        def delayed_status(body):
            if body.get('op') == 'status':
                status_entered.set()
                time.sleep(.4)
            return original_dispatch(body)
        service.dispatch = delayed_status
        js('document.querySelector(\'button[aria-label="关闭 录音"]\').click()')
        native_until(status_entered.is_set)
        click('停止并保存工程后关闭', 'button')
        until('!document.querySelector("#tab-M16")')
        service.dispatch = original_dispatch
        assert not bridge.capturing and len(service.project.data['takes']) == 2
        assert raw_hashes() == original_hashes
        checks.append('Hidden M16 keeps recording during M17 input/Space; cancel-close continues, stop-save-close retains both takes')

        click('IPA', '.ipa-plus-page button')
        until('!!document.querySelector("[data-symbol-id=ipa-pulmonic-001]")')
        assert js('document.querySelector(".m17-editor").value') == text
        js('document.querySelector("[data-symbol-id=ipa-pulmonic-001]").click()')
        until('document.querySelector(".m17-editor").value===' + json.dumps(text + 'p'))
        click('撤销', '.ipa-plus-page button')
        until('document.querySelector(".m17-editor").value===' + json.dumps(text))
        click('重做', '.ipa-plus-page button')
        until('document.querySelector(".m17-editor").value===' + json.dumps(text + 'p'))
        until('document.querySelector(".m17-save-state").textContent.includes("已保存")')
        font = js('({ready:document.querySelector(".m17-body").dataset.fontReady,computed:getComputedStyle(document.querySelector(".m17-editor")).fontFamily,faces:[...document.fonts].filter(f=>f.family.includes("PTB-IPA-Plus")).map(f=>({family:f.family,status:f.status}))})')
        assert font['ready'] == 'true' and 'PTB-IPA-Plus' in font['computed']
        assert any(face['status'] == 'loaded' for face in font['faces']), font
        report['m17_packaged_font'] = font
        checks.append('M17 packaged PTB IPA Plus font loads; CJK/combining/non-BMP text, symbol insertion and undo/redo remain exact')

        for width, height in [(1366, 768), (1920, 1080)]:
            window.resize(width, height)
            pause(350)
            for label, system, count in [('IPA', 'ipa', 244), ('extIPA', 'extipa', 191), ('VoQS', 'voqs', 65)]:
                click(label, '.ipa-plus-page button')
                until('!!document.querySelector("[data-chart=' + system + ']")')
                pause(250)
                geometry = js('''(()=>{const v=document.querySelector('.m17-chart-viewport'),e=document.querySelector('.m17-editor'),b=v.getBoundingClientRect(),r=e.getBoundingClientRect(),symbols=[...v.querySelectorAll('[data-symbol-id]')],outside=symbols.filter(x=>{const a=x.getBoundingClientRect();return !a.width||!a.height||a.left<b.left-1||a.right>b.right+1||a.top<b.top-1||a.bottom>b.bottom+1}).map(x=>x.dataset.symbolId);return {viewport:[innerWidth,innerHeight],chart:{width:v.clientWidth,height:v.clientHeight,scrollWidth:v.scrollWidth,scrollHeight:v.scrollHeight,bottom:b.bottom,rectWidth:b.width,rectHeight:b.height,css:{border:getComputedStyle(v).border,padding:getComputedStyle(v).padding,zoom:getComputedStyle(v).zoom,transform:getComputedStyle(v).transform,outline:getComputedStyle(v).outline}},overflow:[...v.querySelectorAll('*')].filter(x=>{const a=x.getBoundingClientRect();return a.right>b.right+1||a.bottom>b.bottom+1}).map(x=>({tag:x.tagName,cls:x.className,text:x.textContent?.slice(0,30),rect:JSON.stringify(x.getBoundingClientRect())})).slice(0,20),editor:{top:r.top,bottom:r.bottom,height:r.height},symbolCount:symbols.length,uniqueCount:new Set(symbols.map(x=>x.dataset.symbolId)).size,outside}})()''')
                layouts.append({'width': width, 'height': height, 'system': system, **geometry})
                assert geometry['symbolCount'] == geometry['uniqueCount'] == count, geometry
                assert not geometry['outside'], geometry
                assert geometry['chart']['scrollWidth'] <= geometry['chart']['width'] + 2, geometry
                assert geometry['chart']['scrollHeight'] <= geometry['chart']['height'] + 2, geometry
                assert geometry['editor']['bottom'] <= geometry['viewport'][1] + 1, geometry
                assert geometry['editor']['top'] >= geometry['chart']['bottom'] - 1, geometry
                snapshot(f'm17-{width}x{height}-{system}.png')
        checks.append('M17 all 244/191/65 symbols visible in six measured layouts at 1366x768/1920x1080, without table scrolling; editor below each table')

        example = js('''(()=>{const e=[...document.querySelectorAll('.m17-chart-viewport .m17-example')].sort((a,b)=>Array.from(b.querySelector('.m17-ipa').textContent).length-Array.from(a.querySelector('.m17-ipa').textContent).length)[0];if(!e)return null;const text=e.querySelector('.m17-ipa').textContent;e.dispatchEvent(new MouseEvent('contextmenu',{bubbles:true,cancelable:true}));return {id:e.dataset.symbolId,text,codePointCount:Array.from(text).length}})()''')
        assert example and example['codePointCount'] > 8, example
        until('!!document.querySelector(".m17-detail-panel")')
        assert js('document.querySelector(".m17-detail-panel code")===null && !/U\\+[0-9A-F]{4,}/.test(document.querySelector(".m17-detail-panel").innerText)')
        report['m17_long_example'] = example
        snapshot('m17-long-example.png')
        click('收起', '.m17-detail-panel button')
        checks.append('M17 long VoQS example opens readable details without a Unicode U+ code-point string')
        report['m16_m17_success'] = True
    finally:
        # Keep ownership if cleanup fails so the caller can report/block shutdown.
        cleanup = service.close()
        service.dispatch = original_dispatch
        report['m16_m17_cleanup'] = cleanup
        if cleanup:
            service.backend = original_backend
            bridge.choose = original_choose
        window.resize(original_size)
        subset = {key: value for key, value in report.items()
                  if key.startswith(('m16_', 'm17_'))}
        (evidence / 'report.json').write_text(json.dumps(subset, ensure_ascii=False, indent=2), 'utf8')
        if not cleanup:
            report['m16_m17_success'] = False
            raise AssertionError('M16 synthetic smoke could not finish cleanup; ownership retained')

    return report
