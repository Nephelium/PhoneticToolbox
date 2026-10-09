"""Capture actual M11 resource and task states in an owned maximized Qt window.

Only the isolated component registry, public probe corpus and owned task state
change. Existing MFA environments and model sources are never packaged.
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
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
DIRS = ('desktop/src', 'backend/src', 'packages/phonetic_core/src', 'scripts')
for directory in DIRS:
    sys.path.insert(0, str(ROOT / directory))
os.environ['PYTHONPATH'] = os.pathsep.join(str(ROOT / p) for p in DIRS)
os.environ.setdefault('QT_QPA_PLATFORM', 'windows')
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS', '--disable-gpu --disable-gpu-compositing --mute-audio')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    from PyQt6.QtCore import QEventLoop, QTimer, Qt, QPoint
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication, QFileDialog
    from ptb_desktop.host import Workbench, register_scheme
    from ptb_worker.local_workspace import prepare_workspace
    from ptb_worker.mfa.probe import generate

    out = ROOT / 'output/manual-work/m11-captures' / uuid4().hex
    out.mkdir(parents=True)
    captures, components, inputs, saved = [out / p for p in ('screenshots', 'components', 'inputs', 'saved')]
    for directory in (captures, components, inputs, saved):
        directory.mkdir()
    source_registry = ROOT / 'output/m11c-028b881d/registry.json'
    shutil.copyfile(source_registry, components / 'registry.json')
    registry = json.loads(source_registry.read_text('utf8'))
    pair = next(m for m in registry['models'] if m.get('dictionary_name') == 'mandarin_pinyin_tab.dict')
    original_model = Path.home() / 'Desktop/PhoneticToolbox/mfa_models/acoustic/mandarin.zip'
    original_dictionary = Path.home() / 'Desktop/PhoneticToolbox/mfa_models/dictionary/mandarin_pinyin_tab.dict'
    sources = [source_registry, original_model, original_dictionary]
    if any(not p.is_file() for p in sources):
        raise FileNotFoundError('Existing registry, compatible model or dictionary is unavailable')
    source_hashes = {str(p): sha(p) for p in sources}
    os.environ['PTB_M11_COMPONENT_ROOT'] = str(components)
    os.environ.pop('PTB_M11_BUNDLED_COMPONENTS', None)
    generate(inputs, word='a1')
    # Write the public fixture as standard long TextGrid; the actual MFA
    # runtime parses it with its own pinned praatio dependency.
    lines = ['File type = "ooTextFile"', 'Object class = "TextGrid"', '',
             'xmin = 0', 'xmax = 2.8', 'tiers? <exists>', 'size = 2', 'item []:']
    for index, (name, intervals) in enumerate((
        ('speaker_words', [(0., .8, 'a1'), (.8, 1.4, 'a1'), (1.4, 2., 'a1'), (2., 2.8, 'a1')]),
        ('speaker_phones', [(0., 2.8, 'a')]),
    ), 1):
        lines.extend([f'    item [{index}]:', '        class = "IntervalTier"',
                      f'        name = "{name}"', '        xmin = 0', '        xmax = 2.8',
                      f'        intervals: size = {len(intervals)}'])
        for number, (start, end, label) in enumerate(intervals, 1):
            lines.extend([f'        intervals [{number}]:', f'            xmin = {start}',
                          f'            xmax = {end}', f'            text = "{label}"'])
    (inputs / 'probe.TextGrid').write_text('\n'.join(lines) + '\n', encoding='utf8')
    input_hashes = {p.name: sha(p) for p in inputs.iterdir()}
    db, cache = prepare_workspace(out / 'state', ROOT / 'backend/migrations')
    report = {
        'success': False, 'chapterId': 'm11', 'out': str(out),
        'scope': 'Actual Windows Qt/QWebChannel, owned hidden maximized window and real MFA 3.3.8; public synthetic probe only; no physical audio, DPI, DWM or accuracy claim',
        'python': sys.executable, 'flags': os.environ.get('QTWEBENGINE_CHROMIUM_FLAGS'),
        'captures': [], 'checks': [], 'errors': [], 'terminations': [],
        'sourceHashes': source_hashes, 'inputHashes': input_hashes,
        'sourceEvidence': [{'path': str(ROOT / p), 'sha256': sha(ROOT / p)} for p in (
            'scripts/manual/capture_m11_states.py', 'frontend/src/modules/mfa/MfaAlignmentPage.vue',
            'frontend/src/modules/mfa/state.ts', 'backend/src/ptb_worker/mfa/child.py',
            'frontend/dist/index.html', 'manual/V2_STYLE_GUIDE.md',
            'output/manual-work/v2-reference/html/chapter-08.html')],
    }
    choice = {'corpus': inputs, 'model': original_model, 'dictionary': original_dictionary}
    def file_picker(*args, **kwargs):
        return (str(choice['model'] if str(args[1]).endswith('model') else choice['dictionary']), '')
    QFileDialog.getOpenFileName = file_picker
    QFileDialog.getExistingDirectory = lambda *a, **k: str(choice['corpus'] if 'corpus' in str(a) else saved)
    register_scheme()
    app = QApplication(['PTB-owned-M11-manual-capture'])
    w = Workbench(ROOT / 'frontend/dist', test=True, jobs_path=db, local_files_root=cache, vocal_profile=out / 'vocal')
    w.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
    w.showMaximized()
    w.page.setAudioMuted(True)
    w.page.renderProcessTerminated.connect(lambda s, c: report['terminations'].append([s.name, c]))

    def pause(ms=120):
        loop = QEventLoop()
        QTimer.singleShot(ms, loop.quit)
        loop.exec()

    def js(code):
        loop, values = QEventLoop(), []
        w.page.runJavaScript(code, lambda v: (values.append(v), loop.quit()))
        QTimer.singleShot(6000, loop.quit)
        loop.exec()
        if not values:
            raise RuntimeError('JavaScript timeout')
        return values[0]

    def until(code, seconds=180):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if js(code):
                return
            pause()
        raise AssertionError(code + '\n' + str(js('document.querySelector(".mfa-page")?.innerText.slice(-4500)')))

    def idle():
        until('document.querySelector(".mfa-page")?.getAttribute("aria-busy")==="false"')

    def click(label):
        box = js('(()=>{const e=[...document.querySelectorAll("button")].find(b=>b.offsetParent&&b.textContent.trim()===' + json.dumps(label) + ');if(!e||e.disabled)return null;e.scrollIntoView({block:"nearest"});const r=e.getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2}})()')
        if not box:
            raise AssertionError('Button unavailable: ' + label)
        QTest.mouseClick(w.view.focusProxy() or w.view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(round(box['x']), round(box['y'])))
        pause()

    def select(label, value):
        assert js('(()=>{const e=[...document.querySelectorAll(".mfa-page label")].find(l=>l.childNodes[0].textContent.trim()===' + json.dumps(label) + ')?.querySelector("select");if(!e||e.disabled)return false;e.value=' + json.dumps(value) + ';e.dispatchEvent(new Event("change",{bubbles:true}));return true})()'), label
        pause()
        idle()

    def snapshot(name, caption, state):
        assert w.isMaximized(), 'Window must be maximized before capture'
        pause(350)
        w.view.repaint()
        w.view.grab()
        pause(250)
        app.processEvents()
        pix = w.view.grab()
        path = captures / (name + '.png')
        assert pix.save(str(path))
        report['captures'].append({
            'id': name, 'chapterId': 'm11', 'file': str(path), 'caption': caption, 'state': state,
            'maximized': True, 'window': [w.width(), w.height()],
            'frame': [w.frameGeometry().width(), w.frameGeometry().height()],
            'image': [pix.width(), pix.height()], 'devicePixelRatio': pix.devicePixelRatio(),
            'theme': js('document.documentElement.dataset.theme'), 'palette': js('document.documentElement.dataset.palette'),
            'fontSize': js('getComputedStyle(document.documentElement).fontSize'),
            'distribution': 'software-only', 'git': False, 'sha256': sha(path),
        })
        (out / 'progress.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
        print('Captured ' + name, flush=True)

    def page_has(s):
        return 'document.querySelector(".mfa-page")?.innerText.includes(' + json.dumps(s) + ')'

    try:
        until('!!document.querySelector("nav")&&document.fonts.status==="loaded"')
        js('document.documentElement.dataset.theme="light"')
        click('MFA 自动标注')
        until('!!document.querySelector(".mfa-page")')
        idle()
        until('document.querySelector(".mfa-form select")?.options.length>1')
        select('声学模型', pair['id'])
        select('发音词典', 'import')
        select('声学模型', 'import')
        assert js(page_has('待检查'))
        assert js('[...document.querySelectorAll(".mfa-page button")].find(b=>b.textContent.trim()==="开始对齐").disabled')
        snapshot('m11-pending-resources-light', '顶部导入模型与词典后进入待检查状态，已应用组合保留，开始对齐暂时禁用。', 'real native picker handler; pending compatible files')
        click('检查并应用所选资源')
        until('document.querySelector(".mfa-page").getAttribute("aria-busy")==="true"')
        snapshot('m11-check-running-light', '检查并应用所选资源正在执行真实 MFA 小任务，等待完成后再读入语料。', 'real compatibility check running; no asserted result yet')
        idle()
        assert js(page_has('已检查并应用'))
        snapshot('m11-checked-resources-light', '检查通过后自动应用 mandarin.zip 与配套拼音词典，待检查提示消失。', 'real public a1 self-test succeeded')
        report['checks'].append('Real top resource import, a1 probe and automatic application; prior environment reused')
        click('打开音频目录')
        idle()
        assert js(page_has('唯一同名转写'))
        snapshot('m11-transcript-ambiguous-light', '同名 LAB 与 TextGrid 共存时，自动来源要求唯一转写，需选择本次读取的格式。', 'actual directory ambiguity error; no input deletion')
        select('转写来源', '.lab')
        click('打开音频目录')
        idle()
        until('document.querySelectorAll(".corpus-list li").length===1')
        snapshot('m11-lab-config-light', '明确选择 LAB 并重新读取后，清单显示一组配对文件，搜索参数为 100 和 400。', 'real corpus .lab admission; ready to submit')
        select('转写来源', '.TextGrid')
        snapshot('m11-transcript-reload-light', '更改转写来源后旧清单暂停提交，按提示重新打开目录或导入文件。', 'actual sourceChanged guard')
        click('打开音频目录')
        idle()
        assert js('document.querySelector(".corpus-list")?.innerText.includes(".TextGrid")')
        snapshot('m11-textgrid-config-light', '重新读取 TextGrid 来源后，任务采用该格式；原 LAB 和 TextGrid 仍同时保留。', 'real TextGrid corpus admission')
        click('开始对齐')
        idle()
        until(page_has('MFA 正在对齐'), 100)
        snapshot('m11-task-aligning-light', '任务运行中显示实际阶段与日志，进度只代表阶段，不估算剩余秒数。', 'real CPU MFA task, synthetic words TextGrid input')
        until('document.querySelectorAll(".result-row").length===2&&document.querySelector(".job-state")?.textContent.includes("完整结果")')
        snapshot('m11-task-results-light', '完整任务发布后列出 TextGrid 和 m11-provenance.json，可单独下载或保存全部成果。', 'actual successful TextGrid task and published manifest')
        click('保存完整结果到输出目录')
        idle()
        snapshot('m11-saved-results-light', '完整保存后提示独立 MFA 结果子目录，原音频和原始 TextGrid 保留。', 'actual native output grant and hashed complete save')
        provenance_path = max(saved.rglob('m11-provenance.json'), key=lambda p: p.stat().st_mtime)
        result = json.loads(provenance_path.read_text('utf8'))
        assert result['model']['sha256'] == sha(original_model)
        assert result['dictionary_sha256'] == sha(original_dictionary)
        assert result['transcript_adaptations'][0]['mode'] == 'words-to-whole-recording-transcript'
        labels = [e[2] for t in result['textgrids'] for tier in t['tiers'] if tier['name'].endswith('words') for e in tier['entries'] if e[2]]
        assert labels == ['a1'] * 4, labels
        assert all(sha(inputs / name) == value for name, value in input_hashes.items())
        report['savedResult'] = {'provenance': str(provenance_path), 'sha256': sha(provenance_path), 'wordLabels': labels, 'transcriptAdaptations': result['transcript_adaptations']}
        report['checks'].append('Real LAB/TextGrid source selection, guard, TextGrid adaptation, task publication and complete hashed save')
        bad = out / 'oov-input'
        generate(bad, word='ptbunknownword')
        choice['corpus'] = bad
        select('转写来源', '.lab')
        click('打开音频目录')
        idle()
        click('开始对齐')
        idle()
        until('document.querySelector(".job-state")?.textContent.includes("任务失败")')
        assert js(page_has('词典未收录'))
        js('(()=>{const d=[...document.querySelectorAll(".mfa-page details")].find(x=>x.querySelector("summary")?.textContent.includes("查看脱敏 MFA 日志"));if(d)d.open=true})()')
        snapshot('m11-oov-failure-light', '转写含词典未收录的词时任务失败，展开脱敏 MFA 日志查具体词并修正后重新提交。', 'real native OOV rejection; no full result published')
        report['checks'].append('Real OOV task error and retained diagnostic log')
        report['originalSourcesUnchanged'] = all(sha(Path(p)) == value for p, value in source_hashes.items())
        assert report['originalSourcesUnchanged']
        report['success'] = True
    except Exception as exc:
        report['errors'].append({'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()})
        try:
            snapshot('m11-diagnostic-failure-light', '截图任务的诊断画面，仅用于核对失败原因。', 'diagnostic only; not intended for manual')
        except Exception:
            pass
    finally:
        report['expectedMissingStates'] = [] if report['success'] else ['Complete capture flow interrupted; inspect errors and captures before registration']
        (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
        w.closing = True
        w.close()
        pause(500)
        app.quit()
        print(json.dumps({'success': report['success'], 'report': str(out / 'report.json'), 'captures': len(report['captures']), 'errors': report['errors']}, ensure_ascii=False), flush=True)
    return 0 if report['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
