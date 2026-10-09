"""Capture distinct real M07 states in an owned maximized Qt workbench.

Read registered example WAVs without copying them. Work only in a new ignored
capture directory and a copy of the existing test database. No user UI or DDL.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
for relative in ('desktop/src', 'backend/src', 'packages/phonetic_core/src', 'scripts'):
    sys.path.insert(0, str(ROOT / relative))
os.environ['PYTHONPATH'] = os.pathsep.join(str(ROOT / p) for p in
                                        ('desktop/src', 'backend/src', 'packages/phonetic_core/src', 'scripts'))
os.environ.setdefault('QT_QPA_PLATFORM', 'windows')
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS', '--mute-audio --autoplay-policy=no-user-gesture-required')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    from PyQt6.QtCore import QEventLoop, QTimer, Qt
    from PyQt6.QtWidgets import QApplication, QFileDialog
    from ptb_desktop.host import Workbench, register_scheme
    from ptb_worker.local_acoustic_files import initialize_local_files

    out = ROOT / 'output/manual-work/m07-captures' / uuid4().hex
    out.mkdir(parents=True)
    saved = out / 'saved'
    saved.mkdir()
    database = out / 'jobs.sqlite3'
    template = ROOT / 'output/validation/p06/local-state.sqlite3'
    with sqlite3.connect(template.as_uri() + '?mode=ro', uri=True) as source, sqlite3.connect(database) as dest:
        assert not source.execute("SELECT 1 FROM jobs WHERE state IN ('queued','running','cancel_requested')").fetchone()
        source.backup(dest)
    cache = out / 'cache'
    cache.mkdir()
    initialize_local_files(cache)
    inputs = ROOT / 'manual/assets/software-only'
    input_hashes = {name: hashlib.sha256((inputs / name).read_bytes()).hexdigest()
                    for name in ('single-original.wav', 'single-target.wav')}
    choices = {'directory': inputs}
    QFileDialog.getExistingDirectory = lambda *args, **kwargs: str(choices['directory'])
    QFileDialog.getSaveFileName = lambda *args, **kwargs: (str(saved / Path(args[2]).name), '')

    register_scheme()
    app = QApplication(['PTB-owned-M07-manual-capture'])
    window = Workbench(ROOT / 'frontend/dist', test=True, jobs_path=database,
                       local_files_root=cache, vocal_profile=out / 'vocal',
                       reaper_binary=ROOT / 'phonetic_toolbox/core/acoustic/reaper.exe')
    window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
    window.showMaximized()
    window.page.setAudioMuted(True)
    report = {'success': False, 'chapterId': 'm07', 'out': str(out),
              'scope': 'Actual current built frontend and scientific tasks in maximized hidden Windows Qt; muted audio; no physical DPI/compositor or hearing claim',
              'captures': [], 'checks': [], 'terminations': [], 'inputHashes': input_hashes,
              'settings': {'f0_backend': 'parselmouth', 'min_f0_hz': 50, 'max_f0_hz': 600,
                           'f0_frame_interval_ms': 1, 'point_count': 21, 'alignment': 'normalize',
                           'continuum_type': 2, 'reverse_direction': False, 'step_count': 9,
                           'energy_match': True, 'normalize_to_source': True, 'output_peak_limit': .98}}
    report['sourceSnapshot'] = [{'path': str(file.relative_to(ROOT)), 'sha256': hashlib.sha256(file.read_bytes()).hexdigest()}
                                for file in sorted((ROOT / 'frontend/dist').rglob('*')) if file.is_file()
                                and file.suffix in ('.html', '.js', '.css')]
    window.page.renderProcessTerminated.connect(lambda status, code: report['terminations'].append([status.name, code]))

    def pause(ms=100):
        loop = QEventLoop()
        QTimer.singleShot(ms, loop.quit)
        loop.exec()

    def js(code):
        loop, box = QEventLoop(), []
        window.page.runJavaScript(code, lambda value: (box.append(value), loop.quit()))
        QTimer.singleShot(6000, loop.quit)
        loop.exec()
        if not box:
            raise RuntimeError('JavaScript timeout')
        return box[0]

    def until(code, seconds=100):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if js(code):
                return
            pause()
        raise AssertionError(code + ' ' + str(js('document.body.innerText.slice(-2000)')))

    def click(text):
        assert js('(()=>{const e=[...document.querySelectorAll("button")].find(b=>b.offsetParent&&b.textContent.trim()===' + json.dumps(text) + ');if(!e||e.disabled)return false;e.click();return true})()'), text
        pause(100)

    def fill(label, value):
        selector = json.dumps('[aria-label="' + label + '"]')
        assert js('(()=>{const e=document.querySelector(' + selector + ');if(!e||e.disabled)return false;e.value=' + json.dumps(str(value)) + ';e.dispatchEvent(new Event("input",{bubbles:true}));e.dispatchEvent(new Event("change",{bubbles:true}));return true})()'), label
        pause(100)

    def select(label, name):
        selector = json.dumps('select[aria-label="' + label + '"]')
        until('!![...document.querySelector(' + selector + ').options].find(x=>x.textContent===' + json.dumps(name) + ')')
        assert js('(()=>{const e=document.querySelector(' + selector + ');e.value=[...e.options].find(x=>x.textContent===' + json.dumps(name) + ').value;e.dispatchEvent(new Event("change",{bubbles:true}));return true})()')
        pause(300)

    def idle():
        until('document.querySelector(".m07-page")?.getAttribute("aria-busy")==="false"')

    def capture(name, caption, state, settle=True):
        assert window.isMaximized()
        if settle:
            pause(300)
        window.view.repaint()
        app.processEvents()
        file = out / (name + '.png')
        pixmap = window.view.grab()
        assert pixmap.save(str(file))
        report['captures'].append({'id': name, 'chapterId': 'm07', 'file': str(file),
                                   'caption': caption, 'state': state, 'distribution': 'software-only', 'git': False,
                                   'maximized': True, 'window': [window.width(), window.height()],
                                   'image': [pixmap.width(), pixmap.height()], 'devicePixelRatio': pixmap.devicePixelRatio(),
                                   'theme': js('document.documentElement.dataset.theme'),
                                   'palette': js('document.documentElement.dataset.palette'),
                                   'fontSize': js('getComputedStyle(document.documentElement).fontSize'),
                                   'sha256': hashlib.sha256(file.read_bytes()).hexdigest(),
                                   'observed': js('(()=>{const p=document.querySelector(".m07-page");return {busy:p.getAttribute("aria-busy"),controls:document.querySelectorAll(".table-scroll tbody tr").length,synthesisCurves:document.querySelectorAll(".f0-plot [data-curve=synthesis]").length,resultButtons:document.querySelectorAll(".result-audios button").length,selectedTask:document.querySelector(".history-select[aria-pressed=true]")?.innerText||"",pending:document.body.innerText.includes("控制点或对齐方式有未应用修改"),status:[...document.querySelectorAll(".workbench-right-body .module-status")].map(x=>x.innerText)}})()')})
        print('Captured ' + name, flush=True)

    print(json.dumps({'captureDirectory': str(out)}), flush=True)
    try:
        until('!!document.querySelector(".home-page")&&document.fonts.status==="loaded"')
        click('设置')
        click('浅色')
        until('document.documentElement.dataset.theme==="light"')
        click('发声类型合成')
        until('!!document.querySelector(".m07-page")')
        click('打开音频目录')
        select('源音频', 'single-original.wav')
        select('目标音频', 'single-target.wav')
        until('[...document.querySelectorAll("button")].some(b=>b.textContent.trim()==="提取 F0"&&!b.disabled)')
        assert js('document.querySelectorAll(".table-scroll tbody tr").length') == 0
        capture('m07-input-loaded-light', '源与目标 WAV 已载入，两幅实际波形可见；F0 控制表和合成结果尚未建立。', 'two-inputs-loaded-no-analysis')

        fill('最高 F0 (Hz)', 600)
        click('提取 F0')
        idle()
        until('document.body.innerText.includes("分析完成，控制点尚未修改")')
        assert js('document.querySelectorAll(".table-scroll tbody tr").length') == 21
        assert js('document.querySelectorAll(".f0-plot [data-curve=synthesis]").length') == 0
        capture('m07-f0-extracted-light', '提取 F0 后出现源与目标控制曲线及 21 点数值表；合成音频仍为空。示例最高 F0 为 600 Hz。', 'analysis-complete-no-generation')

        js('document.querySelector(".m07-page details").open=true')
        pause(150)
        capture('m07-advanced-parameters-light', '展开高级分析参数，核对帧长、帧移、LPC 阶数、预加重和脉冲设置；改变这些值需要重新分析。', 'advanced-analysis-expanded')
        js('document.querySelector(".m07-page details").open=false')
        fill('源 F0 第 1 点', 110)
        until('document.body.innerText.includes("控制点或对齐方式有未应用修改")')
        assert js('[...document.querySelectorAll("button")].find(x=>x.textContent.trim()==="生成当前").disabled')
        capture('m07-f0-edit-pending-light', '修改第一项源 F0 后，控制表与图面立即更新，右栏提示未应用编辑，生成按钮暂时禁用。', 'f0-first-point-edited-pending-apply')
        click('应用编辑')
        idle()
        until('document.body.innerText.includes("F0 编辑已应用")')
        capture('m07-f0-edit-applied-light', '应用编辑成功后形成新的 F0 快照，提示 LPC、残差和脉冲继续复用，生成按钮恢复可用。', 'f0-edit-applied')

        # Reanalyze the unchanged example inputs before the demonstration group.
        click('提取 F0')
        idle()
        until('document.body.innerText.includes("分析完成，控制点尚未修改")')
        click('生成当前')
        if js('document.querySelector(".m07-page").getAttribute("aria-busy")==="true"'):
            capture('m07-generation-task-light', '当前九步连续统已提交，任务状态与批次完成数显示在右栏；成功组才会提供试听和保存。', 'actual-generation-in-progress', settle=False)
        idle()
        until('document.querySelectorAll(".result-audios button").length===10&&document.querySelectorAll(".f0-plot [data-curve=synthesis]").length===9')
        capture('m07-nine-step-result-light', '源到目标、仅 F0 变化的九步结果已完整成功，整组波形、各步生成控制 F0、任务描述和保存入口可见。', 'nine-step-group-complete')
        click('step05')
        until('document.querySelectorAll(".f0-plot [data-curve=synthesis]").length===1')
        pause(700)
        capture('m07-step05-comparison-light', '选择 step05 后显示该步波形及生成控制 F0；当前试听信息明确标出同一组的第五步。', 'selected-step05-single-wave-and-curve')
        # Select the complete group without introducing fake playback positions.
        click('整组试听')
        until('document.querySelectorAll(".f0-plot [data-curve=synthesis]").length===9')
        assert js('(()=>{const e=document.querySelector(".synthesized-plot .audio-transport button:nth-child(2)");e.click();return true})()')
        choices['directory'] = saved
        click('输出位置')
        click('保存完整组')
        until('document.body.innerText.includes("已保存本组完整文件和参数清单")')
        assert len(list(saved.glob('M07-*'))) == 1
        group = next(saved.glob('M07-*'))
        files = list(group.iterdir())
        assert len(files) == 12
        capture('m07-save-complete-light', '完整九步组已保存，右栏显示保存成功，顶部输出位置供核对；PNG 仍需独立导出。', 'complete-group-saved')
        report['savedFiles'] = [{'name': f.name, 'sizeBytes': f.stat().st_size,
                                 'sha256': hashlib.sha256(f.read_bytes()).hexdigest()} for f in files]
        report['checks'] += ['Two registered original inputs read without modifying them',
                             'Actual analysis, edit/apply, reanalysis, nine-step generation and complete-group save',
                             'Distinct UI state evidence retained for every capture',
                             'All screenshots captured from the maximized actual Qt window']
        assert input_hashes == {name: hashlib.sha256((inputs / name).read_bytes()).hexdigest() for name in input_hashes}
        assert not report['terminations']
        report['success'] = True
    except Exception as exc:
        report['error'] = repr(exc)
        raise
    finally:
        (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        js('window.onbeforeunload=null')
        window.closing = True
        window.close()
        window.page.deleteLater()
        app.processEvents()
        app.quit()
        print(json.dumps({'success': report['success'], 'report': str(out / 'report.json'),
                          'captures': len(report['captures'])}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
