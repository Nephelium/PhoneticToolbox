"""Owned, maximized Windows Qt M16 screenshots with synthetic audio only.

Never opens a physical audio endpoint or an existing user project. Creates a
new ignored capture directory; source captures and failed attempts are retained.
Media registration belongs to the main manual author, not this script.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
PATHS = ('desktop/src', 'backend/src', 'packages/phonetic_core/src', 'scripts')
for relative in PATHS:
    sys.path.insert(0, str(ROOT / relative))
os.environ['PYTHONPATH'] = os.pathsep.join(str(ROOT / p) for p in PATHS)
os.environ.setdefault('QT_QPA_PLATFORM', 'windows')
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS', '--mute-audio --autoplay-policy=no-user-gesture-required')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    import numpy as np
    import soundfile as sf
    from PyQt6.QtCore import QEventLoop, QTimer, Qt
    from PyQt6.QtWidgets import QApplication
    from ptb_desktop.host import Workbench, register_scheme
    from ptb_desktop.recording.capture import Capture
    from ptb_desktop.recording.storage import read_range
    from m16_test_backend import Backend

    class DemonstrationBackend(Backend):
        def query_hostapis(self):
            return [{'name': '合成测试后端'}]

        def query_devices(self):
            return [
                {'name': '测试演示·双通道合成输入输出', 'hostapi': 0,
                 'max_input_channels': 2, 'max_output_channels': 2, 'default_samplerate': 48000},
                {'name': '测试演示·单通道合成输入', 'hostapi': 0,
                 'max_input_channels': 1, 'max_output_channels': 0, 'default_samplerate': 48000},
            ]

    out = ROOT / 'output/manual-work/m16-captures' / uuid4().hex
    out.mkdir(parents=True)
    project = out / '测试演示-录音工程'
    project.mkdir()
    other = out / '测试演示-临时工程'
    other.mkdir()
    exports = out / 'exports'
    exports.mkdir()
    choice = {'project': project}
    register_scheme()
    app = QApplication(['PTB-owned-M16-manual-capture'])
    window = Workbench(ROOT / 'frontend/dist', test=True, jobs_path=None,
                       local_files_root=None, vocal_profile=out / 'vocal-profile', start_module='M16')
    bridge = window.bridge.recording_bridge()
    bridge.service.backend = DemonstrationBackend()
    bridge.choose = lambda purpose: bridge.service.grant(exports if purpose == 'export' else choice['project'], purpose)
    window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
    window.showMaximized()
    window.page.setAudioMuted(True)
    report = {'success': False, 'chapterId': 'm16', 'out': str(out),
              'scope': 'Actual maximized hidden Windows Qt and QWebChannel, synthetic endpoints and input/output only; no physical microphone, speaker, user recording, database, hardware latency or DPI claim',
              'captures': [], 'checks': [], 'terminations': [],
              'input': {'type': 'synthetic', 'seed': 1602, 'sampleRate': 48000,
                        'microphone': 'Gaussian noise SD .012 plus .15 sine 180Hz after .3s',
                        'secondChannel': '.45 sine 90Hz, explicitly assigned EGG role for channel-preservation demonstration; not a physiological EGG'},
              'sourceSnapshot': [{'path': str(p.relative_to(ROOT)), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                                 for p in sorted((ROOT / 'frontend/dist').rglob('*'))
                                 if p.is_file() and p.suffix in ('.html', '.js', '.css')]}
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

    def until(code, seconds=40):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if js(code):
                return
            pause()
        raise AssertionError(code + '\n' + str(js('document.body.innerText.slice(-4000)')))

    def click(text):
        expr = '[...document.querySelectorAll(".recording-page button")].find(b=>b.offsetParent&&b.textContent.trim()===' + json.dumps(text) + ')'
        until('!!(' + expr + ')&&!(' + expr + ').disabled')
        js(expr + '.click()')
        pause(120)

    def field(label, value, tag='input,textarea'):
        until('(()=>{const l=[...document.querySelectorAll(".recording-page label")].find(l=>l.textContent.trim().startsWith(' + json.dumps(label) + '));const e=l?.querySelector(' + json.dumps(tag) + ');return !!e&&!e.disabled})()')
        assert js('(()=>{const l=[...document.querySelectorAll(".recording-page label")].find(l=>l.textContent.trim().startsWith(' + json.dumps(label) + '));const e=l?.querySelector(' + json.dumps(tag) + ');if(!e||e.disabled)return false;e.value=' + json.dumps(str(value)) + ';e.dispatchEvent(new Event("input",{bubbles:true}));e.dispatchEvent(new Event("change",{bubbles:true}));return true})()'), label
        pause(150)

    def upload(name, content):
        payload = base64.b64encode(content.encode('utf-8')).decode()
        js('(()=>{const b=Uint8Array.from(atob(' + json.dumps(payload) + '),c=>c.charCodeAt(0));const d=new DataTransfer();d.items.add(new File([b],' + json.dumps(name) + ',{type:"text/csv"}));const e=document.querySelector(".recording-page input[type=file]");e.files=d.files;e.dispatchEvent(new Event("change",{bubbles:true}));})()')
        until('!!document.querySelector("[aria-label=任务表格导入预览]")')

    def position(selector=None):
        if selector:
            js('document.querySelector(' + json.dumps(selector) + ')?.scrollIntoView({block:"end"})')
        else:
            js('document.querySelector(".recording-page .module-toolbar")?.scrollIntoView({block:"start"})')
        pause(250)

    def capture(name, caption, alt, anchor, state, selector=None):
        assert window.isMaximized()
        position(selector)
        window.view.repaint()
        app.processEvents()
        window.view.grab()
        pause(200)
        app.processEvents()
        pixmap = window.view.grab()
        file = out / (name + '.png')
        assert pixmap.save(str(file))
        report['captures'].append({'id': name, 'chapterId': 'm16', 'file': str(file),
                                  'caption': caption, 'alt': alt, 'anchor': anchor, 'state': state,
                                  'distribution': 'software-only', 'git': False, 'sourceType': 'synthetic-test-demo',
                                  'maximized': True, 'window': [window.width(), window.height()],
                                  'frameGeometry': [window.frameGeometry().width(), window.frameGeometry().height()],
                                  'image': [pixmap.width(), pixmap.height()], 'devicePixelRatio': pixmap.devicePixelRatio(),
                                  'theme': js('document.documentElement.dataset.theme'),
                                  'palette': js('document.documentElement.dataset.palette'),
                                  'fontSize': js('getComputedStyle(document.documentElement).fontSize'),
                                  'sha256': hashlib.sha256(file.read_bytes()).hexdigest(),
                                  'observed': js('(()=>{const p=document.querySelector(".recording-page");return {notice:document.querySelector(".sidebar-notice")?.innerText||"",error:document.querySelector(".module-status.error")?.innerText||"",waves:document.querySelectorAll(".wave-lane").length,tasks:document.querySelectorAll(".task-row").length,version:document.querySelector(".version-control select")?.selectedOptions[0]?.textContent||"",recoveries:document.querySelectorAll(".recovery-list button").length,dialog:document.querySelector(".m16-dialog")?.getAttribute("aria-label")||"",text:p.innerText}})()')})
        print('Captured ' + name, flush=True)

    print(json.dumps({'captureDirectory': str(out)}), flush=True)
    try:
        until('!!document.querySelector(".recording-page")&&document.fonts.status==="loaded"')
        js('document.documentElement.dataset.theme="light"')
        click('新建工程')
        until('document.body.innerText.includes("本地录音工程已建立")')
        capture('m16-project-created', '测试演示：新工程已建立，左侧显示工程名称和成功提示，录音与历史尚为空。', '测试演示的最大化录音工作台，新工程成功提示和空任务列表', 'm16-project-controls', 'new-project-created')

        click('导入任务')
        upload('测试演示任务.csv', '任务编号,任务名称,录音内容,文件名,次数,分组,备注,启用\nSYN001,合成信号演示,测试演示：合成噪声与正弦输入；无实体麦克风,syn-001,1,演示,非自然录音,是\nSYN002,重复任务演示,测试演示：同一文本重复两次,syn-002,2,演示,非自然录音,是\n')
        capture('m16-import-preview', '测试演示：CSV 导入预览自动映射八列，第二行按次数展开，确认导入前可核对三条任务。', '测试演示 CSV 列映射与三条展开任务，确认导入按钮可用', 'm16-task-import', 'csv-mapped-preview')
        click('确认导入')
        assert len(bridge.service.project.data['tasks']) == 0
        click('保存清单')
        assert len(bridge.service.project.data['tasks']) == 3
        capture('m16-tasks-saved', '测试演示：确认导入后点击保存清单，三条任务已写入工程，当前朗读卡显示第一条材料。', '测试演示的三条任务已保存，左栏出现保存提示，右栏显示第一条录音内容', 'm16-task-import', 'task-list-committed')
        report['checks'].append('CSV maps eight columns, repeat expansion creates three tasks; import changes draft only; Save commits all three')

        click('设备与录前检测')
        devices = bridge.service.dispatch({'op': 'devices'})
        field('输入设备', devices[1]['id'], 'select')
        field('扬声器 / 耳机', devices[0]['id'], 'select')
        assert js('document.querySelector(".device-grid input[type=number]").value') == '1'
        capture('m16-single-channel', '测试演示：选择只有一个输入的合成设备后自动回退单声道，输入和独立试听输出分别设置。', '测试演示的单通道合成设备、一个物理输入角色、采样率和独立输出选择', 'm16-device-controls', 'single-input-fallback', '.devices-panel')
        field('输入设备', devices[0]['id'], 'select')
        field('物理输入 2 角色', 'egg', 'select')
        click('开始录前检测（不保存音频）')
        pause(500)
        until('document.querySelectorAll(".meter-row").length===2')
        capture('m16-preflight', '测试演示：合成输入的录前检测显示两通道电平，第二道被指定为 EGG 角色；检测不创建录音。此正弦信号不代表生理 EGG。', '测试演示录前检测，两条合成通道电平和停止检测按钮，第二道仅演示 EGG 角色', 'm16-preparation', 'synthetic-preflight-no-take', '.devices-panel')
        click('停止检测')
        assert not bridge.service.project.data['takes']
        click('收起')
        report['checks'].append('Synthetic one-input fallback and two-input role recheck; preflight creates no take')

        click('● 开始录音')
        pause(900)
        capture('m16-recording-active', '测试演示：合成输入正在录音，朗读卡和停止按钮显示采集中状态，原始采样在隔离工程落盘。', '测试演示录音中指示、可用停止按钮、实时波形和已确认帧数', 'm16-capture-controls', 'synthetic-recording-active')
        pause(1100)
        click('■ 停止录音')
        until('document.body.innerText.includes("已存入本地工程，尚未导出")')
        first = bridge.service.project.data['takes'][0]
        raw = read_range(project, first['versions'][0]['spans'], 0, 4800)
        assert first['config']['roles'] == ['microphone', 'egg']
        assert js('document.querySelectorAll(".wave-lane").length') == 1
        capture('m16-raw-saved', '测试演示：停止后第一条合成录音自动存入工程，历史与原始版本已出现，默认仅显示左声道；此时仍未导出 WAV。', '测试演示的原始录音保存成功、第一条历史、单道波形和 Praat 语谱图', 'm16-capture-controls', 'raw-internal-save-no-wav')
        click('▶ 播放')
        assert bridge.service.player is not None
        capture('m16-playback-active', '测试演示：选择合成输出后播放已保存录音，播放按钮切换为停止播放；测试后端不会驱动实体扬声器。', '测试演示录音试听中，停止播放按钮和当前版本选择', 'm16-capture-controls', 'synthetic-output-playback-active')
        if js('document.body.innerText.includes("停止播放")'):
            click('■ 停止播放')
        assert bridge.service.player is None
        report['checks'].append('Start/stop through actual Qt bridge; task snapshot and raw f32 saved; first channel waveform is default')
        report['checks'].append('Playback uses explicit synthetic OutputStream only, Stop releases owned player')

        js('[...document.querySelectorAll(".plot-toolbar label")].find(l=>l.textContent.includes("显示所有声道")).querySelector("input").click()')
        field('起点（帧）', 48000)
        field('终点（帧）', 96000)
        capture('m16-selection', '测试演示：显示所有声道后用整数帧设置共同选区，两道波形与语谱共用相同切点；第二道合成信号用于演示保留规则。', '测试演示双通道波形中的共同选区及 48000 到 96000 帧数输入', 'm16-editing', 'shared-integer-frame-selection')
        before = sum(s['end'] - s['start'] for s in first['versions'][0]['spans'])
        click('删除选区')
        first = bridge.service.project.data['takes'][0]
        assert sum(s['end'] - s['start'] for s in first['versions'][first['head']]['spans']) == before - 48000
        capture('m16-edit-version', '测试演示：删除选区后整段缩短一秒，版本下拉新增编辑版本，左侧确认编辑已保存且原始录音保留。', '测试演示的同步删除结果、新编辑版本、缩短后的时长和工程保存提示', 'm16-editing', 'delete-created-version')
        click('撤销')
        assert bridge.service.project.data['takes'][0]['head'] == 0
        report['checks'].append('Shared [48000,96000) frame deletion shortens both channels by 48000 frames; undo returns raw head')

        field('起点（帧）', 0)
        field('终点（帧）', 10000)
        click('将选区设为噪声样本')
        capture('m16-noise-sample', '测试演示：取合成输入开始处的噪声作为样本，画面记录来源版本、通道及帧范围，EGG 角色默认不参与处理。', '测试演示的噪声样本记录、选区和仅音频道可勾选的处理选项', 'm16-processing', 'noise-sample-selected')
        click('整段降噪')
        until('document.body.innerText.includes("后台处理已结束")', 90)
        first = bridge.service.project.data['takes'][0]
        assert first['versions'][first['head']]['kind'] == 'denoised'
        processed = read_range(project, first['versions'][first['head']]['spans'], 0, 4800)
        assert np.array_equal(raw[:, 1], processed[:, 1])
        assert not np.array_equal(raw[:, 0], processed[:, 0])
        capture('m16-denoised-version', '测试演示：真实谱减处理完成后新增降噪版本，噪声样本记录仍可见；仅勾选的音频道改变，EGG 角色合成通道保持原采样。', '测试演示降噪版本的波形语谱、后台完成提示和 EGG 保留标记', 'm16-processing', 'actual-spectral-subtraction-completed')
        report['checks'].append('Real child-process spectral subtraction on synthetic PCM: microphone changed, EGG-role samples exactly unchanged')

        click('保存 / 批量导出')
        field('导出范围', 'selected', 'select')
        field('WAV 格式', 'PCM_24', 'select')
        capture('m16-export-options', '测试演示：导出对话框选择未跳过任务的选用录音与 24-bit PCM，缺录任务不会生成空 WAV，导出另建结果目录。', '测试演示 WAV 导出范围、当前处理版本、24-bit PCM 格式和缺录统计', 'm16-export-workflow', 'export-selected-pcm24-options')
        click('选择目录并导出')
        until('document.body.innerText.includes("已导出 1 / 1")', 60)
        waves = list(exports.rglob('*.wav'))
        assert len(waves) == 1
        info = sf.info(waves[0])
        assert (info.samplerate, info.channels, info.subtype) == (48000, 2, 'PCM_24')
        assert list(exports.rglob('manifest.json')) and list(exports.rglob('manifest.csv'))
        capture('m16-export-complete', '测试演示：导出成功后左侧给出新目录和 WAV 文件名，工程内历史与降噪版本继续保留；导出 JSON 与 CSV 清单已回读。', '测试演示的成功导出清单、结果目录、WAV 文件名和仍保留的录音工程', 'm16-export-workflow', 'wav-export-and-manifests-completed')
        report['checks'].append('Selected export produces one WAV and JSON/CSV manifests; actual WAV readback is 48000Hz two-channel PCM24')

        field('音量增益', 24)
        click('应用增益')
        until('document.body.innerText.includes("后台处理已结束")', 90)
        gain_take = bridge.service.project.data['takes'][0]
        assert gain_take['versions'][gain_take['head']]['kind'] == 'gain'
        gained = read_range(project, gain_take['versions'][gain_take['head']]['spans'], 24000, 48000)
        assert np.max(np.abs(gained[:, 0])) > 1
        capture('m16-gain-version', '测试演示：为演示超幅分支，显式应用 +24 dB 数字增益并生成新版本；该数值仅用于测试，不是正式录音建议。', '测试演示的 +24 dB 数字增益新版本和原始保留提示', 'm16-processing', 'gain-version-created-for-overrange-demonstration')
        click('保存 / 批量导出')
        field('导出范围', 'current', 'select')
        field('WAV 格式', 'PCM_16', 'select')
        click('选择目录并导出')
        until('document.body.innerText.includes("已导出 0 / 1")', 60)
        assert len(list(exports.rglob('*.wav'))) == 1
        capture('m16-pcm-overrange-failure', '测试演示：超满幅增益版本导出 PCM16 被拒绝，左侧记录失败原因；先前成功 WAV 与工程历史保留，可降低增益或改用 FLOAT 后重试。', '测试演示的 PCM 超量化范围失败清单和已保留的原始工程版本', 'm16-error-actions', 'pcm-overrange-refused-with-history-preserved')
        report['checks'].append('Explicit +24dB test gain creates values above ±1; PCM16 export refuses, produces zero new complete WAV and retains earlier success')

        # Build a safe, uncommitted durable prefix to demonstrate recovery without
        # crashing the host or accessing a physical device. The real scanner and
        # UI Recover action consume this isolated fixture after project reopen.
        config = copy.deepcopy(first['config'])
        prefix = Capture(project, config, first['task_snapshot'], backend=bridge.service.backend)
        prefix.start(open_stream=False)
        rng = np.random.default_rng(1606)
        for _ in range(8):
            prefix.submit(rng.normal(0, .01, (4096, 2)).astype(np.float32))
        stopped = prefix.stop()
        assert stopped['frames'] == 32768
        choice['project'] = other
        click('新建工程')
        choice['project'] = project
        click('打开工程')
        until('document.body.innerText.includes("发现可恢复录音")')
        capture('m16-recovery-found', '测试演示：重新打开含未提交前缀的隔离工程，真实恢复扫描显示 32768 个已落盘帧及末尾可能缺失提示；此状态由安全测试夹具形成。', '测试演示工程重新打开后发现可恢复录音、32768 帧和恢复按钮', 'm16-interrupted', 'safe-durable-prefix-found')
        click('恢复')
        until('document.body.innerText.includes("已恢复已落盘片段")')
        assert len(bridge.service.project.data['takes']) == 2
        assert not bridge.service.project.recoveries
        recovered = bridge.service.project.data['takes'][-1]
        assert sum(s['end'] - s['start'] for s in recovered['versions'][0]['spans']) == 32768
        field('录音历史', recovered['id'], 'select')
        until('document.body.innerText.includes("异常中断恢复，仅包含已落盘且哈希通过的前缀，末尾可能缺失")')
        capture('m16-recovery-complete', '测试演示：点击恢复后前缀成为新的不完整录音条目，原录音继续保留，界面提示已恢复已落盘片段。', '测试演示恢复成功提示、两条录音历史和末尾可能缺失的警告', 'm16-interrupted', 'durable-prefix-restored-new-take')
        report['checks'].append('Isolated uncommitted 32768-frame fixture scanned by actual Project reopen; UI recovery creates new incomplete take and preserves original')
        report['success'] = True
        report['projectSummary'] = bridge.service.view()
    finally:
        report['capturesCount'] = len(report['captures'])
        (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        window.closing = True
        window.close()
        pause(300)
        app.quit()
        print(json.dumps({'success': report['success'], 'report': str(out / 'report.json'), 'captures': len(report['captures'])}), flush=True)
    return report


if __name__ == '__main__':
    main()
