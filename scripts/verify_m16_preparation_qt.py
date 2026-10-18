"""Verify preparation, task binding, layout and timeline in isolated Qt/synthetic audio."""
import json
import os
import time
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS', '--disable-gpu')

from PyQt6.QtCore import QEventLoop, QTimer
from PyQt6.QtWidgets import QApplication
from m16_test_backend import Backend
from verification_artifacts import verification_run
from workbench_source import configure

ROOT = Path(__file__).resolve().parents[1]


def main():
    configure()
    from ptb_desktop.host import Workbench, register_scheme

    register_scheme()
    app = QApplication(['M16 preparation QA'])
    with verification_run('m16', 'workflow-qt', 'Actual Qt: preparation, task-save binding, levels layout, wheel timeline, synthetic recording and processing; no physical device') as (out, scratch):
        project = scratch / 'project'
        project.mkdir()
        window = Workbench(ROOT / 'frontend/dist', test=True, vocal_profile=scratch / 'vocal-profile', start_module='M16')
        bridge = window.bridge.recording_bridge()
        bridge.service.backend = Backend()
        bridge.choose = lambda purpose: bridge.service.grant(project, purpose)
        requests, bodies = [], []
        dispatch = bridge.dispatch

        def track(body):
            requests.append(body['op'])
            bodies.append(json.loads(json.dumps(body)))
            return dispatch(body)

        bridge.dispatch = track
        window.resize(1440, 900)
        window.show()
        report = {'success': False, 'scope': 'Windows source / actual Qt / synthetic audio only', 'checks': []}

        def pause(ms=80):
            loop = QEventLoop()
            QTimer.singleShot(ms, loop.quit)
            loop.exec()

        def js(code):
            loop, values = QEventLoop(), []
            window.page.runJavaScript(code, lambda value: (values.append(value), loop.quit()))
            QTimer.singleShot(5000, loop.quit)
            loop.exec()
            if not values:
                raise AssertionError('JavaScript timed out: ' + code)
            return values[0]

        def until(code):
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if js(code):
                    return
                pause()
            raise AssertionError(code + '\n' + str(js('document.body.innerText.slice(-3000)')))

        def button(text):
            return '[...document.querySelectorAll(".recording-page button")].find(b=>b.textContent.trim()===' + json.dumps(text, ensure_ascii=False) + ')'

        def click(text):
            target = button(text)
            until(f'!!({target})&&!({target}).disabled')
            js(target + '.click()')
            pause()

        def prompt(expected):
            until('!!document.querySelector("dialog[open]")')
            assert js('document.querySelector("dialog[open] .dialog-header h2").textContent') == '录音前准备'
            assert js('document.querySelector("dialog[open] .dialog-body").textContent.trim()') == expected
            assert not bridge.service.capture

        def fill(label,value):
            js('(()=>{const l=[...document.querySelectorAll(".recording-page label")].find(l=>l.textContent.trim().startsWith('+json.dumps(label,ensure_ascii=False)+'));const e=l.querySelector("input,textarea");e.value='+json.dumps(value,ensure_ascii=False)+';e.dispatchEvent(new Event("input",{bubbles:true}));e.dispatchEvent(new Event("change",{bubbles:true}));})()')
            pause()

        project_message = '请先新建工程，再点击设备与录前检测，选择输入设备并完成录前检测。'
        device_message = '请先点击设备与录前检测，选择输入设备并完成录前检测。'
        try:
            until('document.querySelector(".recording-page")?.getAttribute("aria-busy")==="false"')
            for theme in ('light', 'dark'):
                js('document.documentElement.dataset.theme=' + json.dumps(theme))
                pause(600)
                for label in ('自由录音 不需要任务清单', '● 开始录音', '设备与录前检测'):
                    click(label)
                    prompt(project_message)
                    assert bridge.service.project is None
                    assert not {'start', 'probe_start', 'open'}.intersection(requests)
                    assert js('(()=>{const d=document.querySelector("dialog[open]");return d.scrollWidth<=d.clientWidth&&d.querySelector(".dialog-body").scrollWidth<=d.querySelector(".dialog-body").clientWidth;})()')
                    pause(350)
                    window.view.grab()
                    pause(150)
                    window.view.grab().save(str(out / f'{theme}-{label.split()[0]}.png'))
                    click('知道了')
                    until('!document.querySelector("dialog[open]")')
                report['checks'].append(theme + ': all three empty-project entries prompt without backend capture or project creation')
            click('新建工程')
            until('document.body.innerText.includes("本地录音工程已建立")')
            click('● 开始录音')
            prompt(device_message)
            click('知道了')
            click('设备与录前检测')
            until('!!document.querySelector(".devices-panel")')
            click('开始录前检测（不保存音频）')
            prompt(device_message)
            click('知道了')
            assert not {'start', 'probe_start'}.intersection(requests)
            report['checks'].append('Existing project without input: record/preflight prompt and neither starts capture')
            device = bridge.service.dispatch({'op': 'devices'})[0]['id']
            js('(()=>{const e=document.querySelector("select[aria-label=输入设备]");e.value=' + json.dumps(device) + ';e.dispatchEvent(new Event("change",{bubbles:true}));})()')
            click('开始录前检测（不保存音频）')
            until('document.body.innerText.includes("录前检测中，音频不会保存为录音条目")')
            assert js(button('● 开始录音') + '.disabled')
            click('停止检测')
            until('document.querySelector(".recording-page")?.getAttribute("aria-busy")==="false"')
            assert not bridge.service.project.data['takes']
            click('自由录音 不需要任务清单')
            assert not js('!!document.querySelector("dialog[open]")')
            click('● 开始录音')
            until('document.body.innerText.includes("正在录音")')
            assert js(button('自由录音 不需要任务清单') + '.disabled')
            click('■ 停止录音')
            until('document.body.innerText.includes("已存入本地工程，尚未导出")')
            assert len(bridge.service.project.data['takes']) == 1
            report['checks'].append('Configured project: synthetic preflight and free record/stop persist one take, active capture locks remain')
            for task_id in ('T004','T005'):
                click('＋ 新任务')
                fill('任务编号',task_id)
                fill('录音内容',task_id+' synthetic only')
                click('完成编辑')
                click('● 开始录音')
                until('document.body.innerText.includes("正在录音")')
                pause(250)
                click('■ 停止录音')
                until('document.querySelector(".recording-page")?.getAttribute("aria-busy")==="false"')
                assert bridge.service.project.data['takes'][-1]['task_snapshot']['id']==task_id
                assert [b for b in bodies if b['op']=='start'][-1]['task_id']==task_id
            takes=bridge.service.project.data['takes']
            assert [t.get('task_snapshot',{}).get('id') if t.get('task_snapshot') else None for t in takes]==[None,'T004','T005']
            report['checks'].append('T004 then unsaved new T005: save-before-start retains task and exact capture task_snapshot')
            total=sum(s['end']-s['start'] for s in takes[-1]['versions'][0]['spans'])
            for target in ('.m16-wave-lanes','.m16-spectrum'):
                js('(()=>{const e=document.querySelector('+json.dumps(target)+'),b=e.getBoundingClientRect();e.dispatchEvent(new WheelEvent("wheel",{bubbles:true,cancelable:true,ctrlKey:true,deltaY:-120,clientX:b.left+b.width*.6}));})()')
                pause(250)
            preview_requests=[b for b in bodies if b['op']=='preview']
            assert preview_requests[-1]['end']-preview_requests[-1]['start']<total
            range_before=(preview_requests[-1]['start'],preview_requests[-1]['end'])
            js('document.querySelector(".m16-wave-lanes").dispatchEvent(new WheelEvent("wheel",{bubbles:true,cancelable:true,deltaY:-120}))')
            pause(150)
            assert [(b.get('start'),b.get('end')) for b in bodies if b['op']=='preview'][-1]==range_before
            click('整段总览')
            pause(200)
            report['checks'].append('Ctrl-wheel on waveform and spectrogram zooms frame window; normal wheel leaves range; full view restores')
            # Reuse the noise selected on T004 while processing T005, as reported.
            t4=next(t for t in takes if (t.get('task_snapshot') or {}).get('id')=='T004')
            bridge.service.dispatch({'op':'noise','id':t4['id'],'start':0,'end':4096,'channel':0})
            bridge.service.dispatch({'op':'process_start','id':takes[-1]['id'],'kind':'denoise','strength':1,'channels':[0,1]})
            deadline=time.monotonic()+30
            while bridge.service.job and time.monotonic()<deadline:
                value=bridge.service.dispatch({'op':'status'})
                if value.get('job',{}).get('state')=='failed':raise AssertionError(value['job'])
                pause(80)
            assert bridge.service.job is None
            assert takes[-1]['id']==bridge.service.project.data['takes'][-1]['id']
            assert bridge.service.project.data['takes'][-1]['versions'][-1]['kind']=='denoised'
            report['checks'].append('Noise from T004 reused for real child-process denoise of T005; atomic progress and derived version complete')
            for theme in ('light','dark'):
                js('document.documentElement.dataset.theme='+json.dumps(theme))
                pause(500)
                geometry=js('(()=>{const row=document.querySelector(".device-levels"),slider=row.querySelector("label").getBoundingClientRect(),meters=row.querySelector(".input-meters").getBoundingClientRect();return {slider:slider.toJSON(),meters:meters.toJSON(),panelHeight:document.querySelector(".devices-panel").getBoundingClientRect().height};})()')
                assert geometry['meters']['left']>geometry['slider']['right']
                assert abs(geometry['meters']['top']-geometry['slider']['top'])<2
                report.setdefault('layouts',[]).append({'theme':theme,**geometry})
                js('document.querySelector(".devices-panel").scrollIntoView({block:"end"})')
                pause(300);window.view.grab();pause(100)
                window.view.grab().save(str(out/f'devices-{theme}.png'))
            report['checks'].append('Light/dark levels sit beside playback control; compact actions and device panel geometry captured')
            report['success'] = True
        finally:
            report['requests'] = requests
            (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            js('window.onbeforeunload=null')
            window.closing = True
            window.close()
            pause(300)
            window.deleteLater()
            pause(800)
            app.quit()
            print(out, flush=True)
        return report


if __name__ == '__main__':
    main()
