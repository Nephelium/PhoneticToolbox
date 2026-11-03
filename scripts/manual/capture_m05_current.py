"""Current M05 manual images from isolated, maximized Qt and public test media."""
from __future__ import annotations

import base64
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from workbench_source import configure
from verification_artifacts import verification_run
from manual.capture_locale import CaptureLocale

os.environ['QT_QPA_PLATFORM'] = 'windows'
os.environ['QTWEBENGINE_CHROMIUM_FLAGS'] = '--disable-gpu --mute-audio --autoplay-policy=no-user-gesture-required'
for key in ('QT_ENABLE_HIGHDPI_SCALING', 'QT_SCALE_FACTOR', 'QT_SCREEN_SCALE_FACTORS'):
    os.environ.pop(key, None)


def main(args):
    configure()
    ui = CaptureLocale(args.locale, ('lip-extraction', 'workbench', 'components', 'common'))
    source_paths = [Path(__file__), ROOT / 'scripts/manual/capture_locale.py', args.frontend_dist / 'index.html',
                    ROOT / 'desktop/src/ptb_desktop/host.py', ROOT / 'desktop/src/ptb_desktop/presentation.py',
                    ROOT / 'tests/support/m05_offline_media.py', ROOT / 'tests/support/m05_r2_capture.js',
                    ROOT / 'frontend/src/components/MethodReferences.vue']
    source_paths += list((ROOT / 'frontend/src/modules/lip-extraction').glob('*.vue'))
    source_paths += [ROOT / f'frontend/src/i18n/locales/{language}/lip-extraction.json' for language in ('zh-CN', args.locale)]
    sources = {str(i): {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for i, path in enumerate(dict.fromkeys(source_paths))}
    from PyQt6 import sip
    from PyQt6.QtCore import QEvent, QEventLoop, QTimer, Qt
    from PyQt6.QtGui import QColor, QPainter, QPen
    from PyQt6.QtWidgets import QApplication
    from ptb_desktop.host import Workbench, register_scheme
    from ptb_worker.local_workspace import prepare_workspace

    register_scheme()
    app = QApplication(['PTB-M05-manual-images'])
    app.setQuitOnLastWindowClosed(False)
    recipe = 'Current Qt, native DPR1.5/maximized/page100/light; 6-frame FFV1 video from public NASA astronaut fixture and 1s sine; media-only synthetic camera for retained counters; no physical camera/microphone'
    with verification_run('m05-manual', args.locale, recipe) as (out, scratch):
        print(out, flush=True)
        db, cache = prepare_workspace(scratch / 'state', ROOT / 'backend/migrations')
        inputs = scratch / 'inputs'
        inputs.mkdir()
        subprocess.run([os.environ['PTB_M05_PYTHON'], '-B', '-X', 'utf8', str(ROOT / 'tests/support/m05_offline_media.py'), 'generate', str(inputs)], check=True, creationflags=subprocess.CREATE_NO_WINDOW)
        window = Workbench(args.frontend_dist.resolve(), test=True, jobs_path=db, local_files_root=cache, vocal_profile=scratch / 'vocal', start_module='M05')
        window.resize(window.screen().geometry().width(), window.screen().geometry().height())
        window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
        window.showMaximized()
        window.view.setZoomFactor(1)
        window.page.setAudioMuted(True)
        report = dict(success=False, captures=[], checks=[], errors=[], terminations=[], scope=recipe,
                      sources=sources, sourceChapter='m05', locale=args.locale)
        window.page.renderProcessTerminated.connect(lambda status, code: report['terminations'].append([status.name, code]))

        def pause(ms=100):
            loop = QEventLoop()
            QTimer.singleShot(ms, loop.quit)
            loop.exec()

        def js(code):
            wrapped = '(()=>{try{return {ok:true,value:eval(' + json.dumps(code) + ')??null};}catch(e){return {ok:false,error:String(e)};}})()'
            loop, values = QEventLoop(), []
            window.page.runJavaScript(wrapped, lambda value: (values.append(value), loop.quit()))
            QTimer.singleShot(6000, loop.quit)
            loop.exec()
            if not values or not values[0] or not values[0].get('ok'):
                raise RuntimeError('Qt JavaScript: ' + str(values))
            return values[0].get('value')

        def until(code, seconds=60):
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                if js(code):
                    return
                pause()
            raise AssertionError(code + '\n' + str(js('document.body.innerText.slice(-4000)')))

        def click(label):
            label = ui.label(label)
            text = json.dumps(label)
            expr = '[...document.querySelectorAll("button")].find(e=>e.offsetParent&&(e.getAttribute("aria-label")===' + text + '||e.textContent.trim()===' + text + '||e.getAttribute("role")==="tab"&&e.textContent.trim().startsWith(' + text + ')))'
            until('!!(' + expr + ')&&!(' + expr + ').disabled')
            js(expr + '.click()')
            pause(150)

        def mode(value):
            js('(()=>{const e=document.querySelector("select[aria-label="+CSS.escape(' + json.dumps(ui.label('采集模式')) + ')+"]");e.value=' + json.dumps(value) + ';e.dispatchEvent(new Event("change",{bubbles:true}));})()')
            pause(200)

        def fps(value):
            js('(()=>{const label=[...document.querySelectorAll(".lip-fields label")].find(e=>e.textContent.trim().startsWith(' + json.dumps(ui.label('请求帧率')) + '));const e=label.querySelector("input");e.value=' + json.dumps(str(value)) + ';e.dispatchEvent(new Event("input",{bubbles:true}));})()')
            pause(100)

        def capture(name, regions=None):
            if args.scene and name not in args.scene:
                return
            until('document.fonts.status==="loaded"&&!document.documentElement.hasAttribute("data-theme-switching")')
            pause(400)
            window.view.repaint()
            window.view.grab()
            pause(150)
            image = window.view.grab().toImage()
            size, dpr = [image.width(), image.height()], image.devicePixelRatio()
            assert size in ([2560, 1600], [2561, 1601]) and abs(dpr - 1.5) < .01, (size, dpr)
            assert window.isMaximized() and window.view.zoomFactor() == 1
            assert js('document.documentElement.dataset.theme') == 'light'
            assert js('document.querySelector(".app-shell").classList.contains("collapsed")')
            if size != [2560, 1600]:
                image = image.copy(0, 0, 2560, 1600)
            path = out / (name + '.png')
            assert image.save(str(path))
            boxes = {}
            for key, label in (regions or {}).items():
                box = js('(()=>{const e=[...document.querySelectorAll(".lip-fields label")].find(e=>e.textContent.trim().startsWith(' + json.dumps(ui.label(label)) + '));if(!e)return null;const r=e.getBoundingClientRect();return [r.x,r.y,r.width,r.height]})()')
                assert box and box[2] > 0 and box[3] > 0
                boxes[key] = {'selector': '.lip-fields label: ' + ui.label(label), 'box': box, 'units': 'css-px'}
            metrics = ui.metrics(js)
            report['captures'].append(dict(name=name, file=path.name, image=[2560, 1600], raw_image=size, dpr=dpr, pageZoom=1, maximized=window.isMaximized(), theme='light', navigationCollapsed=True, regions=boxes,
                                          locale=metrics['locale'], fontLocale=metrics['fontLocale'], fontFamily=metrics['fontFamily'], sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            print('Captured ' + name, flush=True)
            return image

        try:
            until('!!document.querySelector(".lip-page")&&document.fonts.status==="loaded"')
            ui.prepare_qt(js, until)
            click('设置')
            click('浅色')
            click('唇形提取')
            click('收起侧栏')
            until('document.querySelector(".app-shell").classList.contains("collapsed")')
            assert js('[...document.querySelector("select[aria-label="+CSS.escape(' + json.dumps(ui.label('采集模式')) + ')+"]").options].map(e=>e.textContent)') == [ui.label(s) for s in ['预览', '实时计算', '视频录制', '离线分析']]
            capture('m05-preview-ready')
            mode('realtime')
            capture('m05-realtime-ready')
            mode('record_then_analyze')
            fps(120)
            capture('m05-video-recording-ready')
            mode('preview')
            fps(30)
            filter_label = json.dumps(ui.label('特征点防抖'))
            js('[...document.querySelectorAll(".lip-fields label")].find(e=>e.textContent.includes(' + filter_label + ')).querySelector("input").click()')
            capture('m05-filter-off-ready', {'stabilization': '特征点防抖', 'cutoff': '截止频率'})
            js('[...document.querySelectorAll(".lip-fields label")].find(e=>e.textContent.includes(' + filter_label + ')).querySelector("input").click()')
            click('方法与引用')
            click('软件与代码来源')
            until('!!document.querySelector(".reference-row")')
            capture('m05-methods-software')
            click('关闭对话框')
            mode('offline')
            capture('m05-offline-ready')
            raw = base64.b64encode((inputs / 'first.mkv').read_bytes()).decode('ascii')
            js('(()=>{const d=new DataTransfer();d.items.add(new File([Uint8Array.from(atob(' + json.dumps(raw) + '),c=>c.charCodeAt(0))],"唇形示例.mkv"));const e=document.querySelector("input[type=file]");e.files=d.files;e.dispatchEvent(new Event("change",{bubbles:true}));})()')
            click('分析所选文件')
            until('document.querySelector(".lip-page").getAttribute("aria-busy")==="false"')
            until('document.querySelector(".lip-recording-preview")?.videoWidth>0&&document.querySelector(".lip-recording-preview")?.readyState>=2')
            js('document.querySelector(".lip-recording-preview").currentTime=.3')
            until('Number(document.querySelector("[aria-label="+CSS.escape(' + json.dumps(ui.label('回放帧')) + ')+"]").value)>=2')
            until('document.querySelector(".lip-metrics strong")?.textContent!=="—"')
            capture('m05-offline-replay')
            report['checks'].append('Real legacy analysis of six public fixture frames, original picture and lip replay shown together')
            click('放弃未保存内容')
            click('放弃')
            js((ROOT / 'tests/support/m05_r2_capture.js').read_text('utf8'))
            portrait = 'data:image/png;base64,' + base64.b64encode((ROOT / 'output/validation/m05/inputs/astronaut.png').read_bytes()).decode('ascii')
            js('window.installM05R2Capture(' + json.dumps(portrait) + ').then(()=>window.manualFixtureReady=true)')
            until('window.manualFixtureReady')
            mode('record_then_analyze')
            click('开始录制')
            until('Number(document.querySelector(".lip-state[data-phase] span:nth-child(2)").textContent.match(/[0-9]+/)?.[0])>8')
            pause(1200)
            click('结束录制')
            until('document.querySelector(".lip-state").dataset.phase==="ready"&&document.querySelector(".lip-page").getAttribute("aria-busy")==="false"')
            before = js('document.querySelector(".lip-state").textContent')
            mode('offline')
            assert js('document.querySelector(".lip-state").textContent') == before
            js('document.querySelector(".lip-stat-help").open=true')
            expected_level = ui.key('modules.lip-extraction.stoppedInputLevel', {'level': '-20.0 dBFS'})
            assert js('document.querySelector(".lip-input-level span:last-child")?.textContent') == expected_level
            capture('m05-capture-statistics')
            report['checks'].append('Media-only synthetic recording statistics retained in offline mode; five definitions expanded, no physical capture')
            assert not report['terminations']
            report['success'] = True
        except Exception as error:
            report['errors'].append(repr(error))
            window.view.grab().save(str(out / 'failed.png'))
            raise
        finally:
            report['sourceChangesDuringRun'] = [item['path'] for item in sources.values() if hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() != item['sha256']]
            report['page'] = js('document.body.innerText')
            (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', 'utf8')
            js('window.onbeforeunload=null')
            window.closing = True
            window.close()
            app.processEvents()
            sip.delete(window)
            app.processEvents()
            sip.delete(app)
        print(json.dumps(dict(success=True, out=str(out), captures=len(report['captures']))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--locale', choices=('zh-CN', 'en', 'ja'), default='zh-CN')
    parser.add_argument('--frontend-dist', type=Path, default=ROOT / 'frontend/dist')
    parser.add_argument('--scene', action='append', default=[], choices=('m05-methods-software', 'm05-capture-statistics'), help='Only retain selected correction scenes, using the complete original setup')
    main(parser.parse_args())
