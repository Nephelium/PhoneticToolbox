"""Current M05 manual images from isolated, maximized Qt and public test media."""
from __future__ import annotations

import base64
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

os.environ['QT_QPA_PLATFORM'] = 'windows'
os.environ['QTWEBENGINE_CHROMIUM_FLAGS'] = '--disable-gpu --mute-audio --autoplay-policy=no-user-gesture-required'
for key in ('QT_ENABLE_HIGHDPI_SCALING', 'QT_SCALE_FACTOR', 'QT_SCREEN_SCALE_FACTORS'):
    os.environ.pop(key, None)


def main():
    configure()
    from PyQt6.QtCore import QEvent, QEventLoop, QTimer, Qt
    from PyQt6.QtGui import QColor, QPainter, QPen
    from PyQt6.QtWidgets import QApplication
    from ptb_desktop.host import Workbench, register_scheme
    from ptb_worker.local_workspace import prepare_workspace

    register_scheme()
    app = QApplication(['PTB-M05-manual-images'])
    recipe = 'Current Qt, native DPR1.5/maximized/page100/light; 6-frame FFV1 video from public NASA astronaut fixture and 1s sine; media-only synthetic camera for retained counters; no physical camera/microphone'
    with verification_run('m05-manual', 'screenshots-20261018', recipe) as (out, scratch):
        print(out, flush=True)
        db, cache = prepare_workspace(scratch / 'state', ROOT / 'backend/migrations')
        inputs = scratch / 'inputs'
        inputs.mkdir()
        subprocess.run([os.environ['PTB_M05_PYTHON'], '-B', '-X', 'utf8', str(ROOT / 'tests/support/m05_offline_media.py'), 'generate', str(inputs)], check=True, creationflags=subprocess.CREATE_NO_WINDOW)
        window = Workbench(ROOT / 'frontend/dist', test=True, jobs_path=db, local_files_root=cache, vocal_profile=scratch / 'vocal', start_module='M05')
        window.resize(window.screen().geometry().width(), window.screen().geometry().height())
        window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
        window.showMaximized()
        window.view.setZoomFactor(1)
        window.page.setAudioMuted(True)
        report = dict(success=False, captures=[], checks=[], errors=[], terminations=[], scope=recipe)
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
            text = json.dumps(label)
            expr = '[...document.querySelectorAll("button")].find(e=>e.offsetParent&&(e.getAttribute("aria-label")===' + text + '||e.textContent.trim()===' + text + '||e.getAttribute("role")==="tab"&&e.textContent.trim().startsWith(' + text + ')))'
            until('!!(' + expr + ')&&!(' + expr + ').disabled')
            js(expr + '.click()')
            pause(150)

        def mode(value):
            js('(()=>{const e=document.querySelector("select[aria-label=采集模式]");e.value=' + json.dumps(value) + ';e.dispatchEvent(new Event("change",{bubbles:true}));})()')
            pause(200)

        def fps(value):
            js('(()=>{const label=[...document.querySelectorAll(".lip-fields label")].find(e=>e.textContent.trim().startsWith("请求帧率"));const e=label.querySelector("input");e.value=' + json.dumps(str(value)) + ';e.dispatchEvent(new Event("input",{bubbles:true}));})()')
            pause(100)

        def capture(name):
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
            report['captures'].append(dict(name=name, file=str(path), image=[2560, 1600], raw_image=size, dpr=dpr, pageZoom=1, maximized=True, theme='light', navigationCollapsed=True, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            print('Captured ' + name, flush=True)
            return image

        try:
            until('!!document.querySelector(".lip-page")&&document.fonts.status==="loaded"')
            click('设置')
            click('浅色')
            click('唇形提取')
            click('收起侧栏')
            until('document.querySelector(".app-shell").classList.contains("collapsed")')
            assert js('[...document.querySelector("select[aria-label=采集模式]").options].map(e=>e.textContent)') == ['预览', '实时计算', '视频录制', '离线分析']
            capture('m05-preview-ready')
            mode('realtime')
            capture('m05-realtime-ready')
            mode('record_then_analyze')
            fps(120)
            capture('m05-video-recording-ready')
            mode('preview')
            fps(30)
            js('[...document.querySelectorAll(".lip-fields label")].find(e=>e.textContent.includes("特征点防抖")).querySelector("input").click()')
            full = capture('m05-filter-off-ready')
            boxes = js('(()=>{const labels=[...document.querySelectorAll(".lip-fields label")];return ["特征点防抖","截止频率"].map(t=>{const r=labels.find(e=>e.textContent.trim().startsWith(t)).getBoundingClientRect();return [r.x,r.y,r.right,r.bottom];});})()')
            dpr = full.devicePixelRatio()
            x, y = int(min(b[0] for b in boxes) * dpr) - 18, int(min(b[1] for b in boxes) * dpr) - 18
            right, bottom = int(max(b[2] for b in boxes) * dpr) + 18, int(max(b[3] for b in boxes) * dpr) + 18
            detail = full.copy(x, y, right - x, bottom - y)
            # Pixel coordinates on the crop, without editing UI or result text.
            detail.setDevicePixelRatio(1)
            painter = QPainter(detail)
            painter.setPen(QPen(QColor('#dc2626'), 3))
            annotations = []
            for left, top, r, b in boxes:
                box = [int(left * dpr) - x - 3, int(top * dpr) - y - 3, int(r * dpr) - x + 3, int(b * dpr) - y + 3]
                painter.drawRoundedRect(box[0], box[1], box[2] - box[0], box[3] - box[1], 7, 7)
                annotations.append(dict(type='roundedRectangle', box=box))
            painter.end()
            detail_path = out / 'm05-filter-detail.png'
            assert detail.save(str(detail_path))
            report['captures'].append(dict(name='m05-filter-detail', file=str(detail_path), image=[detail.width(), detail.height()], dpr=dpr, pageZoom=1, maximized=True, theme='light', navigationCollapsed=True, sha256=hashlib.sha256(detail_path.read_bytes()).hexdigest(), original='m05-filter-off-ready', crop=[x, y, right, bottom], annotations=annotations))
            js('[...document.querySelectorAll(".lip-fields label")].find(e=>e.textContent.includes("特征点防抖")).querySelector("input").click()')
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
            until('Number(document.querySelector("[aria-label=回放帧]").value)>=2')
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
            until('Number(document.querySelector(".lip-state").textContent.match(/呈现 (\\d+)/)?.[1])>8')
            pause(1200)
            click('结束录制')
            until('document.querySelector(".lip-state").dataset.phase==="ready"&&document.querySelector(".lip-page").getAttribute("aria-busy")==="false"')
            before = js('document.querySelector(".lip-state").textContent')
            mode('offline')
            assert js('document.querySelector(".lip-state").textContent') == before
            js('document.querySelector(".lip-stat-help").open=true')
            capture('m05-capture-statistics')
            report['checks'].append('Media-only synthetic recording statistics retained in offline mode; five definitions expanded, no physical capture')
            assert not report['terminations']
            report['success'] = True
        except Exception as error:
            report['errors'].append(repr(error))
            window.view.grab().save(str(out / 'failed.png'))
            raise
        finally:
            report['source'] = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in ['frontend/src/modules/lip-extraction/LipExtractionPage.vue', 'frontend/dist/index.html', 'scripts/manual/capture_m05_current.py']}
            report['page'] = js('document.body.innerText')
            (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', 'utf8')
            window.accept_close()
            pause(200)
            window.page.deleteLater()
            app.processEvents()
            window.profile.deleteLater()
            app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            app.processEvents()
        print(json.dumps(dict(success=True, out=str(out), captures=len(report['captures']))), flush=True)


if __name__ == '__main__':
    main()
