"""Verify the rebuilt M14 chapter in the real Qt reader, with isolated state."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
DIRS = ('desktop/src', 'backend/src', 'packages/phonetic_core/src', 'scripts')
for folder in DIRS:
    sys.path.insert(0, str(ROOT / folder))
os.environ['PYTHONPATH'] = os.pathsep.join(str(ROOT / folder) for folder in DIRS)
os.environ.setdefault('QT_QPA_PLATFORM', 'windows')
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS', '--disable-gpu --disable-gpu-compositing --mute-audio')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    from PyQt6.QtCore import QEventLoop, QTimer, Qt
    from PyQt6.QtGui import QGuiApplication
    from PyQt6.QtWidgets import QApplication
    from ptb_desktop.host import Workbench, register_scheme
    from verify_m14_wiring import setup
    task_out, db, cache = setup()
    register_scheme()
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.Floor)
    app = QApplication(['PTB-owned-M14-manual-reader'])
    window = Workbench(ROOT / 'frontend/dist', test=True, jobs_path=db,
                       local_files_root=cache, vocal_profile=task_out / 'vocal')
    window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
    window.show()
    window.setFixedSize(2560, 1440)
    report = dict(success=False, checks=[], screenshots=[], errors=[], ownedTaskState=str(task_out))

    def pause(ms=100):
        loop = QEventLoop(); QTimer.singleShot(ms, loop.quit); loop.exec()

    def js(code):
        loop, box = QEventLoop(), []
        window.page.runJavaScript(code, lambda v: (box.append(v), loop.quit()))
        QTimer.singleShot(8000, loop.quit); loop.exec()
        assert box, 'JavaScript timeout'
        return box[0]

    def until(code):
        end = time.monotonic() + 45
        while time.monotonic() < end:
            if js(code): return
            pause()
        raise AssertionError(code + '\n' + str(js('document.body.innerText.slice(-2000)')))

    def click(text):
        expr = '[...document.querySelectorAll("button")].find(b=>b.offsetParent&&b.textContent.trim()===' + json.dumps(text) + ')'
        until('!!(' + expr + ')'); js(expr + '.click()'); pause(120)

    def capture(name):
        pause(350); window.grab(); pause(350)
        path = args.output / (name + '.png')
        pix = window.grab()
        assert [pix.width(), pix.height()] == [2560, 1440]
        assert pix.save(str(path))
        report['screenshots'].append(str(path))

    try:
        until('!!document.querySelector("nav")&&document.fonts.status==="loaded"')
        click('设置'); click('浅色'); click('音系归纳'); click('帮助')
        until('document.querySelector(".manual-document")?.dataset.manualChapter==="m14"&&/^\\s*17\\s*音系归纳\\s*$/.test(document.querySelector(".manual-chapter-header h1")?.textContent??"")')
        until('document.querySelectorAll(".manual-document img").length===17')
        assert js('document.querySelector(".manual-document").innerText.includes("不代表真实调类")')
        assert js('document.querySelector(".manual-document").innerText.includes("5,453")')
        assert not js('document.querySelectorAll(".manual-unsupported,.manual-media-error").length')
        images = js('[...document.querySelectorAll(".manual-document img")].map(i=>({src:i.src,alt:i.alt,id:i.closest("figure")?.id}))')
        assert len(images) == 17
        for item in images:
            js('document.getElementById(' + json.dumps(item['id']) + ').scrollIntoView({block:"start"})')
            until('(()=>{const i=document.getElementById(' + json.dumps(item['id']) + ').querySelector("img");return i.complete&&i.naturalWidth===2560&&i.naturalHeight===1440})()')
            assert 'm14-test-' in item['src'] and item['alt']
        report['checks'].append('All 17 new full-window PNGs loaded at natural 2560x1440 with captions and alt text')
        for key in ('m14-purpose', 'm14-tone-example', 'm14-generated-example'):
            js('document.getElementById("manual-m14--' + key + '")?.scrollIntoView({block:"start"})')
            capture('reader-light-' + key)
        matrix = next(item for item in images if 'm14-test-matrix-' in item['src'])
        js('document.getElementById(' + json.dumps(matrix['id']) + ').scrollIntoView({block:"start"})')
        js('document.getElementById(' + json.dumps(matrix['id']) + ').querySelector(".manual-image-button").click()')
        until('(()=>{const i=document.querySelector(".manual-image-overlay img");return i?.complete&&i.naturalWidth===2560&&i.naturalHeight===1440})()')
        assert js('document.querySelector(".manual-image-overlay p").textContent.includes("二维快照")')
        capture('reader-light-matrix-zoom')
        click('关闭图片')
        assert not js('document.querySelector(".manual-image-overlay")')
        report['checks'].append('The matrix opens its original full-window PNG in the image viewer and closes correctly')
        click('返回 音系归纳')
        until('!!document.querySelector(".phonology-page")')
        click('设置'); click('深色'); click('音系归纳'); click('帮助')
        until('document.querySelector(".manual-document")?.dataset.manualChapter==="m14"&&/^\\s*17\\s*音系归纳\\s*$/.test(document.querySelector(".manual-chapter-header h1")?.textContent??"")')
        until('document.documentElement.dataset.theme==="dark"')
        assert js('document.querySelectorAll(".manual-document img").length===17')
        capture('reader-dark-introduction')
        click('返回 音系归纳')
        until('!!document.querySelector(".phonology-page")')
        report['checks'].append('M14 help opens the edited chapter and returns to M14 in light and dark themes')
        report['success'] = True
    except Exception:
        report['errors'].append(traceback.format_exc())
    finally:
        (args.output / 'reader-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        window.close(); pause(300); app.quit()
        print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if report['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
