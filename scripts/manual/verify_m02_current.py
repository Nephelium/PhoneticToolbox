"""Read back current M02 manual media and interaction wording in actual Qt."""
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import time
import zlib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
os.environ['QT_QPA_PLATFORM'] = 'windows'
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS', '--mute-audio')
from verification_artifacts import verification_run
from workbench_source import configure


def main():
    configure()
    from PyQt6.QtCore import QEvent, QEventLoop, QTimer, Qt
    from PyQt6.QtWidgets import QApplication
    from ptb_desktop.host import Workbench, register_scheme
    project = json.loads((ROOT / 'manual/project.json').read_text('utf8'))
    chapter = json.loads((ROOT / 'manual/chapters/m02.json').read_text('utf8'))
    expected = [a for a in project['assets'] if 'software-only/m02-current-20261018/' in a['path']]
    assert len(expected) == 16
    for asset in expected:
        hashes = [hashlib.sha256((ROOT / folder / asset['path']).read_bytes()).hexdigest() for folder in ['manual', 'frontend/public/manual', 'frontend/dist/manual']]
        assert hashes == [asset['sha256']] * 3
    for folder in ['frontend/public/manual', 'frontend/dist/manual']:
        assert json.loads((ROOT / folder / 'chapters/m02.json').read_text('utf8')) == chapter
    register_scheme()
    app = QApplication(['M02 current manual reader QA'])
    with verification_run('m02', 'manual-reader', 'Actual Windows Qt; 16 current source/public/dist image hashes, light/dark load/zoom, current wording, independently verified PNG pHYs/CRC; synthetic input only') as (out, scratch):
        report = dict(success=False, media=[a['id'] for a in expected], themes=[])
        window = Workbench(ROOT / 'frontend/dist', test=True, vocal_profile=scratch / 'vocal', start_module='M02')
        window.resize(1500, 1000)
        window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
        window.show()

        def pause(ms=90):
            loop = QEventLoop()
            QTimer.singleShot(ms, loop.quit)
            loop.exec()

        def js(code):
            loop, values = QEventLoop(), []
            window.page.runJavaScript(code, lambda v: (values.append(v), loop.quit()))
            QTimer.singleShot(6000, loop.quit)
            loop.exec()
            assert values, code
            return values[0]

        def until(code):
            deadline = time.monotonic() + 35
            while time.monotonic() < deadline:
                if js(code):
                    return
                pause()
            raise AssertionError(code)

        def click(label):
            expression = '[...document.querySelectorAll("button")].find(e=>e.offsetParent&&e.textContent.trim()===' + json.dumps(label) + ')'
            until('!!(' + expression + ')')
            js(expression + '.click()')
            pause()

        try:
            until('!!document.querySelector(".m02-page")')
            click('使用说明')
            until('!!document.querySelector(".manual-toc [data-manual-chapter=m02]")')
            js('document.querySelector(".manual-toc [data-manual-chapter=m02]").click()')
            until('document.querySelector(".manual-document")?.dataset.manualChapter==="m02"')
            for phrase in ['勾选立即绘制', '新窗全不勾选', '长名称完整换行', '曲线与当前窗勾选一起恢复']:
                assert js('document.querySelector(".manual-document").innerText.includes(' + json.dumps(phrase) + ')'), phrase
            for phrase in ['将 N 项分配到图窗', '从图中移除勾选项', '清空选定图窗', '取消勾选不删曲线', '图例可能省略']:
                assert not js('document.querySelector(".manual-document").innerText.includes(' + json.dumps(phrase) + ')'), phrase
            for mode in ['light', 'dark']:
                js('document.documentElement.dataset.theme=' + json.dumps(mode))
                for asset in expected:
                    expression = 'document.querySelector(' + json.dumps('[data-asset-id="' + asset['id'] + '"]') + ')'
                    js(expression + '.scrollIntoView({block:"center"})')
                    until('(()=>{const i=' + expression + '.querySelector("img");return i?.complete&&i.naturalWidth===' + str(asset['width']) + '&&i.naturalHeight===' + str(asset['height']) + '})()')
                    js(expression + '.querySelector("button").click()')
                    until('document.querySelector(".manual-image-overlay img")?.complete')
                    assert js('document.querySelector(".manual-image-overlay img").naturalWidth') == asset['width']
                    click('关闭图片')
                assert not js('!!document.querySelector(".manual-media-error")')
                assert js('document.documentElement.scrollWidth<=innerWidth+1')
                js('document.getElementById("manual-m02--m02-selection-assignment").scrollIntoView({block:"start"})')
                pause(250)
                window.view.grab().save(str(out / ('reader-' + mode + '.png')))
                report['themes'].append(dict(theme=mode, loaded=16, zoomed=16, mediaErrors=0))
            raw = (ROOT / 'manual' / next(a['path'] for a in expected if 'whole-figure' in a['id'])).read_bytes()
            offset, dpi = 8, None
            while offset < len(raw):
                length = struct.unpack_from('>I', raw, offset)[0]
                kind, data = raw[offset + 4:offset + 8], raw[offset + 8:offset + 8 + length]
                assert zlib.crc32(kind + data) == struct.unpack_from('>I', raw, offset + 8 + length)[0]
                if kind == b'pHYs':
                    x, y, unit = struct.unpack('>IIB', data)
                    assert unit == 1 and abs(x * .0254 - 300) < .01 and x == y
                    dpi = x * .0254
                offset += length + 12
            assert dpi is not None
            report.update(dpi=dpi, success=True)
        except Exception as error:
            report['error'] = repr(error)
            window.view.grab().save(str(out / 'failure.png'))
            raise
        finally:
            (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
            window.closing = True
            window.close()
            window.page.deleteLater()
            pause(120)
            window.profile.deleteLater()
            app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            app.processEvents()
            window.deleteLater()
            pause(150)
            print(json.dumps(dict(out=str(out), success=report['success'])), flush=True)


if __name__ == '__main__':
    main()
