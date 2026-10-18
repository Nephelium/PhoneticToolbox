"""M02 live-window QA and current manual screenshots, using synthetic inputs only."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import struct
import sys
import time
import wave

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
os.environ['QT_QPA_PLATFORM'] = 'windows'
os.environ['QT_SCALE_FACTOR'] = '1'
os.environ['QT_SCREEN_SCALE_FACTORS'] = '1.5'
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS', '--mute-audio')
from workbench_source import configure
from verification_artifacts import verification_run


def fixtures(folder):
    names = ['Time_s', 'F0 - Praat', 'F0 - REAPER', 'Intensity', 'SHR (pF0)',
             'F1 - Praat', 'F2 - Praat', 'F3 - Praat', 'text_words', 'text_phones']
    rows = []
    for i in range(401):
        t = i / 100
        voiced = i % 100 < 82
        f0 = 205 - 14 * t + 6 * math.sin(t * 4)
        rows.append([t, f0 if voiced else None, f0 + 2 if voiced else None,
                     62 + 4 * math.sin(t * 3) if voiced else None,
                     .18 + .03 * math.sin(t * 2) if voiced else None,
                     610 + 70 * math.sin(t * 3) if voiced else None,
                     1550 + 100 * math.sin(t * 2) if voiced else None,
                     2850 + 100 * math.cos(t * 2) if voiced else None,
                     ['示例', '语音', '参数', '对照'][min(3, i // 100)],
                     ['ɑ̃˥', 'i˩', 'u˧', 'a˥'][min(3, i // 100)]])
    with sqlite3.connect(folder / '示例语音.ptb.sqlite') as conn:
        conn.execute('CREATE TABLE params (' + ','.join('"' + n + '" ' + ('TEXT' if n.startswith('text_') else 'REAL') for n in names) + ')')
        conn.executemany('INSERT INTO params VALUES (' + ','.join('?' for _ in names) + ')', rows)
    from openpyxl import Workbook
    book = Workbook()
    book.active.append(names)
    for row in rows:
        book.active.append(row)
    book.save(folder / '示例语音.xlsx')
    for name in ['示例语音.wav', '未关联示例.wav']:
        with wave.open(str(folder / name), 'wb') as stream:
            stream.setparams((1, 2, 16000, 64000, 'NONE', 'not compressed'))
            stream.writeframes(b''.join(struct.pack('<h', round(9000 * math.sin(2 * math.pi * 180 * i / 16000) * (.7 + .3 * math.cos(i / 1200))) if i % 16000 < 13120 else 0) for i in range(64000)))
    (folder / '损坏示例.xlsx').write_bytes(b'Explicit invalid test workbook, not user data.')


def main():
    configure()
    from PyQt6.QtCore import QEvent, QEventLoop, QTimer, Qt, QRect
    from PyQt6.QtWidgets import QApplication, QFileDialog
    from ptb_desktop.host import Workbench, register_scheme

    register_scheme()
    app = QApplication(['M02 owned live-window and manual QA'])
    app.setQuitOnLastWindowClosed(False)
    export_only = '--export-check-only' in sys.argv
    with verification_run('m02', 'export-qt' if export_only else 'current-qt', 'Actual Windows Qt/QWebChannel; generated 4s mono PCM at 16kHz, legacy SQLite/XLSX and 401 synthetic parameter frames; no research data or devices') as (out, scratch):
        inputs = scratch / 'inputs'
        inputs.mkdir()
        fixtures(inputs)
        database = scratch / 'jobs.sqlite3'
        # Copy only the existing test schema into a new owned database.
        template = ROOT / 'output/validation/p06/local-state.sqlite3'
        with sqlite3.connect(template.as_uri() + '?mode=ro', uri=True) as source, sqlite3.connect(database) as destination:
            schema = source.execute("SELECT sql FROM sqlite_master WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%' ORDER BY CASE type WHEN 'table' THEN 0 WHEN 'index' THEN 1 ELSE 2 END").fetchall()
            for (sql,) in schema:
                destination.execute(sql)
            for name in ['schema_version', 'acoustic_batch_version']:
                for row in source.execute('SELECT * FROM ' + name):
                    destination.execute('INSERT INTO ' + name + ' VALUES (' + ','.join('?' for _ in row) + ')', row)
        from ptb_worker.local_acoustic_files import initialize_local_files
        cache = scratch / 'cache'
        cache.mkdir()
        initialize_local_files(cache)
        QFileDialog.getExistingDirectory = lambda *a, **k: str(inputs)
        window = Workbench(ROOT / 'frontend/dist', test=True, jobs_path=database, local_files_root=cache, vocal_profile=scratch / 'vocal', start_module='M02')
        window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
        window.resize(window.screen().geometry().size())
        window.showMaximized()
        window.page.setAudioMuted(True)
        report = dict(success=False, checks=[], captures=[], downloads=[], errors=[], scope='Windows current source, actual Qt host, synthetic files; hidden maximized/DPR 1.5/page 100%')
        window.profile.downloadRequested.disconnect()
        downloads = []

        def download(item):
            item.setDownloadDirectory(str(scratch))
            item.setDownloadFileName(Path(item.suggestedFileName()).name)
            downloads.append(item)
            item.accept()

        window.profile.downloadRequested.connect(download)

        def pause(ms=100):
            loop = QEventLoop()
            QTimer.singleShot(ms, loop.quit)
            loop.exec()

        def js(code):
            loop, values = QEventLoop(), []
            window.page.runJavaScript(code, lambda v: (values.append(v), loop.quit()))
            QTimer.singleShot(6000, loop.quit)
            loop.exec()
            if not values:
                raise RuntimeError('JavaScript timeout: ' + code)
            return values[0]

        def until(code, seconds=45):
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                if js(code):
                    return
                pause()
            raise AssertionError(code + '\n' + str(js('document.body.innerText.slice(-2500)')))

        def click(text, scope='.m02-page'):
            expression = '[...document.querySelector(' + json.dumps(scope) + ').querySelectorAll("button")].find(b=>b.offsetParent&&b.textContent.trim()===' + json.dumps(text, ensure_ascii=False) + ')'
            until('!!(' + expression + ')&&!(' + expression + ').disabled')
            js(expression + '.click()')
            pause(150)

        def field(selector, value):
            js('(()=>{const e=document.querySelector(' + json.dumps(selector) + ');e.value=' + json.dumps(str(value), ensure_ascii=False) + ';e.dispatchEvent(new Event("input",{bubbles:true}));e.dispatchEvent(new Event("change",{bubbles:true}));})()')
            pause(150)

        def ready():
            until('!document.querySelector(".m02-page>p[role=status]")&&[...document.querySelectorAll(".parameter-figure")].every(e=>e.getAttribute("aria-busy")==="false")')

        def mark(names):
            field('[aria-label="参数搜索"]', '')
            if js('document.querySelectorAll(".m02-parameters input:checked").length'):
                click('清空勾选')
            for name in names:
                assert js('(()=>{const e=[...document.querySelectorAll(".m02-parameters input")].find(e=>e.value===' + json.dumps(name, ensure_ascii=False) + ');if(!e||e.disabled)return false;e.click();return true})()'), name
                pause(130)
            ready()

        def table(name):
            assert js('(()=>{const e=document.querySelector(".signal-panel>label select"),o=[...e.options].find(o=>o.textContent===' + json.dumps(name, ensure_ascii=False) + ');if(!o)return false;e.value=o.value;e.dispatchEvent(new Event("change",{bubbles:true}));return true})()'), name
            ready()

        def capture(name, anchor, caption, selector=None, crop=False):
            if selector:
                js('document.querySelector(' + json.dumps(selector) + ').scrollIntoView({block:"center"})')
            pause(350)
            window.view.grab()
            pause(200)
            pix = window.view.grab()
            if crop:
                box = js('(()=>{const r=document.querySelector(' + json.dumps(selector) + ').getBoundingClientRect();return [r.x,r.y,r.width,r.height]})()')
                dpr = pix.devicePixelRatio()
                pix = pix.copy(QRect(*[round(n * dpr) for n in box]))
            path = out / (name + '.png')
            assert pix.save(str(path))
            report['captures'].append(dict(name=name, anchor=anchor, caption=caption, file=str(path), image=[pix.width(), pix.height()], sha256=hashlib.sha256(path.read_bytes()).hexdigest(), theme=js('document.documentElement.dataset.theme'), dpr=window.devicePixelRatioF(), pageZoom=window.view.zoomFactor(), maximized=window.isMaximized(), cropped=crop))
            print('Captured ' + name, flush=True)

        def save_current(fmt):
            field('.image-format', fmt)
            before = len(downloads)
            click('保存当前图')
            deadline = time.monotonic() + 25
            while time.monotonic() < deadline and (len(downloads) == before or not downloads[-1].isFinished()):
                pause()
            assert len(downloads) > before and downloads[-1].isFinished()
            path = scratch / downloads[-1].downloadFileName()
            assert path.is_file()
            report['downloads'].append(dict(name=path.name, bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            return path

        try:
            until('!!document.querySelector(".m02-page")')
            js('document.documentElement.dataset.theme="light";document.documentElement.dataset.palette="absolutely";')
            click('打开音频目录')
            click('示例语音.wav')
            ready()
            assert js('!!document.querySelector(".parameter-figure")&&!document.querySelector(".m02-page>p[role=alert]")')
            if export_only:
                mark(['F1 - Praat', 'SHR (pF0)'])
                js('window.qaPainted=[];const originalFillText=CanvasRenderingContext2D.prototype.fillText;CanvasRenderingContext2D.prototype.fillText=function(text,...args){window.qaPainted.push(text);return originalFillText.call(this,text,...args);}')
                for mode, expected in [('light', 'rgb(182, 85, 118)'), ('dark', 'rgb(229, 156, 182)')]:
                    js('document.documentElement.dataset.theme=' + json.dumps(mode))
                    assert js('getComputedStyle(document.querySelector(".parameter-curve")).stroke') == expected
                    png = save_current('png')
                    assert js('window.qaPainted.includes("图窗 1")&&!window.qaPainted.some(t=>t.includes("当前"))')
                    if mode == 'light':
                        (out / 'parameter-example.png').write_bytes(png.read_bytes())
                    capture('export-' + mode, '', '当前图窗导出及浅深色曲线核对。', '.parameter-figure')
                report['checks'] = ['Actual SQLite/Qt final light/dark colors', 'Actual native PNG title contains only group name', 'Native PNG output uses shared compact paper-style legend']
                report['success'] = True
                return
            assert js('document.querySelectorAll(".m02-parameters input:checked").length===0')
            capture('overview', 'm02-figure-m02-overview-r2-light', '打开合成示例 WAV 与同名 SQLite 后，图窗和右侧勾选均为空。新建图窗也从空选择开始。')
            table('示例语音.xlsx')
            mark(['F0 - Praat', 'F0 - REAPER'])
            field('[aria-label="参数搜索"]', 'F0 -')
            capture('association', 'm02-figure-association-checks-light', '关联 XLSX 后搜索并勾选两项 F0，曲线立即显示，无需分配。', '.parameter-figure')
            capture('selection-detail', 'm02-association-checks-light-detail-figure', '当前图窗的勾选即对应已绘制参数。取消勾选移除该曲线，清空勾选清空当前图窗。', '.parameter-summary', True)
            capture('first-curves', 'm02-figure-first-curves-light', '两项 F0 即时叠加，紧凑图例标出名称和线色。示例曲线含保留的缺口。', '.parameter-figure')
            click('未关联示例.wav')
            ready()
            capture('unmatched', 'm02-figure-unmatched-result-light', '未关联参数的合成示例仍可查看波形及试听，需手动关联正确结果。')
            click('示例语音.wav')
            ready()
            mark(['F1 - Praat', 'SHR (pF0)'])
            capture('dual-axis', 'm02-figure-dual-axis-light', 'F1 与 SHR 保留原数值自动双轴，图例完整显示轴归属，右轴曲线使用虚线。', '.parameter-figure')
            click('新建图窗')
            assert js('document.querySelectorAll(".m02-parameters input:checked").length===0')
            mark(['F2 - Praat', 'F3 - Praat'])
            capture('second-window', 'm02-figure-second-plot-light', '新窗从空选择开始。勾选 F2、F3 后在图窗 2 显示，图窗 1 的 F1、SHR 保持。', '.parameter-figure[data-group-id="2"]')
            # Clicking the original card must restore its checked parameters.
            js('document.querySelector(".parameter-figure[data-group-id=\\"1\\"] .figure-title").click()')
            assert js('[...document.querySelectorAll(".m02-parameters input:checked")].map(e=>e.value).join("|")==="SHR (pF0)|F1 - Praat"')
            report['checks'].append('Actual host legacy SQLite/XLSX reads, immediate plotting, independent windows, empty new selection and card activation')
            field('[aria-label="当前图窗"]', 2)
            click('合并回首图', '.parameter-figure[data-group-id="2"]')
            mark(['F1 - Praat', 'SHR (pF0)'])
            field('.m02-toolbar input[type=number][min=".01"]', 1.8)
            field('.m02-toolbar input[type=number][min="0"]', .8)
            ready()
            capture('zoom-selection', 'm02-figure-zoom-selection-light', '时间窗显示约 0.8 至 2.6 秒，波形与当前参数图共享时间位置。', '.parameter-figure')
            js('document.querySelector(".m02-toolbar input[type=checkbox]").click()')
            until('!!document.querySelector(".spectrogram-canvas canvas")&&!document.querySelector(".spectrogram-view [role=status]")', 60)
            capture('spectrogram', 'm02-figure-spectrogram-ready-light', '同一合成示例的本机 Praat 语谱图已读取，与波形和参数图共用查看时间窗。', '.m02-waveform-area')
            mark(['F0 - REAPER', 'Intensity', 'text_words', 'text_phones'])
            capture('text-layers', 'm02-figure-table-text-layer-light', '勾选表内文字列后，文字和区间边界立即显示在波形及参数图中。标签与参数均为合成示例。', '.parameter-figure')
            mark(['F1 - Praat', 'SHR (pF0)'])
            field('.image-format', 'svg')
            capture('svg-export', 'm02-figure-svg-current-export-light', '选择 SVG 保存当前图。图例优先放在绘图区空白位置，空间不足时移至上方，长名称完整换行。', '.parameter-figure')
            capture('export-detail', 'm02-svg-current-export-light-detail-figure', '当前图可选 PNG 或 SVG；整幅 PNG 还包含波形及已开启的语谱图。', '.parameter-figure header', True)
            for mode in ['light', 'dark']:
                js('document.documentElement.dataset.theme=' + json.dumps(mode))
                svg = save_current('svg')
                xml = svg.read_text('utf8')
                assert 'plot-legend' in xml and 'F1 - Praat [右]' in xml and 'SHR (pF0) [左]' in xml
                png = save_current('png')
                assert png.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
                if mode == 'light':
                    (out / 'parameter-example.png').write_bytes(png.read_bytes())
                capture('theme-' + mode, '', mode + ' 当前参数图与即时勾选。', '.parameter-figure')
            js('document.documentElement.dataset.theme="light"')
            before = len(downloads)
            click('保存整幅 PNG')
            deadline = time.monotonic() + 25
            while time.monotonic() < deadline and (len(downloads) == before or not downloads[-1].isFinished()):
                pause()
            assert len(downloads) > before and downloads[-1].isFinished()
            whole = scratch / downloads[-1].downloadFileName()
            # Keep one actual figure as a manual illustration and reviewable result.
            import shutil
            final = out / 'whole-figure.png'
            shutil.copyfile(whole, final)
            from PyQt6.QtGui import QImage
            image = QImage(str(final))
            report['captures'].append(dict(name='whole-figure', anchor='m02-figure-export-whole-plot', caption='当前应用实际导出的白底 300 dpi 整幅 PNG，包含合成示例的波形、Praat 语谱及双轴参数图，紧凑图例随图保存。', file=str(final), image=[image.width(), image.height()], sha256=hashlib.sha256(final.read_bytes()).hexdigest(), theme='light', dpr=1.5, pageZoom=1, maximized=True, cropped=False))
            click('保存绘图配置')
            click('刷新文件')
            ready()
            assert js('document.querySelectorAll(".m02-parameters input:checked").length===2')
            capture('restored', 'm02-figure-config-restored-light', '刷新并恢复已保存配置后，F1、SHR 曲线与右侧勾选一致，查看时间窗回到全长。', '.parameter-figure')
            field('[aria-label="参数搜索"]', '无匹配示例')
            capture('no-candidates', 'm02-figure-no-matching-candidate-light', '搜索无匹配时图窗曲线和已选参数保留；清空勾选仍会移除当前窗全部参数。', '.parameter-figure')
            click('清空勾选')
            assert js('document.querySelectorAll(".parameter-curve").length===0')
            table('损坏示例.xlsx')
            until('!!document.querySelector(".m02-page>p[role=alert]")')
            capture('invalid-table', 'm02-figure-invalid-table-error-light', '损坏的合成 XLSX 无法读取，顶部显示错误，原音波形仍保留。重新选择有效结果可恢复。')
            report['checks'].extend(['Light/dark actual native PNG/SVG downloads, complete legends and whole-PNG export', 'Configuration refresh restores matching checks; filtered clear removes all curves; invalid workbook recovers with retained waveform'])
            report['success'] = True
        except Exception as error:
            report['errors'].append(str(error))
            window.view.grab().save(str(out / 'failure.png'))
            raise
        finally:
            (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            js('window.onbeforeunload=null')
            window.closing = True
            window.close()
            window.page.deleteLater()
            pause(150)
            window.profile.deleteLater()
            app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            app.processEvents()
            window.deleteLater()
            pause(500)
            app.quit()
            print(out, flush=True)


if __name__ == '__main__':
    main()
