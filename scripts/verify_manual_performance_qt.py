"""Measure real manual navigation in an owned, visible default-GPU Qt workbench."""
import argparse
import json
import os
from pathlib import Path
import time

from workbench_source import configure
from verification_artifacts import verification_run, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    configure(root)
    os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS', '--mute-audio')
    from PyQt6.QtCore import QEventLoop, QTimer, QT_VERSION_STR, Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication
    from ptb_desktop.host import Workbench, register_scheme
    from ptb_desktop.web_assets import Assets

    register_scheme()
    app = QApplication(['Manual-performance-owned-QA'])
    with verification_run('manual-performance', 'qt', 'Actual built manual and M17, visible default GPU, isolated in-memory browser profile; no user files or devices') as (out, scratch):
        report = dict(success=False, qt=QT_VERSION_STR, chromium_flags=os.environ.get('QTWEBENGINE_CHROMIUM_FLAGS'), cases=[], focus=[], resources=[], render_terminations=[])
        if args.baseline:
            baseline = json.loads(args.baseline.read_text('utf-8'))
            report['baseline'] = dict(scope=baseline['scope'], cases=baseline['cases'])
        original = Assets.requestStarted

        def measured(handler, job):
            start = time.perf_counter()
            try:
                return original(handler, job)
            finally:
                report['resources'].append(dict(path=job.requestUrl().path(), ms=round((time.perf_counter()-start)*1000, 3)))

        Assets.requestStarted = measured
        window = None

        def pause(ms=100):
            loop = QEventLoop()
            QTimer.singleShot(ms, loop.quit)
            loop.exec()

        def js(code):
            loop, values = QEventLoop(), []
            window.page.runJavaScript(code, lambda value: (values.append(value), loop.quit()))
            QTimer.singleShot(7000, loop.quit)
            loop.exec()
            assert values, 'JavaScript callback timeout: ' + code[:100]
            return values[0]

        def until(code):
            end = time.monotonic()+25
            while not js(code):
                assert time.monotonic() < end, code
                pause(50)

        def click(selector):
            assert js('(()=>{const node=document.querySelector('+json.dumps(selector)+');if(!node)return false;node.click();return true})()'), selector

        def label(text):
            assert js('(()=>{const node=[...document.querySelectorAll("button")].find(e=>e.offsetParent&&e.textContent.trim()==='+json.dumps(text)+');if(!node)return false;node.click();return true})()'), text

        def reset():
            js('window.__long=[];window.__writes=0;window.__ticks=[];window.__last=performance.now()')
            report['resources'].clear()

        def collect(name, start=None):
            pause(100)
            result = js('({long_tasks_ms:window.__long,storage_writes:window.__writes,renderer_interval_max_ms:Math.max(0,...window.__ticks)})')
            result.update(name=name, elapsed_ms=round((time.perf_counter()-start)*1000, 2) if start else None,
                          resource_count=len(report['resources']), resource_total_ms=round(sum(r['ms'] for r in report['resources']), 2))
            report['cases'].append(result)
            return result

        try:
            window = Workbench(root/'frontend/dist', test=True, vocal_profile=scratch/'vocal', start_module='M17')
            window.resize(1440, 900)
            window.show()
            window.page.renderProcessTerminated.connect(lambda status, code: report['render_terminations'].append([str(status), code]))
            until('document.querySelector(".m17-body")?.dataset.loaded==="true"&&document.querySelector(".m17-body")?.dataset.fontReady==="true"')
            js('window.__long=[];window.__writes=0;window.__ticks=[];window.__last=performance.now();window.__originalWrite=Storage.prototype.setItem;Storage.prototype.setItem=function(key,...args){if(key==="ptb.v3.manual-location")window.__writes++;return window.__originalWrite.call(this,key,...args)};window.__performanceObserver=new PerformanceObserver(list=>window.__long.push(...list.getEntries().map(e=>e.duration)));window.__performanceObserver.observe({entryTypes:["longtask"]});window.__tickTimer=setInterval(()=>{let now=performance.now();window.__ticks.push(now-window.__last);window.__last=now},20)')
            for theme in ['light', 'dark']:
                js('document.documentElement.dataset.theme='+json.dumps(theme))
                for name, system in [('extIPA', 'extipa'), ('VoQS', 'voqs'), ('IPA', 'ipa')]:
                    reset(); start = time.perf_counter(); label(name)
                    until('!!document.querySelector("[data-chart='+system+']")')
                    collect('M17 '+theme+' '+system, start)
            label('使用说明')
            until('!!document.querySelector(".manual-document")')
            for theme, chapter in [('light', 'm01'), ('dark', 'm03')]:
                js('document.documentElement.dataset.theme='+json.dumps(theme))
                click('.manual-toc [data-manual-chapter='+chapter+']')
                until('document.querySelector(".manual-document")?.dataset.manualChapter==='+json.dumps(chapter))
                pause(400)
                js('window.__document=document.querySelector(".manual-document");window.__media=[...window.__document.querySelectorAll("img,audio,video")]')
                reset(); start = time.perf_counter()
                click('.manual-toc [data-manual-section]')
                pause(60)
                result = collect('same chapter '+chapter, start)
                result['same_document'] = js('window.__document===document.querySelector(".manual-document")&&window.__media.every(e=>e.isConnected)')
                assert result['same_document'], result
                assert result['resource_count'] == 0, result
                js('document.querySelector(".manual-toc [data-manual-section]").focus()')
                QTest.keyClick(window.view.focusProxy() or window.view, Qt.Key.Key_Return)
                pause(50)
                focus = js('(()=>{const e=document.activeElement,s=getComputedStyle(e);return {selected:e.matches(".manual-toc .active"),visible:e.matches(":focus-visible"),outline:s.outlineStyle,decoration:s.textDecorationLine}})()')
                assert focus['selected'] and focus['visible'] and focus['outline'] == 'none' and 'underline' in focus['decoration'], focus
                report['focus'].append(dict(theme=theme, **focus))
                reset()
                for _ in range(12):
                    js('document.querySelector(".manual-reading-scroll").scrollTop+=60')
                    pause(25)
                pause(280)
                result = collect('manual scroll '+chapter)
                assert 0 < result['storage_writes'] <= 4, result
                assert js('JSON.parse(localStorage.getItem("ptb.v3.manual-location")).scrollTop===document.querySelector(".manual-reading-scroll").scrollTop')
                window.view.grab().save(str(out/(theme+'.png')))

            # Flush before the timer, restore a hidden tab and preserve on close.
            js('document.querySelector(".manual-reading-scroll").scrollTop+=37')
            pause(40)
            expected = js('document.querySelector(".manual-reading-scroll").scrollTop')
            click('#tab-M17'); pause(60)
            saved = js('JSON.parse(localStorage.getItem("ptb.v3.manual-location"))')
            assert saved['chapterId'] == 'm03' and saved['scrollTop'] == expected, (saved, expected)
            click('#tab-help'); pause(80)
            assert js('document.querySelector(".manual-reading-scroll").scrollTop') == expected
            click('[aria-label="关闭 使用说明"]'); pause(60)
            assert not js('!!document.querySelector(".manual-document")')
            assert js('JSON.parse(localStorage.getItem("ptb.v3.manual-location"))') == saved
            report['preserved_on_close'] = saved
            click('#tab-home')
            label('使用说明')
            # The host deliberately supplies getting-started for global help.
            until('document.querySelector(".manual-document")?.dataset.manualChapter==="getting-started"')
            report['restored_location'] = saved
            reset()
            for _ in range(6):
                click('#tab-M17'); pause(80)
                click('#tab-help'); pause(80)
            collect('M17/help tab switches x12')
            assert not report['render_terminations'], report['render_terminations']
            report['success'] = True
        except Exception as error:
            report['error'] = repr(error)
            raise
        finally:
            if window:
                window.closing = True
                window.close()
                window.page.deleteLater()
                pause(100)
                window.profile.deleteLater()
                pause(200)
                window.deleteLater()
                pause(300)
            Assets.requestStarted = original
            write_json(out/'report.json', report)
            print(json.dumps(dict(out=str(out), success=report['success'], cases=report['cases'], error=report.get('error')), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
