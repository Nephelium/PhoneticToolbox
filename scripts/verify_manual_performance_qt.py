"""Measure real manual navigation in an owned, visible default-GPU Qt workbench."""
import argparse
import faulthandler
import json
import os
from pathlib import Path
import time

from workbench_source import configure
from verification_artifacts import verification_run, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--frontend', type=Path, help='Independent built frontend; defaults to the current source build')
    parser.add_argument('--glass-matrix', action='store_true', help='Generated wallpaper, native wheel, font/size/theme/window matrix')
    parser.add_argument('--label', default='qt', help='Owned evidence run prefix')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    configure(root)
    os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS', '--mute-audio')
    from PyQt6.QtCore import QEventLoop, QTimer, QT_VERSION_STR, Qt, QPoint, QPointF
    from PyQt6.QtGui import QWheelEvent
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication
    from ptb_desktop.host import Workbench, register_scheme
    from ptb_desktop.web_assets import Assets

    register_scheme()
    app = QApplication(['Manual-performance-owned-QA'])
    frontend=(args.frontend or root/'frontend/dist').resolve()
    assert (frontend/'index.html').is_file(), frontend
    with verification_run('manual-performance', args.label, 'Actual built manual and M17, visible default GPU, isolated in-memory browser profile; no user files or devices') as (out, scratch):
        report = dict(success=False, qt=QT_VERSION_STR, chromium_flags=os.environ.get('QTWEBENGINE_CHROMIUM_FLAGS'), cases=[], focus=[], resources=[], render_terminations=[])
        report['frontend']=str(frontend)
        faulthandler.dump_traceback_later(60,repeat=True)
        print(json.dumps(dict(phase='start',out=str(out),frontend=str(frontend))),flush=True)
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
            report['last_js']=code
            loop, values = QEventLoop(), []
            timeout=QTimer();timeout.setSingleShot(True);timeout.timeout.connect(loop.quit)
            window.page.runJavaScript(code, lambda value: (values.append(value), loop.quit()))
            if not values:
                timeout.start(7000);loop.exec()
            timeout.stop()
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
            faulthandler.dump_traceback_later(60,repeat=False)
            write_json(out/'report.json',report)
            print(json.dumps(dict(phase='case',**result)),flush=True)
            return result

        try:
            window = Workbench(frontend, test=True, vocal_profile=scratch/'vocal', start_module='M17')
            presentation=dict(attach_calls=0,scan_calls=0,attach_ms=[],scans=[])
            report['presentation']=presentation
            original_attach=window._first_maximize.attach
            original_find=window.view.findChildren
            def attach_measured():
                began=time.perf_counter();presentation['attach_calls']+=1
                try:return original_attach()
                finally:presentation['attach_ms'].append(round((time.perf_counter()-began)*1000,3))
            def find_measured(*values,**options):
                began=time.perf_counter();presentation['scan_calls']+=1
                try:return original_find(*values,**options)
                finally:presentation['scans'].append(dict(ms=round((time.perf_counter()-began)*1000,3),options=str(options)))
            window._first_maximize.attach=attach_measured
            window.view.findChildren=find_measured
            window.resize(1440, 900)
            window.show()
            print(json.dumps(dict(phase='window-visible',handle=int(window.winId()))),flush=True)
            window.page.renderProcessTerminated.connect(lambda status, code: report['render_terminations'].append([str(status), code]))
            until('document.querySelector(".m17-body")?.dataset.loaded==="true"&&document.querySelector(".m17-body")?.dataset.fontReady==="true"')
            report['renderer']=js('(()=>{const c=document.createElement("canvas"),g=c.getContext("webgl"),e=g?.getExtension("WEBGL_debug_renderer_info");return {userAgent:navigator.userAgent,devicePixelRatio,renderer:e?g.getParameter(e.UNMASKED_RENDERER_WEBGL):null}})()')
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
            if args.glass_matrix:
                print(json.dumps(dict(phase='glass-matrix')),flush=True)
                label('设置')
                js('(()=>{const c=document.createElement("canvas");c.width=1800;c.height=1000;const x=c.getContext("2d");for(let y=0;y<1000;y+=40)for(let v=0;v<1800;v+=40){x.fillStyle=(v+y)%80?"#4477aa":"#eebbaa";x.fillRect(v,y,40,40)}c.toBlob(blob=>{const d=new DataTransfer();d.items.add(new File([blob],"reader-performance-checker.png",{type:"image/png"}));const e=document.querySelector("input[aria-label=上传背景图]");e.files=d.files;e.dispatchEvent(new Event("change",{bubbles:true}));});})()')
                until('getComputedStyle(document.documentElement).getPropertyValue("--button-glass-image").includes("blob:")')
                pause(500)
                report['matrix']=[]
                for maximized in (False,True):
                    if maximized:window.showMaximized()
                    else:window.showNormal();window.resize(1440,900)
                    pause(150)
                    for theme in ('light','dark'):
                        for font in ('SimSun','Microsoft YaHei','KaiTi'):
                            for size in (10,12,14,15,18,20,24):
                                js('document.documentElement.dataset.theme='+json.dumps(theme)+';document.documentElement.style.setProperty("--font",'+json.dumps('"Times New Roman","'+font+'"')+');document.documentElement.style.setProperty("--body-size",'+json.dumps(str(size)+'px')+')')
                                pause(50)
                                for tab,selector in (('M17','.m17-chart-viewport'),('help','.manual-reading-scroll')):
                                    click('#tab-'+tab)
                                    if tab=='M17':label('extIPA')
                                    else:
                                        click('[data-manual-chapter=m01]')
                                        until('document.querySelector(".manual-document")?.dataset.manualChapter==="m01"')
                                    pause(80)
                                    js('document.querySelector('+json.dumps(selector)+').scrollTop=0')
                                    pause(25);reset()
                                    position=js('(()=>{const s=document.querySelector('+json.dumps(selector)+'),r=s.getBoundingClientRect();return {x:r.x+r.width*.65,y:r.y+r.height*.5,before:s.scrollTop,max:s.scrollHeight-s.clientHeight}})()')
                                    point=QPoint(round(position['x']),round(position['y']))
                                    target=window.view.focusProxy() or window.view
                                    global_point=window.view.mapToGlobal(point)
                                    target_point=target.mapFromGlobal(global_point)
                                    QTest.mouseMove(target,target_point);pause(50)
                                    for delta in (-360,360):
                                        # Exercise three physical-wheel detents with update/end
                                        # phases instead of using one large synthetic event.
                                        for detent in range(3):
                                            event=QWheelEvent(QPointF(target_point),QPointF(global_point),QPoint(0,0),QPoint(0,delta//3),Qt.MouseButton.NoButton,Qt.KeyboardModifier.NoModifier,Qt.ScrollPhase.ScrollBegin if detent==0 else Qt.ScrollPhase.ScrollUpdate,False)
                                            QApplication.sendEvent(target,event);pause(70)
                                        end=QWheelEvent(QPointF(target_point),QPointF(global_point),QPoint(0,0),QPoint(0,0),Qt.MouseButton.NoButton,Qt.KeyboardModifier.NoModifier,Qt.ScrollPhase.ScrollEnd,False)
                                        QApplication.sendEvent(target,end);pause(150)
                                        top=js('document.querySelector('+json.dumps(selector)+').scrollTop')
                                        assert (top>0 if delta<0 and position['max']>0 else top==0),(tab,theme,font,size,delta,position,top)
                                    result=collect('native wheel '+tab+' '+theme+' '+font+' '+str(size))
                                    report['matrix'].append(dict(maximized=maximized,theme=theme,font=font,size=size,tab=tab,long_tasks_ms=result['long_tasks_ms'],renderer_interval_max_ms=result['renderer_interval_max_ms']))
                                if font=='KaiTi' and size==15:
                                    window.view.grab().save(str(out/('glass-'+theme+('-max' if maximized else '-normal')+'.png')))
                # Traverse the complete current book after changing the shared renderer.
                report['chapters']=[]
                for chapter in js('[...document.querySelectorAll("button[data-manual-chapter]")].map(e=>e.dataset.manualChapter)'):
                    click('[data-manual-chapter='+chapter+']')
                    until('document.querySelector(".manual-document")?.dataset.manualChapter==='+json.dumps(chapter))
                    pause(60)
                    assert js('!!document.querySelector(".manual-document h1")'),chapter
                    report['chapters'].append(chapter)
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
            faulthandler.cancel_dump_traceback_later()
            write_json(out/'report.json', report)
            print(json.dumps(dict(out=str(out), success=report['success'], cases=len(report['cases']), matrix=len(report.get('matrix',[])), error=report.get('error')), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
