"""Current static Windows host, isolated profile and generated image only."""
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PARTS = ('scripts', 'desktop/src', 'backend/src', 'packages/phonetic_core/src')
for part in PARTS:
    sys.path.insert(0, str(ROOT / part))
os.environ['PYTHONPATH'] = os.pathsep.join(str(ROOT / part) for part in PARTS)
os.environ['QTWEBENGINE_CHROMIUM_FLAGS'] = '--mute-audio'


def main():
    from verification_artifacts import verification_run
    from PyQt6.QtCore import QEventLoop, QTimer
    from PyQt6.QtGui import QColor
    from PyQt6.QtWidgets import QApplication, QColorDialog
    from ptb_desktop.host import Workbench, register_scheme
    from ptb_worker.local_acoustic_files import initialize_local_files

    with verification_run('custom-appearance', 'qt', 'Owned Windows source Workbench/static dist; copied existing-schema DB; generated red/green PNG; real IndexedDB reload, M10 iframe and native colour dialog. No user image/device capture.') as (out, scratch):
        db = scratch / 'jobs.sqlite3'
        with sqlite3.connect((ROOT / 'output/validation/p06/local-state.sqlite3').as_uri() + '?mode=ro', uri=True) as source, sqlite3.connect(db) as target:
            source.backup(target)
        cache = scratch / 'cache'
        cache.mkdir()
        initialize_local_files(cache)
        register_scheme()
        app = QApplication(['owned-custom-appearance'])
        window = Workbench(ROOT / 'frontend/dist', test=True, jobs_path=db, local_files_root=cache, vocal_profile=scratch / 'vocal')
        window.resize(1440, 900)
        window.show()
        report = dict(success=False, checks=[], errors=[])
        window.page.renderProcessTerminated.connect(lambda status, code: report['errors'].append([status.name, code]))

        def pause(ms=120):
            loop = QEventLoop()
            QTimer.singleShot(ms, loop.quit)
            loop.exec()

        def js(code):
            loop, values = QEventLoop(), []
            window.page.runJavaScript(code, lambda v: (values.append(v), loop.quit()))
            QTimer.singleShot(8000, loop.quit)
            loop.exec()
            assert values, 'JavaScript timeout'
            return values[0]

        def until(code):
            deadline = time.monotonic() + 45
            while time.monotonic() < deadline:
                if js(code):
                    return
                pause()
            raise AssertionError(code)

        def click(label):
            expression = '[...document.querySelectorAll("button")].find(b=>b.offsetParent&&(b.getAttribute("aria-label")||b.textContent.trim())===' + json.dumps(label) + ')'
            until('!!(' + expression + ')&&!(' + expression + ').disabled')
            js(expression + '.click()')

        def set_control(selector, value, event='change'):
            js('(()=>{const e=document.querySelector(' + json.dumps(selector) + ');e.value=' + json.dumps(str(value)) + ';e.dispatchEvent(new Event("input",{bubbles:true}));e.dispatchEvent(new Event(' + json.dumps(event) + ',{bubbles:true}));})()')

        try:
            until("!!document.querySelector('#tool-sidebar')&&document.fonts.status==='loaded'")
            click('设置')
            until("!!document.querySelector('#custom-palette-hex')")
            report['dpr'] = js('devicePixelRatio')
            # Upload via the real input event and File/Blob path in Qt WebEngine.
            js("(()=>{const c=document.createElement('canvas');c.width=400;c.height=240;const x=c.getContext('2d');x.fillStyle='#ee2211';x.fillRect(0,0,200,240);x.fillStyle='#11bb66';x.fillRect(200,0,200,240);c.toBlob(blob=>{const d=new DataTransfer();d.items.add(new File([blob],'背景测试.png',{type:'image/png'}));const e=document.querySelector('input[type=file][aria-label=上传背景图]');e.files=d.files;e.dispatchEvent(new Event('change',{bubbles:true}));});})()")
            until("document.documentElement.dataset.backgroundImage==='on'")
            set_control('#background-blur', 10, 'input')
            set_control('#background-opacity', 40, 'input')
            set_control('#custom-palette-hex', '#789abc')
            until("document.documentElement.dataset.palette==='custom'&&document.documentElement.dataset.customPalette==='#789abc'")
            set_control('#button-material-choice', 'glass')
            set_control('#button-glass-opacity', 20, 'input')
            set_control('#button-glass-blur', 24, 'input')
            for mode in ('light', 'dark'):
                click('浅色' if mode == 'light' else '深色')
                for highlighting in ('auto', 'all', 'plain'):
                    set_control('#button-style-choice', highlighting)
                    pause(250)
                    material = js("(()=>{const b=[...document.querySelectorAll('button')].find(b=>b.textContent==='更换背景图');const c=getComputedStyle(b);return {background:c.backgroundColor,blur:c.backdropFilter,webkit:c.webkitBackdropFilter,variable:getComputedStyle(document.documentElement).getPropertyValue('--button-glass-blur'),support:CSS.supports('backdrop-filter','blur(12px)'),agent:navigator.userAgent};})()")
                    report['last_material']=material
                    assert '0.2' in material['background'] and '24px' in material['blur'], material
                assert js("getComputedStyle(document.querySelector('input[aria-label=搜索工具]')).backgroundColor==='rgba(0, 0, 0, 0)'")
                window.view.grab().save(str(out / ('appearance-' + mode + '.png')))
            report['checks'].append('Real Qt upload; custom palette; transparent text; 2 themes x 3 highlight modes at 20% glass/24px blur')
            # Callback from the actual QWebChannel native picker, with explicit colour/cancel.
            def accept_color():
                dialog = app.activeModalWidget()
                assert isinstance(dialog, QColorDialog)
                dialog.setCurrentColor(QColor('#13579b'))
                dialog.accept()
            QTimer.singleShot(900, accept_color)
            click('吸管取色')
            until("document.documentElement.dataset.customPalette==='#13579b'")
            QTimer.singleShot(900, lambda: app.activeModalWidget().reject())
            click('吸管取色')
            until("[...document.querySelectorAll('button')].some(b=>b.textContent==='吸管取色'&&!b.disabled)")
            assert js("document.documentElement.dataset.customPalette==='#13579b'")
            report['checks'].append('Actual native picker round trip and cancellation preserve prior colour; screen-sampling control unit verified, physical desktop pixel sampling not performed')
            reload_loop=QEventLoop()
            window.page.loadFinished.connect(reload_loop.quit)
            window.page.triggerAction(window.page.WebAction.Reload)
            QTimer.singleShot(15000,reload_loop.quit)
            reload_loop.exec()
            window.page.loadFinished.disconnect(reload_loop.quit)
            until("!!document.querySelector('#tool-sidebar')&&document.documentElement.dataset.backgroundImage==='on'")
            click('设置')
            until("!!document.querySelector('#button-glass-opacity')")
            restored=js("({material:document.querySelector('#button-material-choice').value,opacity:document.querySelector('#button-glass-opacity').value,blur:document.querySelector('#button-glass-blur').value,backgroundBlur:document.querySelector('#background-blur').value,backgroundOpacity:document.querySelector('#background-opacity').value,color:document.documentElement.dataset.customPalette})")
            report['restored']=restored
            assert restored==dict(material='glass',opacity='20',blur='24',backgroundBlur='10',backgroundOpacity='40',color='#13579b'),restored
            report['checks'].append('Static ptbapp profile IndexedDB and lightweight preferences survive actual reload')
            click('生理参数合成')
            until("!!document.querySelector('.vocal-page iframe')?.contentDocument?.querySelector('#ptb-background-style')")
            until("document.querySelector('.vocal-page iframe').contentDocument.documentElement.dataset.buttonStyle==='plain'")
            pause(300)
            state = js("(()=>{const f=document.querySelector('.vocal-page iframe'),d=f.contentDocument,r=d.documentElement,b=d.querySelector('button'),c=getComputedStyle(b),image=d.querySelector('.app-background-layer'),outer=document.querySelector('.app-background-layer'),a=outer.getBoundingClientRect(),i=image.getBoundingClientRect(),fr=f.getBoundingClientRect(),scale=parseFloat(document.documentElement.style.zoom)||1;return {background:r.dataset.backgroundImage,material:r.dataset.buttonMaterial,root:getComputedStyle(r).backgroundColor,body:getComputedStyle(d.body).backgroundColor,glass:c.backdropFilter,alpha:c.backgroundColor,wallpaper:getComputedStyle(image).backgroundImage===getComputedStyle(outer).backgroundImage,offset:[Math.abs(i.left*scale+fr.left-a.left),Math.abs(i.top*scale+fr.top-a.top)],canvas:[...d.querySelectorAll('canvas')].map(c=>getComputedStyle(c).backgroundColor)};})()")
            assert state['background'] == 'on' and state['material'] == 'glass', state
            assert state['root'] != 'rgba(0, 0, 0, 0)' and state['body'] == 'rgba(0, 0, 0, 0)', state
            assert state['wallpaper'] and max(state['offset']) < 1, state
            assert '24px' in state['glass'] and '0.2' in state['alpha'], state
            assert state['canvas'] and all(color != 'rgba(0, 0, 0, 0)' for color in state['canvas']), state
            report['m10'] = state
            window.view.grab().save(str(out / 'm10-glass.png'))
            report['checks'].append('Actual M10 local wallpaper matches outer image/coordinates for backdrop-filter; custom theme/glass sync and scientific canvases opaque')
            click('设置')
            click('移除背景图')
            until("document.documentElement.dataset.backgroundImage==='off'")
            set_control('#button-material-choice', 'solid')
            pause(200)
            assert js("getComputedStyle(document.querySelector('.appearance-settings button')).backdropFilter==='none'")
            assert not report['errors'], report['errors']
            report['checks'].append('Removal and solid material restore original background/controls')
            report['success'] = True
        finally:
            window.close()
            pause(350)
            report['closed'] = not window.isVisible()
            (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', 'utf-8')
            app.quit()
            print(json.dumps(dict(success=report['success'], report=str(out / 'report.json')), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
