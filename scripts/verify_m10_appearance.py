"""M10 appearance regression in an owned Qt host, with isolated local state."""
import json
import os
import time
import uuid
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'windows')

from PyQt6.QtCore import QEventLoop, QTimer, Qt
from PyQt6.QtWidgets import QApplication
from ptb_desktop.host import Workbench, register_scheme

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/validation/m10/appearance-20261001'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    register_scheme()
    app = QApplication(['m10-appearance-qa'])
    app.setApplicationName('M10-Appearance-owned-QA')
    window = Workbench(ROOT / 'frontend/dist', test=True,
                       vocal_profile=OUT / ('profile-' + uuid.uuid4().hex[:8]),
                       vocal_resources=ROOT / 'resources/vocal_tract/native', start_module='M10')
    window.setWindowFlag(Qt.WindowType.Tool)
    window.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
    window.move(-30000, -30000)
    window.show()

    def settle(ms=240):
        loop = QEventLoop()
        QTimer.singleShot(ms, loop.quit)
        loop.exec()

    def js(code):
        loop, result = QEventLoop(), []
        window.page.runJavaScript(code, lambda value: (result.append(value), loop.quit()))
        QTimer.singleShot(5000, loop.quit)
        loop.exec()
        assert result, 'JavaScript callback timed out'
        return result[0]

    def local(code):
        return js("(()=>{const d=document.querySelector('iframe')?.contentDocument;"
                  "if(!d)return null;const win=d.defaultView;" + code + '})()')

    def until(code, seconds=55):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if local('return ' + code + ';'):
                return
            failure = local("return d.querySelector('#loadIndicator')?.textContent.startsWith('载入失败')?d.querySelector('#loadIndicator').textContent:null;")
            assert not failure, failure
            settle(100)
        window.view.grab().save(str(OUT / 'failed.png'))
        detail = local("return JSON.stringify({engine:d.querySelector('#engineStatus')?.textContent,load:d.querySelector('#loadIndicator')?.textContent,toast:d.querySelector('#toast')?.textContent,visibility:d.visibilityState,html:d.body.innerText.slice(0,400)});")
        raise AssertionError('UI timed out: ' + code + ' ' + str(detail))

    def click(selector):
        local('d.querySelector(' + json.dumps(selector) + ').click();')

    def snapshot():
        return json.loads(local("""
          const rect=e=>{const r=e.getBoundingClientRect();return [r.x,r.y,r.width,r.height]};
          return JSON.stringify({
            geometry:['.studio','.analysis','.inspector','#viewport'].map(s=>rect(d.querySelector(s))),
            viewBox:d.querySelector('.sagittal-canvas').getAttribute('viewBox'),
            page:win.getComputedStyle(d.body).backgroundColor,
            surfaces:['#viewport','.analysis','#lipFront'].map(s=>win.getComputedStyle(d.querySelector(s)).backgroundColor),
            rows:win.getComputedStyle(d.querySelector('.workspace')).gridTemplateRows,
            charts:win.getComputedStyle(d.querySelector('.charts')).display,
            panel:rect(d.querySelector('#monitorPanel')),
            iframe:[win.innerWidth,win.innerHeight]
          });
        """))

    def same_geometry(before, after, context):
        for a, b in zip(before['geometry'], after['geometry']):
            assert all(abs(x-y) <= .1 for x, y in zip(a, b)), (context, before, after)
        assert before['viewBox'] == after['viewBox'], (context, 'model camera changed')

    records = []
    try:
        end = time.monotonic() + 20
        while not js("!!document.querySelector('button[title=\"声道工作台\"]')"):
            assert time.monotonic() < end, 'Host did not load'
            settle(100)
        js("document.querySelector('button[title=\"声道工作台\"]').click()")
        settle(3000)
        print('boot:',local("return JSON.stringify({engine:d.querySelector('#engineStatus')?.textContent,load:d.querySelector('#loadIndicator')?.textContent,visibility:d.visibilityState});"),flush=True)
        until("d.querySelector('#engineStatus')?.textContent.includes('已连接')")
        until("d.body.dataset.posePending==='false'")
        assert local("return !d.querySelector('#previewButton')&&!d.querySelector('#glideButton');")
        local("win.__appearanceErrors=[];win.addEventListener('error',e=>win.__appearanceErrors.push(e.message));")
        for theme in ['light', 'dark']:
            js('document.documentElement.dataset.theme=' + json.dumps(theme))
            until('d.documentElement.dataset.theme===' + json.dumps(theme))
            for width, height in [(1280,800),(1650,1000),(1770,1000),(1920,1080),(2560,1440)]:
                window.setFixedSize(width, height)
                for columns in ['auto','two','three']:
                    name = f'{theme}-{width}-{columns}'
                    local("const e=d.querySelector('#columnLayout');e.value=" + json.dumps(columns) + ";e.dispatchEvent(new Event('change'));")
                    click('[data-analysis=acoustics]')
                    settle()
                    before = snapshot()
                    expected_page = 'rgb(17, 27, 38)' if theme == 'dark' else 'rgb(243, 246, 251)'
                    expected_panel = 'rgb(24, 36, 50)' if theme == 'dark' else 'rgb(255, 255, 255)'
                    assert before['page'] == expected_page, (name,before)
                    assert all(c == expected_panel for c in before['surfaces']), (name,before)
                    assert before['charts'] != 'none', name
                    click('[data-analysis=monitor]')
                    immediate = snapshot()
                    same_geometry(before, immediate, name + '-immediate')
                    settle()
                    monitor = snapshot()
                    same_geometry(before, monitor, name + '-settled')
                    assert monitor['charts'] == 'none', (name,'acoustics leaked into monitor')
                    assert monitor['panel'][2] > 0 and monitor['panel'][3] > 0, name
                    assert local("return [...d.querySelectorAll('.monitor-plots canvas')].every(c=>c.width>0&&c.height>0);")
                    if width == 1770 and columns == 'auto':
                        window.view.grab().save(str(OUT / (name + '-monitor.png')))
                    click('[data-analysis=acoustics]')
                    settle()
                    same_geometry(before, snapshot(), name + '-return')
                    assert local("return win.__appearanceErrors.length===0;"), (name,local("return JSON.stringify(win.__appearanceErrors);"))
                    if width == 1770 and columns == 'auto':
                        window.view.grab().save(str(OUT / (name + '-acoustics.png')))
                    records.append({'name':name,'before':before,'monitor':monitor})
                    print(name, 'passed', flush=True)
        # Moving the existing monitor into a dialog and back must preserve the page.
        click('[data-analysis=monitor]')
        settle()
        before = snapshot()
        click('#monitorExpand')
        settle()
        assert local("return d.querySelector('#monitorDialog').open;")
        same_geometry(before, snapshot(), 'expanded')
        click('#monitorClose')
        settle()
        same_geometry(before, snapshot(), 'dialog-return')
        assert local("return d.querySelector('#monitorPanel').parentElement.id==='monitorHome';")
        click('#tab-sound')
        assert local("return !!d.querySelector('#oneSecondButton')?.onclick&&!!d.querySelector('#liveButton')?.onclick;")
        window.setFixedSize(1770,1000)
        settle()
        window.view.grab().save(str(OUT / 'dark-sound-buttons.png'))
        click('#threeButton')
        settle(400)
        window.view.grab().save(str(OUT / 'dark-three-dimensional.png'))
        assert local("return win.__appearanceErrors.length===0;"), local("return JSON.stringify(win.__appearanceErrors);")
        (OUT / 'result.json').write_text(json.dumps({'status':'passed','cases':records,
          'dialog_roundtrip':True,'removed_buttons_absent':True,'javascript_errors':[]}, ensure_ascii=False,indent=2),encoding='utf-8')
        print('M10 appearance passed:',len(records),'layout/theme cases; dialog; buttons; 3D',flush=True)
    finally:
        window.closing=True
        window.close()
        settle(300)
        app.quit()


if __name__ == '__main__':
    main()
