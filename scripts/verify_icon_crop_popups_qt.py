"""Current static Qt UI: icon crop/key, native replace, mode cards and opaque popups."""
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
PARTS=('scripts','desktop/src','backend/src','packages/phonetic_core/src')
for part in PARTS:sys.path.insert(0,str(ROOT/part))
os.environ['PYTHONPATH']=os.pathsep.join(str(ROOT/part) for part in PARTS)
os.environ['QTWEBENGINE_CHROMIUM_FLAGS']='--mute-audio'


def main():
    from verification_artifacts import verification_run
    from PyQt6.QtCore import QEventLoop,QTimer
    from PyQt6.QtWidgets import QApplication
    from ptb_desktop.host import Workbench,register_scheme
    from ptb_desktop.icon_preferences import decode_icon
    from ptb_worker.local_acoustic_files import initialize_local_files
    with verification_run('icon-crop-popups','qt','Current static Workbench, isolated schema DB and native icon preferences; generated 400x240 white/red PNG; real crop/key/import/replace/original retention; wallpaper dialogs/dropdowns/mode cards and M10 dialog') as (out,scratch):
        db=scratch/'jobs.sqlite3'
        with sqlite3.connect((ROOT/'output/validation/p06/local-state.sqlite3').as_uri()+'?mode=ro',uri=True) as source,sqlite3.connect(db) as target:source.backup(target)
        cache=scratch/'cache';cache.mkdir();initialize_local_files(cache)
        register_scheme();app=QApplication(['owned-icon-crop-popups'])
        window=Workbench(ROOT/'frontend/dist',test=True,jobs_path=db,local_files_root=cache,vocal_profile=scratch/'vocal')
        window.resize(1440,900);window.show();report=dict(success=False,checks=[],errors=[])
        window.page.renderProcessTerminated.connect(lambda status,code:report['errors'].append([status.name,code]))
        def pause(ms=150):
            loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
        def js(code):
            loop,values=QEventLoop(),[]
            window.page.runJavaScript(code,lambda v:(values.append(v),loop.quit()));QTimer.singleShot(10000,loop.quit);loop.exec()
            assert values,'JavaScript timeout';return values[0]
        def until(code):
            end=time.monotonic()+40
            while time.monotonic()<end:
                if js(code):return
                pause()
            raise AssertionError(code)
        def click(label):
            expression='[...document.querySelectorAll("button")].find(b=>b.offsetParent&&(b.getAttribute("aria-label")||b.textContent.trim())==='+json.dumps(label)+')'
            until('!!('+expression+')&&!('+expression+').disabled');js(expression+'.click()')
        def select(selector,value):
            js('(()=>{const e=document.querySelector('+json.dumps(selector)+');e.value='+json.dumps(value)+';e.dispatchEvent(new Event("change",{bubbles:true}));})()')
        def popup(selector):
            return js('(()=>{const e=document.querySelector('+json.dumps(selector)+');const c=getComputedStyle(e);return {background:c.backgroundColor,image:c.backgroundImage,opacity:c.opacity};})()')
        try:
            until("!!document.querySelector('#tool-sidebar')&&document.fonts.status==='loaded'")
            click('设置');js("document.querySelector('#settings-tab-icons').click()")
            js("(()=>{const c=document.createElement('canvas');c.width=400;c.height=240;const x=c.getContext('2d');x.fillStyle='#fff';x.fillRect(0,0,400,240);x.fillStyle='#dc281e';x.fillRect(140,70,120,100);x.fillStyle='#fff';x.fillRect(190,105,20,20);c.toBlob(blob=>{window.__iconFixture=blob;const d=new DataTransfer();d.items.add(new File([blob],'原图测试.png',{type:'image/png'}));const e=document.querySelector('.icon-file-input');e.files=d.files;e.dispatchEvent(new Event('change',{bubbles:true}));});})()")
            until("!!document.querySelector('.crop-selection')")
            assert js("document.querySelector('[aria-label=图标裁剪比例]').value==='square'")
            js("(()=>{const e=[...document.querySelectorAll('dialog input[type=checkbox]')][0];e.checked=true;e.dispatchEvent(new Event('change',{bubbles:true}));})()")
            pause(200)
            window.view.grab().save(str(out/'crop-white-key.png'))
            click('应用裁剪');until("!document.querySelector('.icon-crop-dialog')&&document.querySelectorAll('.custom-icon-item').length===1")
            original=window.icon_preferences.read();image=decode_icon(original['custom'][0]['dataUrl'])[1]
            assert image.pixelColor(0,0).alpha()==0 and image.pixelColor(128,128).alpha()==255
            report['checks'].append('Native import uses default square crop; white key saves transparent edges and enclosed white detail; canonical native PNG applied')
            loop=QEventLoop();window.page.loadFinished.connect(loop.quit);window.page.triggerAction(window.page.WebAction.Reload);QTimer.singleShot(15000,loop.quit);loop.exec();window.page.loadFinished.disconnect(loop.quit)
            until("!!document.querySelector('#tool-sidebar')");click('设置');js("document.querySelector('#settings-tab-icons').click()")
            click('重新裁剪图标 原图测试.png');until("document.querySelector('.crop-surface img')?.naturalWidth===400")
            select('[aria-label=图标裁剪比例]','original');click('应用裁剪');until("!document.querySelector('.icon-crop-dialog')")
            replaced=window.icon_preferences.read();assert len(replaced['custom'])==1 and replaced['selected']!=original['selected']
            report['checks'].append('Original retains 400x240 pixels across actual Qt reload; native replace changes one existing icon slot')
            js("document.querySelector('#settings-tab-appearance').click()")
            # Reload cleared the in-memory variable; generate this fixture again for wallpaper.
            js("(()=>{const c=document.createElement('canvas');c.width=400;c.height=240;const x=c.getContext('2d');x.fillStyle='#38b55a';x.fillRect(0,0,400,240);c.toBlob(blob=>{const d=new DataTransfer();d.items.add(new File([blob],'背景测试.png',{type:'image/png'}));const e=document.querySelector('input[type=file][aria-label=上传背景图]');e.files=d.files;e.dispatchEvent(new Event('change',{bubbles:true}));});})()")
            until("document.documentElement.dataset.backgroundImage==='on'")
            click('浅色');until("document.documentElement.dataset.theme==='light'")
            window.view.grab().save(str(out/'mode-cards.png'))
            for mode in ('light','dark'):
                click('浅色' if mode=='light' else '深色')
                until("document.documentElement.dataset.theme==="+json.dumps(mode))
                assert js("document.querySelectorAll('.theme-mode-card').length===3&&!document.querySelector('.mode-choices')")
                for material in ('solid','glass'):
                    select('#button-material-choice',material);click('关于');until("!!document.querySelector('dialog[open]')")
                    about=popup('dialog[open]');assert about['background']!='rgba(0, 0, 0, 0)' and about['image']=='none' and about['opacity']=='1',about
                    window.view.grab().save(str(out/('about-'+mode+'-'+material+'.png')));click('关闭对话框')
                    click('配色方案');until("!!document.querySelector('#palette-list')")
                    menu=popup('#palette-list');assert menu['background']==about['background'] and menu['image']=='none',menu
                    window.view.grab().save(str(out/('dropdown-'+mode+'-'+material+'.png')))
                    js("document.querySelector('#palette-list button').click()")
            report['checks'].append('Actual static mode cards switch light/dark; Qt about modal and palette dropdown opaque across 2 themes x 2 materials')
            click('生理参数合成');until("!!document.querySelector('.vocal-page iframe')?.contentDocument?.querySelector('#ptb-background-style')")
            until("!!document.querySelector('.vocal-page iframe')?.contentDocument?.querySelector('#aboutButton')")
            # This case checks the existing iframe dialog's actual CSS/compositing.
            # Scientific-engine readiness and its button wiring are a separate scope.
            report['m10_before_dialog']=js("(()=>{const d=document.querySelector('.vocal-page iframe').contentDocument;return {engine:d.body.dataset.engineState,indicator:d.querySelector('#loadIndicator')?.textContent,buttonReady:typeof d.querySelector('#aboutButton').onclick==='function'};})()")
            js("document.querySelector('.vocal-page iframe').contentDocument.querySelector('#aboutDialog').showModal()")
            until("document.querySelector('.vocal-page iframe').contentDocument.querySelector('#aboutDialog').open")
            m10=js("(()=>{const d=document.querySelector('.vocal-page iframe').contentDocument,e=d.querySelector('#aboutDialog'),c=getComputedStyle(e);return {background:c.backgroundColor,image:c.backgroundImage,opacity:c.opacity};})()")
            assert m10['background']!='rgba(0, 0, 0, 0)' and m10['image']=='none' and m10['opacity']=='1',m10
            window.view.grab().save(str(out/'m10-opaque-dialog.png'));js("document.querySelector('.vocal-page iframe').contentDocument.querySelector('#aboutDialog').close()")
            report['checks'].append('Existing M10 iframe dialog displayed through HTML dialog API is opaque with no wallpaper; scientific startup/button readiness not included')
            report['dpr']=js('devicePixelRatio');assert not report['errors'],report['errors'];report['success']=True
        finally:
            window.close();pause(350);report['closed']=not window.isVisible();(out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n','utf-8');app.quit();print(json.dumps(dict(success=report['success'],report=str(out/'report.json')),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
