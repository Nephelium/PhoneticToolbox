"""Real Chromium audio permission/labels in owned Qt or Chrome, fake hardware only."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from unittest.mock import patch
from verification_artifacts import verification_run

ROOT=Path(__file__).resolve().parents[1]
PROBE='''
window.qaProbes=[];
const originalGetUserMedia=navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
Object.defineProperty(navigator.mediaDevices,'getUserMedia',{value:async constraints=>{
 const probe={constraints};window.qaProbes.push(probe);
 if(window.qaPermissionGate)await window.qaPermissionGate();
 try{const stream=await originalGetUserMedia(constraints);probe.stream=stream;return stream;}
 catch(e){probe.failure={name:e.name,message:e.message};throw e;}
}});
'''


def qt():
    os.environ['QT_QPA_PLATFORM']='offscreen'
    os.environ['QTWEBENGINE_CHROMIUM_FLAGS']='--disable-gpu --mute-audio --use-fake-device-for-media-stream'
    from workbench_source import configure
    configure()
    from PyQt6.QtCore import QEvent, QEventLoop, QTimer, Qt, QPoint
    from PyQt6.QtTest import QTest
    from PyQt6.QtWebEngineCore import QWebEngineScript
    from PyQt6.QtWidgets import QApplication, QMessageBox
    from ptb_desktop.host import Workbench, register_scheme
    from ptb_worker.local_workspace import prepare_workspace
    register_scheme();app=QApplication(['M05 fake hardware permission QA'])
    with verification_run('m05-devices','qt-permissions','Actual Qt permission gate and Chromium fake audio input; no media API stubs and no physical hardware') as (out,scratch):
        print(out,flush=True)
        db,cache=prepare_workspace(scratch/'state',ROOT/'backend/migrations')
        window=Workbench(ROOT/'frontend/dist',test=True,jobs_path=db,local_files_root=cache,vocal_profile=scratch/'vocal',start_module='M05')
        script=QWebEngineScript();script.setWorldId(QWebEngineScript.ScriptWorldId.MainWorld)
        script.setInjectionPoint(QWebEngineScript.InjectionPoint.DocumentCreation);script.setSourceCode(PROBE)
        window.page.scripts().insert(script);window.resize(1440,900);window.show()
        report=dict(success=False,scope='Windows Qt/Chromium real permission flow with fake hardware',checks=[])

        def pause(ms=50):
            loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()

        def js(code):
            values=[];loop=QEventLoop()
            window.page.runJavaScript(code,lambda value:(values.append(value),loop.quit()))
            QTimer.singleShot(5000,loop.quit);loop.exec()
            return values[0] if values else None

        def until(code):
            deadline=time.monotonic()+25
            while time.monotonic()<deadline:
                if js(code):return
                pause()
            raise AssertionError(code+'\n'+str(js('document.body.innerText')))

        mic='document.querySelector("[aria-label=麦克风]")'
        def click():
            point=js('(()=>{const b=[...document.querySelectorAll(".lip-page button")].find(b=>b.textContent.trim()==="授权并刷新设备");b.scrollIntoView({block:"nearest"});const r=b.getBoundingClientRect();return [r.x+r.width/2,r.y+r.height/2];})()')
            window.activateWindow();window.view.setFocus()
            QTest.mouseClick(window.view.focusProxy(),Qt.MouseButton.LeftButton,pos=QPoint(round(point[0]),round(point[1])))
        try:
            until('!!'+mic+'&&'+mic+'.options.length===1&&!'+mic+'.disabled')
            assert js('qaProbes.length')==0
            with patch.object(QMessageBox,'question',return_value=QMessageBox.StandardButton.No) as prompt:
                click();until('document.querySelector(".lip-page").innerText.includes("未获得麦克风权限")')
                report['first_failure']=js('qaProbes[0].failure')
                assert window.m05_media.status()['decision']=='denied_user'
                prompt.assert_called_once()
            assert js(mic+'.options.length')==1
            report['checks'].append('No automatic capture; real Qt audio permission refused without revealing names')
            with patch.object(QMessageBox,'question',return_value=QMessageBox.StandardButton.Yes) as prompt:
                click();until(mic+'.options.length>1&&!'+mic+'.disabled')
                prompt.assert_called_once();assert '读取后立即释放' in prompt.call_args.args[2]
            assert js('qaProbes.map(p=>p.constraints)')==[dict(audio=True,video=False)]*2
            assert js('qaProbes.at(-1).stream.getTracks().every(t=>t.readyState==="ended")')
            labels=js('[...document.querySelector("[aria-label=麦克风]").options].map(o=>o.text)')
            assert any('Fake' in label for label in labels),labels
            report['labels']=labels;report['checks'].append('Retry resets Qt cached denial; real fake audio reveals names and all temporary tracks end')
            js('[...document.querySelectorAll(".lip-page button")].find(b=>b.textContent.trim()==="刷新设备").click()')
            until('!'+mic+'.disabled');assert js('qaProbes.length')==2
            for theme in ('light','dark'):
                js('document.documentElement.dataset.theme='+json.dumps(theme));pause(300)
                assert js('document.documentElement.scrollWidth<=innerWidth+1')
                window.view.grab().save(str(out/(theme+'.png')))
            report['checks'].append('Names survive stream release; refresh does not reopen audio; light/dark layouts fit')
            report['success']=True
        finally:
            report['page']=js('document.body.innerText')
            report['permission_events']=window.m05_media.events
            report['probes']=js('qaProbes.map(p=>({constraints:p.constraints,failure:p.failure,tracks:p.stream?.getTracks().map(t=>t.readyState)}))')
            (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf-8')
            window.accept_close();pause(100);window.page.deleteLater();app.processEvents()
            window.profile.deleteLater();app.sendPostedEvents(None,QEvent.Type.DeferredDelete);app.processEvents()
        print(json.dumps(dict(success=report['success'],out=str(out))),flush=True)


CHROME=r'''
const fs=require('node:fs/promises'),path=require('node:path'),os=require('node:os'),assert=require('node:assert/strict');
(async()=>{
 const [out,scratch]=process.argv.slice(2),root=process.cwd(),report={success:false,scope:'Independent Windows Chrome, built shared page, real permissions with Chromium fake hardware',checks:[]};
 const {preview}=await import(require('node:url').pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')).href);
 const server=await preview({root:path.join(root,'frontend'),logLevel:'error',preview:{host:'127.0.0.1',port:0}});
 const origin=`http://127.0.0.1:${server.httpServer.address().port}`;
 const {chromium}=require(path.join(os.homedir(),'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 let context;
 try{
  context=await chromium.launchPersistentContext(path.join(scratch,'chrome-profile'),{executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,viewport:{width:1440,height:900},args:['--use-fake-device-for-media-stream','--mute-audio']});
  await context.addInitScript(PROBE_CODE);
  const page=await context.newPage(),errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto(origin);await page.getByRole('button',{name:'唇形提取',exact:true}).click();
  const mic=page.getByLabel('麦克风',{exact:true}),authorize=page.getByRole('button',{name:'授权并刷新设备',exact:true});
  await authorize.waitFor();assert.equal(await mic.locator('option').count(),1);assert.equal(await page.evaluate(()=>qaProbes.length),0);
  const cdp=await context.newCDPSession(page);
  let allow=false;await page.exposeFunction('qaPermissionGate',async()=>{if(allow)await cdp.send('Browser.setPermission',{permission:{name:'microphone'},setting:'granted',origin});});
  await cdp.send('Browser.setPermission',{permission:{name:'microphone'},setting:'denied',origin});
  await authorize.click();await page.getByText('未获得麦克风权限。',{exact:false}).waitFor();
  report.checks.push('Mount does not capture; real browser permission rejection leaves list unlabelled and retry available');
  allow=true;
  await authorize.click();await page.getByRole('button',{name:'刷新设备',exact:true}).waitFor();
  report.labels=await mic.locator('option').allTextContents();assert(report.labels.some(v=>v.includes('Fake')));
  assert.deepEqual(await page.evaluate(()=>qaProbes.map(p=>p.constraints)),[{audio:true,video:false},{audio:true,video:false}]);
  assert(await page.evaluate(()=>qaProbes.at(-1).stream.getTracks().every(t=>t.readyState==='ended')));
  const selected=await mic.locator('option').nth(1).getAttribute('value');await mic.selectOption(selected);
  await page.getByRole('button',{name:'刷新设备',exact:true}).click();await mic.waitFor({state:'visible'});
  assert.equal(await mic.inputValue(),selected);assert.equal(await page.evaluate(()=>qaProbes.length),2);
  report.checks.push('Real fake microphone labels exposed, tracks released, named selection and refresh retain IDs without reopening audio');
  for(const theme of ['light','dark']){await page.evaluate(t=>document.documentElement.dataset.theme=t,theme);await page.screenshot({path:path.join(out,theme+'.png')});assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));}
  assert.deepEqual(errors,[]);report.checks.push('Light/dark layouts fit; no page errors');report.success=true;
 }finally{if(context)await context.close();await server.close();await fs.writeFile(path.join(out,'report.json'),JSON.stringify(report,null,2));}
})().catch(e=>{console.error(e);process.exitCode=1;});
'''


def chrome():
    with verification_run('m05-devices','chrome-permissions','Independent Chrome built UI, real audio permissions with fake Chromium hardware; isolated temporary profile') as (out,scratch):
        print(out,flush=True)
        code=CHROME.replace('PROBE_CODE',json.dumps(PROBE))
        script=scratch/'check.cjs';script.write_text(code,'utf-8')
        node=Path.home()/'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe'
        done=subprocess.run([str(node),str(script),str(out),str(scratch)],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace',creationflags=subprocess.CREATE_NO_WINDOW,timeout=150)
        (out/'browser.log').write_text(done.stdout+'\n'+done.stderr,'utf-8')
        if done.returncode:raise RuntimeError('Chrome permission QA failed: '+done.stdout+'\n'+done.stderr)
        print((out/'report.json').read_text('utf-8'),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--browser',choices=('qt','chrome'),default='qt')
    args=parser.parse_args();qt() if args.browser=='qt' else chrome()
