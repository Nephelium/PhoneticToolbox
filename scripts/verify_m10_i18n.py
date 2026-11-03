"""Exercise the real isolated Qt M10 frame across live locale switches."""
from pathlib import Path
import argparse,hashlib,json,os,sys,time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from workbench_source import configure
from verification_artifacts import verification_run
from manual.capture_locale import CaptureLocale

def main(dist):
    configure()
    os.environ['QT_QPA_PLATFORM']='windows'
    os.environ['QTWEBENGINE_CHROMIUM_FLAGS']='--mute-audio'
    for key in ('QT_ENABLE_HIGHDPI_SCALING','QT_SCALE_FACTOR','QT_SCREEN_SCALE_FACTORS'):os.environ.pop(key,None)
    from PyQt6.QtCore import QEventLoop,QTimer,Qt
    from PyQt6.QtWidgets import QApplication
    from PyQt6 import sip
    from ptb_desktop.host import Workbench,register_scheme
    from ptb_worker.local_workspace import prepare_workspace
    sources=[ROOT/'frontend/src/modules/vocal-tract/VocalTractPage.vue',*sorted((ROOT/'frontend/public/vocal-tract').glob('*.js')),*sorted((ROOT/'frontend/public/vocal-tract').glob('*.mjs')),ROOT/'frontend/public/vocal-tract/index.html']
    snapshot=[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}for p in sources]
    register_scheme();app=QApplication(['owned-m10-language-verification'])
    with verification_run('m10-i18n','live-switch','Real Qt/native engine; isolated state; no media playback; language-only updates preserve posture, frames, drafts and DOM nodes')as(out,scratch):
        db,cache=prepare_workspace(scratch/'state',ROOT/'backend/migrations')
        window=Workbench(dist.resolve(),test=True,jobs_path=db,local_files_root=cache,vocal_resources=ROOT/'resources/vocal_tract/native',vocal_profile=scratch/'vocal',start_module='M10')
        window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen);window.resize(window.screen().geometry().width(),window.screen().geometry().height());window.showMaximized();window.view.setZoomFactor(1);window.page.setAudioMuted(True)
        report={'success':False,'sources':snapshot,'checks':[],'captures':[],'physicalCompositorVerified':False}
        def pause(ms=120):
            loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
        def js(code):
            loop=QEventLoop();values=[]
            wrapped='(()=>{try{return {ok:true,value:eval('+json.dumps(code)+')??null};}catch(e){return {ok:false,error:String(e)}}})()'
            window.page.runJavaScript(wrapped,lambda v:(values.append(v),loop.quit()));QTimer.singleShot(15000,loop.quit);loop.exec()
            if not values or not values[0]or not values[0].get('ok'):raise RuntimeError(str(values)+' '+code)
            return values[0].get('value')
        def until(code,seconds=75):
            end=time.monotonic()+seconds
            while time.monotonic()<end:
                if js(code):return
                pause()
            raise AssertionError(code+'\n'+str(js('document.body.innerText.slice(-1800)'))+'\n'+str(js('document.querySelector(".vocal-page iframe")?.contentDocument?.body.innerText.slice(-2500)')))
        def frame(code):return js('(()=>{const f=document.querySelector(".vocal-page iframe");const d=f.contentDocument;'+code+'})()')
        try:
            until('document.querySelector(".vocal-page iframe")?.contentDocument?.body?.dataset.engineState==="ready"')
            until('document.querySelector(".vocal-page iframe")?.contentDocument?.body.dataset.posePending==="false"')
            frame('d.querySelector("#tab-motion").click();d.querySelector("#captureFrame").click();')
            until('document.querySelector(".vocal-page iframe").contentDocument.querySelectorAll(".pose-card").length===1')
            pause(650)
            js('window.m10LocaleRequests=[];window.addEventListener("message",e=>{if(e.data?.type==="m10-request"&&!['+','.join(json.dumps(x)for x in ('heartbeat','status'))+'].includes(e.data.op))window.m10LocaleRequests.push({op:e.data.op,body:e.data.body});});')
            frame('window.m10Identity={frame:f,doc:d,canvas:d.querySelector("#pitchCanvas"),ipa:d.querySelector("#sequenceIpa"),input:d.querySelector("#presetSetName"),revision:d.body.dataset.poseRevision,url:f.src,section:d.querySelector("#sliceRange").value};window.m10Identity.ipa.value="a i u draft";window.m10Identity.input.value="Author set draft";d.querySelector("#savePreset").click();d.querySelector("#presetInputButton").click();d.querySelector("#presetInput").value="";d.querySelector("#presetInputSave").click();')
            until('document.querySelector(".vocal-page iframe").contentDocument.querySelector("#presetStatus").textContent.includes("音标")')
            for locale in ('en','ja','zh-CN','en'):
                ui=CaptureLocale(locale,('vocal-tract','vocal-tract-frame','workbench','components'));ui.prepare_qt(js,until)
                code=json.dumps(locale);until('document.querySelector(".vocal-page iframe").contentDocument.documentElement.lang==='+code+'&&document.querySelector(".vocal-page iframe").contentDocument.documentElement.dataset.fontLocale==='+code)
                pause(300)
                state=frame('const i=window.m10Identity;return {locale:d.documentElement.lang,fontLocale:d.documentElement.dataset.fontLocale,sameFrame:i.frame===f,sameDocument:i.doc===d,sameCanvas:i.canvas===d.querySelector("#pitchCanvas"),sameIpa:i.ipa===d.querySelector("#sequenceIpa"),sameInput:i.input===d.querySelector("#presetSetName"),ipa:i.ipa.value,input:i.input.value,revision:d.body.dataset.poseRevision,section:d.querySelector("#sliceRange").value,url:f.src,error:d.querySelector("#presetStatus").textContent,parameter:d.querySelector("label[for=p-HX]").textContent,frames:d.querySelectorAll(".pose-card").length};')
                assert all(state[k]for k in ('sameFrame','sameDocument','sameCanvas','sameIpa','sameInput')),state
                assert state['ipa']=='a i u draft'and state['input']=='Author set draft'and state['frames']==1,state
                assert state['parameter'].endswith('HX'),state
                assert ('Enter a phonetic symbol.'if locale=='en'else'音声記号を入力してください。'if locale=='ja'else'请输入音标。')in state['error'],state
                assert not js('window.m10LocaleRequests'),js('window.m10LocaleRequests')
                assert frame('return d.body.dataset.poseRevision===window.m10Identity.revision&&f.src===window.m10Identity.url&&d.querySelector("#sliceRange").value===window.m10Identity.section;'),state
                report['checks'].append(state)
                if locale in ('en','ja'):
                    image=window.view.grab().toImage();path=out/(locale+'-'+str(len(report['checks']))+'.png');image.save(str(path));report['captures'].append({'locale':locale,'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'image':[image.width(),image.height()]})
            report['requests']=js('window.m10LocaleRequests');report['frameResources']=frame('return f.contentWindow.performance.getEntriesByType("resource").filter(r=>r.name.includes("/vocal-tract/locales/")).map(r=>r.name);')
            frame("f.contentWindow.eval(\"import('ptbapp://app/vocal-tract/i18n.mjs').then(m=>window.m10LocaleCache=m.localeCacheInfo()).catch(e=>window.m10LocaleCache={error:String(e)})\");")
            until('!!document.querySelector(".vocal-page iframe").contentWindow.m10LocaleCache')
            report['localeCache']=frame('return f.contentWindow.m10LocaleCache;')
            assert report['localeCache']['loads']=={'en':1,'ja':1},report['localeCache']
            # A second, owned frame exercises the real failure DOM without
            # corrupting the native profile or submitting scientific work.
            js('''(()=>{const fixture=document.createElement('iframe');fixture.id='m10ErrorFixture';fixture.hidden=true;window.m10ErrorOps=[];
            window.addEventListener('message',event=>{if(event.source!==fixture.contentWindow||event.data?.type!=='m10-request')return;
              const {id,op}=event.data;window.m10ErrorOps.push(op);
              setTimeout(()=>fixture.contentWindow.postMessage(op==='meta'?{type:'m10-response',id,ok:false,error:'literal fallback',recoveryAvailable:true,
                display:{type:'joined',parts:[{type:'message',key:'platform.invalid_saved_data'},' E_TEST 007',
                 {type:'message',key:'platform.original_data_is_preserved_export_a_recovery_copy'}]}}:{type:'m10-response',id,ok:true,value:{saved:true}},location.origin),250);
            });fixture.src=document.querySelector('.vocal-page iframe').src;document.body.append(fixture);})()''')
            until('document.querySelector("#m10ErrorFixture").contentDocument?.body?.dataset.engineState==="failed"')
            report['delayedErrorChecks']=[]
            for locale in ('en','ja','zh-CN'):
                CaptureLocale(locale,('vocal-tract','workbench','components')).prepare_qt(js,until)
                js('document.querySelector("#m10ErrorFixture").contentWindow.postMessage({type:"m10-locale",locale:'+json.dumps(locale)+'},location.origin)')
                until('document.querySelector("#m10ErrorFixture").contentDocument.documentElement.lang==='+json.dumps(locale))
                state=js('(()=>{const d=document.querySelector("#m10ErrorFixture").contentDocument;return {locale:d.documentElement.lang,error:d.querySelector("#loadIndicator").textContent,recovery:d.querySelector("#loadIndicator button")?.textContent};})()')
                assert state['recovery']==CaptureLocale(locale).key('modules.vocal-tract-frame.text.export_recovery_copy_99'),state
                assert CaptureLocale(locale).key('platform.original_data_is_preserved_export_a_recovery_copy')in state['error'],state
                assert 'E_TEST 007'in state['error'],state
                report['delayedErrorChecks'].append(state)
            js('document.querySelector("#m10ErrorFixture").contentDocument.querySelector("#loadIndicator button").click()')
            until('window.m10ErrorOps.includes("profile/recovery")')
            report['errorFixtureRequests']=js('window.m10ErrorOps')
            report['success']=True
        finally:
            worker_log=scratch/'vocal/native-worker.log'
            if worker_log.is_file():report['nativeWorkerLog']=worker_log.read_text(encoding='utf-8')
            report['sourceChangesDuringRun']=[r['path']for r in snapshot if hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()!=r['sha256']]
            (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
            window.closing=True;window.close();app.processEvents();sip.delete(window);app.processEvents();sip.delete(app)
            print(json.dumps({'success':report['success'],'report':str(out/'report.json')},ensure_ascii=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--frontend-dist',type=Path,required=True);main(parser.parse_args().frontend_dist)
