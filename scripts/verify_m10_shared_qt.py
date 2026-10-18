"""Owned Windows Qt acceptance for the shared sequence/configuration editor."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parent))
from workbench_source import ROOT, configure
from verification_artifacts import verification_run, write_json
configure(ROOT)
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS','--mute-audio')


def main():
    if '--child' not in sys.argv:
        from ptb_desktop.vocal_tract.process_guard import OwnedJob
        with verification_run('m10-shared-20261018','qt','Actual Qt UI/native IPC/file adapters; synthetic profile, test mode does not open an audio device') as (out,scratch):
            with (out/'qt.log').open('w',encoding='utf-8') as log:
                process=subprocess.Popen([sys.executable,'-B','-X','utf8',__file__,'--child',str(out),str(scratch)],cwd=ROOT,stdout=log,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW)
                job=OwnedJob(process)
                try:assert process.wait(timeout=240)==0,str(out/'qt.log')
                finally:
                    if process.poll() is None:process.terminate();process.wait(timeout=10)
                    job.close()
            print(out/'report.json')
        return
    from PyQt6.QtCore import QEventLoop,QTimer,Qt
    from PyQt6.QtWidgets import QApplication
    from ptb_desktop.host import Workbench,register_scheme
    out,scratch=map(Path,sys.argv[-2:]);report={'passed':False,'checks':[]}
    register_scheme();app=QApplication(['M10-shared-owned-QA'])
    w=Workbench(Path(os.environ.get('PTB_M10_QA_FRONTEND',ROOT/'frontend/dist')),test=True,vocal_profile=scratch/'profile',vocal_resources=Path(os.environ.get('PTB_M10_TEST_RESOURCES',ROOT/'resources/vocal_tract/native')),start_module='M10')
    w.vocal.config['playback_allowed']=False
    w.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen);w.resize(1650,1050);w.show()
    pack=scratch/'shared.ptb-poses.json';document=scratch/'sequence.json'
    w.bridge.test_vocal_picker=lambda op:str(pack if op.startswith('preset-pack/') else document)
    def pause(ms=80):
        loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
    def js(code):
        loop=QEventLoop();box=[];w.page.runJavaScript(code,lambda v:(box.append(v),loop.quit()));QTimer.singleShot(10000,loop.quit);loop.exec();assert box,'JS timeout';return box[0]
    def local(code):return js('(()=>{const f=document.querySelector("iframe"),d=f?.contentDocument,win=f?.contentWindow;if(!d)return null;'+code+'})()')
    def until(code,seconds=50):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            if local('return '+code):return
            pause()
        raise AssertionError(code)
    def settled():until('d.body.dataset.engineState==="ready"&&d.body.dataset.posePending==="false"')
    def click(selector):
        assert local('const e=d.querySelector('+json.dumps(selector)+');if(!e||e.disabled)return false;e.click();return true;'),selector
        pause()
    def value(selector,value,event='change'):
        assert local('const e=d.querySelector('+json.dumps(selector)+');if(!e)return false;e.value='+json.dumps(str(value))+';e.dispatchEvent(new win.Event('+json.dumps(event)+',{bubbles:true}));return true;')
        pause()
    def screenshot(name):pause(120);w.view.grab().save(str(out/(name+'.png')))
    try:
        settled();meta=w.vocal.invoke('meta',{});assert meta['default_f0']==120 and meta['max_animation_duration']==60 and meta['geometry_method']=='m10/r21'
        assert meta['audio_locked'] is True
        value('#lip-shape-3',.3,'input');settled()
        click('#savePreset');click('#presetInputButton');value('#presetInput','f');click('#presetInputSave');until('!d.querySelector("#presetDialog").open')
        saved=w.vocal.invoke('presets/load',{});assert saved[0]['lip_shape']==[0,0,0,.3]
        click('#customPreset');click('#presetShare');click('#packExport');until('d.querySelector("#packStatus").textContent.includes("已导出")')
        payload=json.loads(pack.read_text('utf-8'));assert payload['version']==3 and payload['presets'][0]['name']=='f'
        payload['presets'][0]['name']='v';payload['presets'][0]['f0']=207;pack.write_text(json.dumps(payload,ensure_ascii=False),encoding='utf-8')
        click('#packOpen');until('d.querySelector("#packStatus").textContent.includes("尚未写入")');screenshot('pack-preview');value('#packTarget','current');click('#packImport');until('d.querySelector("#packStatus").textContent.includes("新增 1 个")')
        assert len(w.vocal.invoke('presets/load',{}))==2
        click('#packClose');click('#presetCancel');report['checks'].append('independent lip native IPC and atomic selected pack save/open/import')
        click('#tab-motion');value('#sequenceIpa','ta');click('#generateIpa');until('d.querySelectorAll(".pose-card").length===2')
        sequence=w.vocal.invoke('keyframes/load',{});assert sequence['frames'][0]['stop_timing'] and all('frame_id' in p for p in sequence['frames'])
        click('#addFeature');value('#featureParameter','lower_lip_y');value('#featureValue',.1);click('#featureForm button[type=submit]');until('d.querySelectorAll(".feature-row").length===1')
        click('#exportFrames');until('d.querySelector("#frameStatus").textContent.includes("已导出")')
        data=json.loads(document.read_text('utf-8'));assert data['version']==8 and len(data['features'])==1
        click('#clearFrames');click('#importFrames');until('d.querySelectorAll(".pose-card").length===2')
        sequence=w.vocal.invoke('keyframes/load',{});assert sequence['features']==data['features'] and sequence['frames']==data['frames']
        clip=w.vocal.invoke('animation/prepare',sequence);assert clip['duration']==.28
        picture=w.vocal.invoke('animation/picture',{'id':clip['id'],'index':0});assert picture['state']['lip_shape'][3]==.1
        report['checks'].append('IPA stop phases, regional feature and v8 native file round trip reach original worker')
        screenshot('sequence')
        value('#speakerSelect','W02');until('d.body.dataset.engineState==="ready"&&d.querySelector("#speakerSelect")?.value==="W02"');settled();assert local('return +d.querySelector("#f0Range").value')==200
        assert w.vocal.invoke('keyframes/load',{})['frames']==[]
        value('#speakerSelect','JD2');until('d.body.dataset.engineState==="ready"&&d.querySelector("#speakerSelect")?.value==="JD2"');settled();assert w.vocal.invoke('keyframes/load',{})['frames']==data['frames']
        for theme in ('light','dark'):
            js('document.querySelector("button[title=设置]").click()');pause();assert js('(()=>{const e=document.querySelector("[data-preview-mode='+theme+']");if(!e)return false;e.click();return true;})()');js('document.querySelector("#tab-M10").click()');pause();click('#tab-motion');assert not js('document.body.innerText.includes("由于服务器性能限制")');screenshot('sequence-'+theme)
        report['checks'].append('model profiles, 120/200 Hz defaults and light/dark Qt rendering')
        click('#tab-organs');click('#customPreset');value('#presetSetName','Qt 原始套装');click('#presetSetRename');until('d.querySelector("#presetStatus").textContent.includes("当前套装：Qt 原始套装")')
        original_library=w.vocal.invoke('library/load',{});original_id=original_library['active']
        value('#presetSetName','Qt 副本');click('#presetSetCopy');until('d.querySelector("#presetStatus").textContent.includes("当前套装：Qt 副本")')
        assert len(w.vocal.invoke('presets/load',{}))==2
        value('#presetSetName','Qt 空套装');click('#presetSetCreate');until('d.querySelector("#presetStatus").textContent.includes("当前套装：Qt 空套装")')
        assert w.vocal.invoke('presets/load',{})==[]
        value('#presetSet',original_id);until('d.querySelector("#presetStatus").textContent.includes("当前套装：Qt 原始套装")')
        assert len(w.vocal.invoke('presets/load',{}))==2 and w.vocal.invoke('keyframes/load',{})['frames']==data['frames']
        payload['geometry_method']='m10/r20';payload['version']=2;payload['title']='Qt 旧包';pack.write_text(json.dumps(payload,ensure_ascii=False),encoding='utf-8')
        click('#presetShare');click('#packOpen');until('d.querySelector("#packStatus").textContent.includes("尚未写入")')
        assert local('return d.querySelector("#packImport").disabled') is True
        assert len(w.vocal.invoke('library/load',{})['sets'])==3
        click('#packRecompute');click('#packImport');until('d.querySelector("#packStatus").textContent.includes("新增 1 个")')
        library=w.vocal.invoke('library/load',{});assert len(library['sets'])==4
        click('#packClose');screenshot('libraries');click('#presetCancel')
        value('#speakerSelect','W02');until('d.body.dataset.engineState==="ready"&&d.querySelector("#speakerSelect")?.value==="W02"');settled()
        value('#speakerSelect','JD2');until('d.body.dataset.engineState==="ready"&&d.querySelector("#speakerSelect")?.value==="JD2"');settled()
        assert w.vocal.invoke('library/load',{})==library
        report['checks'].append('named sets create/copy/rename/switch, R20 explicit import and worker restart restore atomic on-disk library')
        click('#tab-motion');assert local('return d.querySelector("#captureAspiration")===null')
        click('.pose-card .stop-timing > summary');value('input[aria-label="闭塞毫秒，姿势 1"]',150);value('input[aria-label="释放毫秒，姿势 1"]',8);value('input[aria-label="VOT毫秒，姿势 1"]',-100)
        until('d.querySelector(".stop-landmarks").textContent.includes("浊音起始 50 ms")')
        negative=w.vocal.invoke('keyframes/load',{});assert negative['frames'][0]['stop_timing']['aspiration']==0
        clip=w.vocal.invoke('animation/prepare',negative);assert clip['duration']==.358
        import base64,numpy as np
        raw=w.vocal.invoke('animation/audio',{'id':clip['id']});y=np.frombuffer(base64.b64decode(raw['base64']),dtype='<f4');assert len(y)==17184
        x=y[3600:6720];lag=round(raw['sample_rate']/negative['frames'][0]['f0']);assert np.corrcoef(x[:-lag],x[lag:])[0,1]>.8
        value('input[aria-label="VOT毫秒，姿势 1"]',50);positive=w.vocal.invoke('keyframes/load',{});assert positive['frames'][0]['stop_timing']['aspiration']==.042
        clip=w.vocal.invoke('animation/prepare',positive);raw=w.vocal.invoke('animation/audio',{'id':clip['id']});z=np.frombuffer(base64.b64decode(raw['base64']),dtype='<f4');assert len(z)==19200
        assert np.sqrt(np.mean(x*x))>10*np.sqrt(np.mean(z[3600:6720]**2))
        assert local('return !d.querySelector(".sequence-help").open && !d.querySelector(".stop-help").open')
        screenshot('signed-vot');report['checks'].append('signed VOT actual native prevoicing PCM, phase coupling, exact 358/400 ms totals and removed aspiration button')
        report['passed']=True
    except Exception:screenshot('failure');raise
    finally:
        write_json(out/'report.json',report);w.vocal.close();w.close();w.deleteLater();pause(500)


if __name__=='__main__':main()
