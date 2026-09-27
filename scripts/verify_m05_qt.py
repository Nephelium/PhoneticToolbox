"""M05 actual Workbench/custom-scheme Worker probe, public frames only, no devices."""
import json
import os
from pathlib import Path
import shutil
import sys
import time
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'desktop/src'), str(ROOT / 'backend/src'), str(ROOT / 'packages/phonetic_core/src')]
from PyQt6.QtCore import QEventLoop, QTimer, Qt
from PyQt6.QtWidgets import QApplication
from ptb_desktop.host import Workbench, register_scheme


def main():
    out = ROOT / 'output/validation/m05' / ('qt-' + uuid4().hex)
    dist = out / 'dist'
    dist.mkdir(parents=True)
    shutil.copytree(ROOT / 'frontend/public/m05', dist / 'm05')
    shutil.copytree(ROOT / 'output/validation/m05/inputs', dist / 'inputs')
    fixture = json.loads((dist / 'inputs/manifest.json').read_text('utf-8'))
    script = r'''
window.probeDone=false;window.probeReport={probes:[],userAgent:navigator.userAgent,secure:isSecureContext};
(async()=>{
 for(const delegate of ['CPU','GPU'])for(const mode of ['IMAGE','VIDEO'])for(const c of CASES){
  let worker;const probe={delegate,mode,case:c.name};window.probeReport.probes.push(probe);
  try{
   worker=new Worker('/m05/worker.js');let next=0;const pending=new Map();
   worker.onmessage=({data})=>{const p=pending.get(data.id);if(p){pending.delete(data.id);data.ok?p.resolve(data.value):p.reject(Error(data.error));}};
   worker.onerror=e=>{for(const p of pending.values())p.reject(Error(e.message||'worker_failed'));pending.clear();};
   const call=(op,data={},transfer=[])=>Promise.race([new Promise((resolve,reject)=>{const id=++next;pending.set(id,{resolve,reject});worker.postMessage({id,op,...data},transfer);}),new Promise((_,reject)=>setTimeout(()=>reject(Error('probe_timeout')),15000))]);
   probe.support=await call('init',{delegate,mode});probe.results=[];
   for(const frame of c.frames){const image=await createImageBitmap(await(await fetch('/inputs/'+c.name+'/'+frame.file)).blob());probe.results.push(await call('frame',{frame:image,time_ms:frame.time_s*1000,mode},[image]));}
   probe.success=true;
  }catch(e){probe.success=false;probe.error=String(e);}finally{worker?.terminate();}
 }
})().catch(e=>window.probeReport.error=String(e)).finally(()=>window.probeDone=true);
'''.replace('CASES', json.dumps(fixture['cases']))
    (dist / 'index.html').write_text('<!doctype html><meta charset="utf-8"><p>M05 Qt 公开帧测试，无摄像头或音频</p><script>'+script+'</script>', encoding='utf-8')
    register_scheme()
    app = QApplication(['M05-owned-Qt-QA'])
    app.setApplicationName('M05-owned-Qt-QA')
    window = Workbench(dist, test=True, vocal_profile=out / 'vocal-profile')
    window.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
    window.show()

    def pause(ms=100):
        loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()

    def js(code):
        box=[];loop=QEventLoop()
        window.page.runJavaScript(code,lambda value:(box.append(value),loop.quit()))
        QTimer.singleShot(5000,loop.quit);loop.exec()
        return box[0] if box else None

    started=time.monotonic()
    try:
        while time.monotonic()-started<220:
            pause()
            if js('window.probeDone===true'):break
        report=js('JSON.stringify(window.probeReport)')
        if not report:raise RuntimeError('Qt report not returned')
        report=json.loads(report)
        report['finished']=bool(js('window.probeDone===true'))
        (out / 'report.json').write_text(json.dumps(report),encoding='utf-8')
        window.grab().save(str(out / 'host.png'))
        print(json.dumps(dict(output=str(out),finished=report['finished'],probes=[{k:p.get(k) for k in ('delegate','mode','case','success','error')} for p in report['probes']]),ensure_ascii=False))
    finally:
        window.accept_close();pause(100)


if __name__ == '__main__':main()
