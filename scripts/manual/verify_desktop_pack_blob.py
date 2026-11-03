"""Read a final ZIP with the native manager and decode it at the real Qt origin.

This isolates the media URL filter; it does not replace final EXE acceptance.
"""
from __future__ import annotations
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from verification_artifacts import verification_run
from workbench_source import bind_sources


def main():
    bind_sources();os.environ['QT_QPA_PLATFORM']='windows'
    from PyQt6 import sip
    from PyQt6.QtCore import QByteArray,QBuffer,QIODevice,QUrl,Qt,QEventLoop,QTimer
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtWebEngineCore import QWebEnginePage,QWebEngineProfile,QWebEngineUrlSchemeHandler
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    from ptb_desktop.web_assets import register_scheme
    from ptb_desktop.manual_packs import ManualPackManager
    manifest=ROOT/'output/manual-packs/3.1.2-manual-20261030-2c2f8e7a/candidate-manifest.json'
    candidate=next(item for item in json.loads(manifest.read_text('utf8'))['packs']if item['locale']=='en')
    archive=manifest.parent/candidate['file']
    source=ROOT/'frontend/src/manual/content.ts'
    compile_js="const fs=require('node:fs'),ts=require('./frontend/node_modules/typescript');const text=fs.readFileSync(process.argv[1],'utf8'),start=text.indexOf('export function safeMediaSource('),end=text.indexOf('\\n}',start)+2;if(start<0||end<start)throw Error('Missing production filter');process.stdout.write(ts.transpileModule(text.slice(start,end),{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.ESNext}}).outputText);"
    function=subprocess.check_output(['node','-e',compile_js,str(source)],cwd=ROOT,text=True,encoding='utf8')
    source_sha=hashlib.sha256(source.read_bytes()).hexdigest()
    with verification_run('manual-language-packs','desktop-blob','Native final ZIP import/readAsset payload, production safeMediaSource transpilation, actual ptbapp Qt origin, direct versus filtered image decode; isolated profile, no final EXE claim')as(out,scratch):
        manager=ManualPackManager(scratch/'manual-packs');register_scheme();app=QApplication(['Owned desktop manual Blob probe'])
        profile=QWebEngineProfile('owned-manual-blob',app);profile.setPersistentStoragePath(str(scratch/'qt-profile'));profile.setCachePath(str(scratch/'qt-cache'))
        view=QWebEngineView();page=QWebEnginePage(profile,view);view.setPage(page);view.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen);view.resize(1200,800)
        report={'success':False,'scope':'Minimal real native store / Qt origin / production media filter; not final EXE','filterSha256':source_sha,'archiveSha256':candidate['sha256']}
        try:
            spec={key:candidate[key]for key in ('locale','version','sourceDigest','sha256')}
            request=manager.invoke('importBegin',{'spec':spec,'bytes':archive.stat().st_size})
            with archive.open('rb')as stream:
                offset=0
                while block:=stream.read(1048576):
                    manager.invoke('importChunk',{'id':request['id'],'offset':offset,'base64':base64.b64encode(block).decode('ascii')});offset+=len(block)
            metadata=manager.invoke('importFinish',{'id':request['id']})
            chapter=manager.read_json('en','chapters/getting-started.json')
            def first_image(node):
                if node['type']=='image':return node['attrs']['assetId']
                for child in node.get('content',[]):
                    result=first_image(child)
                    if result:return result
            identity=first_image(chapter['body']);path=metadata['assetOverrides'][identity]
            payload=manager.invoke('readAsset',{'locale':'en','path':path})
            data=base64.b64decode(payload['base64'],validate=True)
            import zipfile
            with zipfile.ZipFile(archive)as zipped:
                expected=json.loads(zipped.read('language-pack.json'))['hashes'][path]
            assert hashlib.sha256(data).hexdigest()==expected
            report['asset']={'id':identity,'path':path,'mime':payload['mime'],'bytes':len(data),'sha256':expected}
            html='''<!doctype html><html><head><meta charset="utf-8"><style>img{max-width:46%;vertical-align:top}</style></head><body><h1>Owned Qt manual Blob probe</h1><script type="module">'''+function+'''
 const bytes=Uint8Array.from(atob('''+json.dumps(payload['base64'])+'''),c=>c.charCodeAt(0));
 const raw=URL.createObjectURL(new Blob([bytes],{type:'''+json.dumps(payload['mime'])+'''}));
 window.probe={origin:location.origin,href:location.href,raw,filtered:safeMediaSource(raw),done:false};
 async function decode(url,id){const img=new Image();img.id=id;document.body.append(img);await new Promise((resolve,reject)=>{img.onload=resolve;img.onerror=()=>reject(Error('Image failed: '+id));img.src=url;});return {width:img.naturalWidth,height:img.naturalHeight};}
 try{window.probe.direct=await decode(raw,'direct');if(window.probe.filtered)window.probe.filteredImage=await decode(window.probe.filtered,'filtered');}
 catch(e){window.probe.error=String(e);}finally{window.probe.done=true;}
 </script></body></html>'''
            class Handler(QWebEngineUrlSchemeHandler):
                def requestStarted(self,job):
                    buffer=QBuffer(job);buffer.setData(QByteArray(html.encode('utf8')));buffer.open(QIODevice.OpenModeFlag.ReadOnly);job.reply(b'text/html',buffer)
            handler=Handler(profile);profile.installUrlSchemeHandler(b'ptbapp',handler)
            def pause(ms=40):
                loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
            def js(expression):
                values=[];loop=QEventLoop();page.runJavaScript(expression,lambda value:(values.append(value),loop.quit()));QTimer.singleShot(2000,loop.quit);loop.exec();return values[0]if values else None
            view.show();page.load(QUrl('ptbapp://app/manual-blob-probe.html'))
            deadline=time.monotonic()+30
            while time.monotonic()<deadline:
                result=js('window.probe??null')
                if result and result.get('done'):break
                pause()
            else:raise AssertionError('Qt Blob probe timeout')
            report['qt']=result
            assert result['origin']=='ptbapp://app' and result['direct']['width']>0 and not result.get('error')
            pause(250);view.grab().save(str(out/'qt-blob.png'))
            report['success']=bool(result.get('filteredImage'))
            assert hashlib.sha256(source.read_bytes()).hexdigest()==source_sha
        finally:
            (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
            manager.close();view.close();sip.delete(view);app.processEvents();sip.delete(profile);app.processEvents();sip.delete(app)
            print(json.dumps({'success':report['success'],'report':str(out/'report.json')},ensure_ascii=False),flush=True)
        if '--require-display'in sys.argv:assert report['success'],'Production media filter rejected the real desktop Blob URL'


if __name__=='__main__':main()
