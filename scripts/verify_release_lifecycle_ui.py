"""Owned real UI regressions, without product changes or production services."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from verification_artifacts import verification_run, write_json
from workbench_source import ROOT, bind_sources


def fixtures(scratch, *, synthesis=True):
    from ptb_api.preferences import FIELDS
    from pypdf import PdfWriter
    from pypdf.generic import NameObject, DictionaryObject, DecodedStreamObject
    roots = [scratch / 'library-a', scratch / 'library-b']
    for root, label, pages in zip(roots, ['A original', 'B different'], [2, 1]):
        root.mkdir()
        writer = PdfWriter()
        for number in range(1, pages + 1):
            page = writer.add_blank_page(width=600, height=800)
            font = DictionaryObject({NameObject('/Type'): NameObject('/Font'), NameObject('/Subtype'): NameObject('/Type1'), NameObject('/BaseFont'): NameObject('/Helvetica')})
            page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): writer._add_object(font)})})
            stream = DecodedStreamObject(); stream.set_data(f'BT /F1 18 Tf 50 700 Td (Neutral library {label} page {number}) Tj ET'.encode())
            page[NameObject('/Contents')] = writer._add_object(stream)
        with (root / 'same.pdf').open('wb') as target: writer.write(target)
    result = dict(allowed_fields=FIELDS,pdfs=[dict(root=str(root), name='same.pdf', bytes=(root/'same.pdf').stat().st_size,
                            sha256=hashlib.sha256((root/'same.pdf').read_bytes()).hexdigest()) for root in roots])
    if synthesis:
        from phonetic_core.synthesis.klatt.api import defaults, export_parameters
        from ptb_worker.m06_child import compute
        config = defaults(); config['duration'] = .2; config['sequence'] = 'a'
        for curve in config['curves'].values(): curve['points'][-1][0] = .2
        parameters = export_parameters(config)
        outputs = {}
        for action in ('generate', 'synthesize'):
            items = dict(compute(dict(parameters=parameters, action=action, seed=23, input_sha256=hashlib.sha256(b'').hexdigest()), b''))
            outputs[action] = dict(files=[dict(name=name, data=base64.b64encode(raw).decode()) for name, raw in items.items()])
            if action == 'synthesize': result['wav_sha256'] = hashlib.sha256(items['synthesis.wav']).hexdigest()
        result.update(parameters=parameters, outputs=outputs)
    write_json(scratch / 'fixtures.json', result)
    return result


def verify_m18(window, out, scratch, report):
    """Source Qt test window only; root choices and PDFs belong to this scratch."""
    from PyQt6.QtCore import QEventLoop, QTimer, QPoint, Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QFileDialog
    from ptb_documents.papers import PaperService
    from pypdf import PdfReader
    assert window.profile.isOffTheRecord(), 'Use the existing test=True source Qt launcher'
    class NoNetwork:
        def open(self, *args, **kwargs): raise AssertionError('Owned M18 fixture cannot request the network')
    service = PaperService(scratch / 'paper-preferences', opener=NoNetwork())
    service.set_auto_receive(False)
    roots = [scratch / 'library-a', scratch / 'library-b']
    original_b = (roots[1] / 'same.pdf').read_bytes()
    service.set_library_root(roots[0]); document = service.library_status()['documents'][0]
    note = dict(id='owned-note', page=1, kind='note', color='yellow', text='Owned original annotation', quote='', rects=[[.2,.2,.025,.025]])
    service.save_annotation(document['id'], 'original', 0, note, False)
    bridge = window.papers_bridge; previous = bridge.service; bridge.service = service
    chosen = [str(roots[1])]; exported = scratch / 'annotated.pdf'
    directory_picker, save_picker = QFileDialog.getExistingDirectory, QFileDialog.getSaveFileName
    QFileDialog.getExistingDirectory = lambda *a, **k: chosen[0]
    QFileDialog.getSaveFileName = lambda *a, **k: (str(exported), 'PDF')
    def pause(ms=80):
        loop=QEventLoop(); QTimer.singleShot(ms,loop.quit); loop.exec()
    def js(code):
        loop=QEventLoop(); values=[]
        window.page.runJavaScript(code,lambda value:(values.append(value),loop.quit()))
        QTimer.singleShot(6000,loop.quit);loop.exec();assert values,'JavaScript timeout';return values[0]
    def until(code, seconds=30):
        deadline=time.monotonic()+seconds
        while time.monotonic()<deadline:
            if js('Boolean('+code+')'):return
            pause()
        raise AssertionError(code+'\n'+str(js('document.body.innerText.slice(-2500)')))
    def click(label):
        lookup='[...document.querySelectorAll("button")].find(e=>e.offsetParent&&!e.disabled&&(e.textContent.trim()==='+json.dumps(label)+'||e.title==='+json.dumps(label)+'||e.getAttribute("aria-label")==='+json.dumps(label)+'))'
        until(lookup)
        point=js('(()=>{const e='+lookup+';e.scrollIntoView({block:"nearest"});const r=e.getBoundingClientRect();return [r.x+r.width/2,r.y+r.height/2]})()')
        QTest.mouseClick(window.view.focusProxy() or window.view,Qt.MouseButton.LeftButton,pos=QPoint(round(point[0]),round(point[1])));pause()
    def text_is(label, pages):
        until('document.querySelector(".pdf-reader")&&[...document.querySelectorAll(".pdf-text-line")].some(e=>e.textContent.includes('+json.dumps(label)+'))&&document.querySelector("[aria-label=页码]")?.max==='+json.dumps(str(pages)))
    try:
        until('document.querySelector("#tool-sidebar")');click('语音学论文精读');click('我的 PDF 1');text_is('A original',2)
        until('[...document.querySelectorAll(".pdf-toolbar button")].some(e=>e.textContent==="批注 1")')
        js('window.__ownedReader=document.querySelector(".pdf-reader");window.__ownedImage=document.querySelector(".pdf-sheet img")?.src;true')
        click('下一页');until('document.querySelector("[aria-label=页码]").value==="2"')
        click('更换文件夹');text_is('B different',1)
        assert js('window.__ownedReader!==document.querySelector(".pdf-reader")')
        assert js('[...document.querySelectorAll(".pdf-toolbar button")].some(e=>e.textContent==="批注 0")')
        window.view.grab().save(str(out/'m18-root-b.png'))
        report['checks'].append('Real Qt native folder picker switches same relative PDF/id from A (2 pages, one note) to B (1 page, no old note); reader remounts')
        js('window.__ownedReader=document.querySelector(".pdf-reader");true')
        (roots[1]/'same.pdf').write_bytes((roots[0]/'same.pdf').read_bytes())
        click('刷新');text_is('A original',2);assert js('window.__ownedReader!==document.querySelector(".pdf-reader")')
        until('[...document.querySelectorAll(".pdf-toolbar button")].some(e=>e.textContent==="批注 1")')
        report['checks'].append('External content replacement plus real Refresh changes visible page/version; same SHA restores its original annotation')
        (roots[1]/'same.pdf').write_bytes(original_b);click('刷新');text_is('B different',1)
        chosen[0]=str(roots[0]);click('更换文件夹');text_is('A original',2)
        until('[...document.querySelectorAll(".pdf-toolbar button")].some(e=>e.textContent==="批注 1")')
        click('导出 PDF');js('document.querySelector(".export-check input").click()');click('选择位置并导出');until('!document.querySelector(".export-check")')
        result=PdfReader(exported);annotations=[a.get_object() for a in result.pages[0]['/Annots']]
        assert any(a.get('/Contents')==note['text']+'\n\n'+note['quote'] for a in annotations)
        assert len(result.pages)==2 and 'A original' in result.pages[0].extract_text()
        assert service.annotations(document['id'],'original')['items'][0]['text']==note['text']
        window.view.grab().save(str(out/'m18-restored-a.png'))
        report['checks'].append('Returning original library retains its original note; actual annotated PDF export contains current A pages and original note')
        report['export']=dict(bytes=exported.stat().st_size,sha256=hashlib.sha256(exported.read_bytes()).hexdigest())
    finally:
        QFileDialog.getExistingDirectory=directory_picker;QFileDialog.getSaveFileName=save_picker
        bridge.pool.submit(lambda:None).result(timeout=30)
        service.close_documents()
        bridge.service=previous


def qt(dist, out, scratch, identity):
    from PyQt6.QtCore import QCoreApplication,QEvent,QEventLoop,QTimer
    from PyQt6.QtWidgets import QApplication
    from ptb_desktop.host import Workbench,register_scheme
    register_scheme();app=QApplication(['owned-release-lifecycle'])
    window=Workbench(dist,test=True,vocal_profile=scratch/'vocal-profile')
    window.resize(1440,1000);window.show()
    report=dict(success=False,checks=[],identity=identity,scope='Source Qt host with frozen source/frontend identity; owned PDFs and preferences; separate from final EXE')
    try:
        verify_m18(window,out,scratch,report);report['success']=True
    except Exception as exc:
        report['failure']=repr(exc);window.view.grab().save(str(out/'qt-failure.png'));raise
    finally:
        window.closing=True;window.close();window.papers_bridge.pool.shutdown(wait=True);window.papers_bridge.network_pool.shutdown(wait=True)
        if window.service.process:window.service.process.wait(timeout=20)
        window.page.deleteLater();window.deleteLater();QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete)
        loop=QEventLoop();QTimer.singleShot(300,loop.quit);loop.exec();write_json(out/'qt-report.json',report)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frontend-dist',type=Path)
    parser.add_argument('--source-root',type=Path,default=ROOT)
    parser.add_argument('--build',action='store_true')
    parser.add_argument('--web-only',action='store_true');parser.add_argument('--qt-only',action='store_true')
    parser.add_argument('--m06-real-only',action='store_true');parser.add_argument('--science-dir',type=Path)
    args=parser.parse_args();source=args.source_root.resolve();bind_sources(source)
    node=Path('C:/Users/13680/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe')
    with verification_run('release-lifecycle-ui','acceptance','Real compiled AppShell/ServerPage with owned identity/API and fault fixture; native-generated M06 bytes; real Qt local PDF libraries, no user library or production DB') as (out,scratch):
        print(out,flush=True);fixtures(scratch,synthesis=not args.qt_only)
        dist=args.frontend_dist.resolve() if args.frontend_dist else source/'frontend/dist'
        if args.build:
            dist=scratch/'frontend-dist'
            with (out/'build.log').open('w',encoding='utf8') as log:subprocess.run([str(node),'node_modules/vite/bin/vite.js','build','--outDir',str(dist)],cwd=source/'frontend',stdout=log,stderr=subprocess.STDOUT,check=True)
        identity={}
        frozen_inputs=ROOT/'output/packaging'/source.parent.name.removeprefix('build-')/'frontend-inputs.json'
        if source!=ROOT:
            recorded=json.loads(frozen_inputs.read_text(encoding='utf8'))
            changed=[name for name,item in recorded.items() if name.startswith('frontend/src/') and hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=item['sha256']]
            assert not changed,'Current frontend source differs from frozen inputs: '+str(changed)
            identity['frozen_frontend_inputs']=dict(path=str(frozen_inputs),sha256=hashlib.sha256(frozen_inputs.read_bytes()).hexdigest(),source_files_matched=sum(name.startswith('frontend/src/') for name in recorded))
        for name in ['frontend/dist/index.html','frontend/src/app/AppShell.vue','frontend/src/modules/speech-synthesis/SpeechSynthesisPage.vue','frontend/src/modules/perception/PerceptionPage.vue','frontend/src/modules/paper-reading/PaperReadingPage.vue','desktop/src/ptb_desktop/host.py','desktop/src/ptb_documents/papers.py','desktop/src/ptb_documents/paper_library.py']:
            path=dist/'index.html' if name=='frontend/dist/index.html' else (ROOT if name.startswith('frontend/src/') else source)/name
            identity[name]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        write_json(out/'identity.json',identity)
        if args.m06_real_only:
            if args.science_dir is None:parser.error('--m06-real-only requires the existing --science-dir')
            from verify_m06_r6 import oracles
            oracle=scratch/'native-oracles.json';oracles(oracle)
            config=scratch/'m06-real.json'
            runtime_lock=source/'resources/manifests/browser-science-runtime.json'
            if not runtime_lock.exists():runtime_lock=ROOT/'resources/manifests/browser-science-runtime.json'
            write_json(config,dict(out=str(out),scratch=str(scratch),dist=str(dist),science=str(args.science_dir.resolve()),overlay=str(args.science_dir.resolve()),oracles=str(oracle),frontendInputs=str(frozen_inputs) if source!=ROOT else None,runtimeLock=str(runtime_lock)))
            with (out/'m06-real.log').open('w',encoding='utf8') as log:subprocess.run([str(node),str(ROOT/'tests/e2e/m06-r6.cjs'),str(config)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        elif not args.qt_only:
            config=scratch/'browser-config.json';write_json(config,dict(out=str(out),scratch=str(scratch),dist=str(dist),fixture=str(scratch/'fixtures.json'),identity=identity))
            with (out/'browser.log').open('w',encoding='utf8') as log:subprocess.run([str(node),str(ROOT/'tests/e2e/release-lifecycle.cjs'),str(config)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        if not args.web_only and not args.m06_real_only:qt(dist,out,scratch,identity)


if __name__=='__main__':main()
