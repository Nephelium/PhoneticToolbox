"""Real localised reconstruction/drawing scenes from existing authorised copies."""
from pathlib import Path
import argparse,hashlib,json,shutil,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from verification_artifacts import verification_run
from manual.capture_a_support import Capture

def run(args):
    with verification_run('m09-manual','localized-'+args.locale,'Actual Qt/native Griffin-Lim and original-phase drawing; existing authorised image and natural WAV copies; light/maximized/DPR1.5/page100; no playback')as(out,scratch):
        c=Capture(args,out,scratch,'m09','spectrogram-to-audio');print(out,flush=True)
        from PyQt6.QtWidgets import QFileDialog
        from PyQt6.QtCore import QPoint,Qt
        from PyQt6.QtTest import QTest
        prior=ROOT/'output/manual-work/m09-capture/0e715ffc1d994ab1bc46101de3e0b313';evidence=json.loads((prior/'capture-report.json').read_text(encoding='utf-8'))
        examples={r['state']['example']['slug']:r['state']['example']for r in evidence['captures']if r.get('state',{}).get('example')}
        images=scratch/'images';images.mkdir();originals={}
        for example in examples.values():
            source=prior/'inputs'/example['name'];assert hashlib.sha256(source.read_bytes()).hexdigest()==example['sourceSha256'];originals[str(source)]=example['sourceSha256'];shutil.copyfile(source,images/example['name'])
        audio=scratch/'audio';audio.mkdir();natural=Path(json.loads((ROOT/'local-data/manual-authoring/sources.json').read_text(encoding='utf-8'))['single']);originals[str(natural)]=hashlib.sha256(natural.read_bytes()).hexdigest();shutil.copyfile(natural,audio/'single-syllable.wav')
        saved=scratch/'saved';saved.mkdir();choice={'directory':images};QFileDialog.getExistingDirectory=lambda *a,**k:str(choice['directory']);QFileDialog.getSaveFileName=lambda *a,**k:(str(saved/Path(a[2]).name),'')
        def aria(key):return '[aria-label='+json.dumps(c.ui.key('modules.spectrogram-to-audio.'+key),ensure_ascii=False)+']'
        def field(key,value):
            label=c.ui.key('modules.spectrogram-to-audio.'+key)
            assert c.js('(()=>{const e=[...document.querySelectorAll(".m09-page label")].find(e=>e.textContent.trim().startsWith('+json.dumps(label)+'))?.querySelector("input");if(!e)return false;e.value='+json.dumps(str(value))+';e.dispatchEvent(new Event("input",{bubbles:true}));e.dispatchEvent(new Event("change",{bubbles:true}));return true})()'),label
            c.pause()
        def select_name(name):
            selector=aria('spectrogram_image');c.until('!!document.querySelector('+json.dumps(selector)+')')
            assert c.js('(()=>{const e=document.querySelector('+json.dumps(selector)+'),o=[...e.options].find(o=>o.textContent==='+json.dumps(name)+');if(!o)return false;e.value=o.value;e.dispatchEvent(new Event("change",{bubbles:true}));return true})()')
            c.until('document.querySelector(".image-frame img")?.naturalWidth>0')
        def corners(example):
            c.click('选择四个角点');w,h=example['dimensions']
            for x,y in example['corners']:
                c.js('(()=>{const e=document.querySelector(".image-frame svg"),r=e.getBoundingClientRect();e.dispatchEvent(new MouseEvent("click",{clientX:r.x+r.width*'+str(x/w)+',clientY:r.y+r.height*'+str(y/h)+',bubbles:true}));})()');c.pause(100)
            c.until('document.querySelectorAll(".image-frame circle").length===4')
        def result():
            c.until('!!document.querySelector(".comparison img")&&document.querySelector(".m09-page").getAttribute("aria-busy")==="false"',180)
            c.until('[...document.querySelectorAll(".comparison img")].every(e=>e.complete&&e.naturalWidth>0)')
            stale=c.ui.key('modules.spectrogram-to-audio.calibration_corners_or_strokes_changed_this_view_still_shows')
            assert not c.js('[...document.querySelectorAll(".m09-card .notice")].some(e=>e.textContent.trim()==='+json.dumps(stale)+')'),'Equivalent native result snapshot was marked stale'
            metadata=c.js('JSON.parse(document.querySelector(".comparison").parentElement.querySelector("pre").textContent)');c.report.setdefault('resultMetadata',[]).append(metadata)
            if args.drawing_only:assert len(metadata['strokes'])==2 and [s['color']for s in metadata['strokes']]==[0,255],'Expected real black and white strokes'
        def stroke(white=False):
            c.click('白色画笔'if white else'黑色画笔')
            c.js('document.querySelector(".spectral-editor canvas").scrollIntoView({block:"center"})');c.pause(180)
            box=c.js('(()=>{const r=document.querySelector(".spectral-editor canvas").getBoundingClientRect();return [r.x,r.y,r.width,r.height]})()');target=c.window.view.focusProxy()or c.window.view
            a=QPoint(round(box[0]+box[2]*(.3 if white else .2)),round(box[1]+box[3]*(.3 if white else .65)));b=QPoint(round(box[0]+box[2]*(.75 if white else .65)),round(box[1]+box[3]*(.65 if white else .3)))
            QTest.mousePress(target,Qt.MouseButton.LeftButton,pos=a);QTest.mouseMove(target,b,150);QTest.mouseRelease(target,Qt.MouseButton.LeftButton,pos=b);c.pause(200)
        try:
            c.until('!!document.querySelector(".m09-page")');c.prepare();c.click('选择图片目录')
            for slug,example in ([] if args.drawing_only else examples.items()):
                select_name(example['name']);field('duration_s',example['duration']);field('end_frequency_hz',example['fmax']);field('minimum_db_white',example['db']);field('maximum_db_black',0);field('window_length_ms',example['window']);corners(example)
                c.shot('m09-v2-'+slug+'-calibration-light',{'corners':'.image-frame','parameters':'.m09-settings'})
                c.click('开始重建');result();c.shot('m09-v2-'+slug+'-result-light')
            if not args.audio_only:
                c.click('图片涂鸦重建');select_name(examples['steve-jobs']['name']);field('duration_s',examples['steve-jobs']['duration']);field('end_frequency_hz',examples['steve-jobs']['fmax']);field('window_length_ms',examples['steve-jobs']['window']);corners(examples['steve-jobs']);c.click('应用校正并绘图');c.until('!!document.querySelector(".spectral-editor canvas")');stroke();stroke(True);c.shot('m09-image-paint-current-light',{'editor':'.spectral-editor'})
                c.click('撤销');c.shot('m09-image-undo-current-light');c.click('重做');c.click('生成涂鸦音频');result();c.js('document.querySelector(".comparison").parentElement.scrollIntoView({block:"start"})');c.pause(200);c.shot('m09-image-drawn-result-current-light')
            c.click('音频藏信息');choice['directory']=audio;c.click('打开音频目录');c.select(aria('audio_for_hidden_information'));c.until('!!document.querySelector(".spectral-editor canvas")',120)
            c.shot('m09-canvas-flow-r3-light');stroke();stroke(True);c.shot('m09-audio-paint-current-light',{'editor':'.spectral-editor'})
            c.js('document.querySelector("#tab-M09").closest(".tab-wrap").querySelector(".tab-close").click()');c.pause(200);c.shot('m09-close-protection-current-light',{'dialog':'dialog[open]'})
            if c.js('!!document.querySelector("dialog[open]")'):c.js('document.querySelector("dialog[open] .dialog-header .icon-button").click()')
            c.click('生成藏信息音频');result();c.js('document.querySelector(".comparison").parentElement.scrollIntoView({block:"start"})');c.pause(200);c.shot('m09-audio-result-current-light');choice['directory']=saved;c.click('保存重建结果');c.pause(800);assert list(saved.iterdir());c.shot('m09-audio-save-current-light')
            assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in originals.items())
            c.report['originals']=[{'path':p,'sha256':sha}for p,sha in originals.items()];c.report['checks'].append('Four authorised source-image calibrations and real native reconstructions; real Qt brush/undo; image drawing and original-phase audio drawing; close protection and native save; originals unchanged')
            c.report['success']=True
        finally:c.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--locale',choices=('en','ja'),required=True);p.add_argument('--frontend-dist',type=Path,required=True);p.add_argument('--drawing-only',action='store_true');p.add_argument('--audio-only',action='store_true');run(p.parse_args())
