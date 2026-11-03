"""Localised real Klatt controls, extraction, generation and owned native saves."""
from pathlib import Path
import argparse,base64,hashlib,json,shutil,sys,time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from verification_artifacts import verification_run
from manual.capture_a_support import Capture
from manual.capture_locale import CaptureLocale

def run(args):
    with verification_run('m06-manual','localized-'+args.locale,'Actual Qt and native extraction/Klatt synthesis; authorised WAV copy; own parameters and exports; light/maximized/DPR1.5/page100; no playback')as(out,scratch):
        c=Capture(args,out,scratch,'m06','speech-synthesis');print(out,flush=True)
        from PyQt6.QtWidgets import QFileDialog
        from PyQt6.QtCore import QPoint,Qt
        from PyQt6.QtTest import QTest
        config=json.loads((ROOT/'local-data/manual-authoring/sources.json').read_text(encoding='utf-8'));original=Path(config['single']);inputs=scratch/'inputs';inputs.mkdir();shutil.copyfile(original,inputs/'single-syllable.wav');original_sha=hashlib.sha256(original.read_bytes()).hexdigest()
        saved=scratch/'saved';saved.mkdir();choice={'directory':inputs}
        QFileDialog.getExistingDirectory=lambda *a,**k:str(choice['directory'])
        QFileDialog.getSaveFileName=lambda *a,**k:(str(saved/Path(a[2]).name),'')
        def aria(key):return '[aria-label='+json.dumps(c.ui.key('modules.speech-synthesis.'+key),ensure_ascii=False)+']'
        def parameter(key):
            assert c.js('(()=>{const e=[...document.querySelectorAll(".parameter-list button")].find(e=>e.textContent.trim().split(/\\s/)[0]==='+json.dumps(key)+');if(!e||e.disabled)return false;e.click();return true})()'),key
            c.pause()
        def confirm():
            if c.js('!!document.querySelector("dialog[open]")'):c.click('应用并覆盖')
        def finished():
            c.until('document.querySelector(".m06-page").getAttribute("aria-busy")==="false"',120)
            assert not c.js('!!document.querySelector(".m06-page .module-status.error")'),c.js('document.querySelector(".m06-page").innerText.slice(-1000)')
        def phonation(value):
            c.fill(aria('phonation_type_preset'),value,'change');c.click('应用预设');confirm()
        def generate(sequence):
            c.fill(aria('ipa_vowel_sequence'),sequence);c.click('生成元音');finished()
        def synthesize():c.click('合成音频');finished();c.until('!!document.querySelector(".m06-page .wave-track svg")')
        def override(value):c.fill(aria('curve_override'),value);c.click('应用覆盖')
        def duration(value):c.fill(aria('total_duration'),value);c.click('应用时长')
        def fade(start,end):
            c.fill('.base-settings .controls>label:nth-of-type(2) input',start);c.fill('.base-settings .controls>label:nth-of-type(3) input',end)
        def export():
            exports=c.report.setdefault('exports',[]);destination=scratch/('export-'+str(len(exports)));destination.mkdir();choice['directory']=destination;c.click('导出音频')
            deadline=time.monotonic()+40
            while not(destination/'m06.ptb.json').is_file():
                assert time.monotonic()<deadline,'Native synthesis export did not finish';c.pause()
            metadata=json.loads((destination/'m06.ptb.json').read_text(encoding='utf-8'))
            exports.append({'config':metadata['config'],'seed':metadata['seed'],'computation_revision':metadata['computation_revision'],'wav_sha256':hashlib.sha256((destination/'synthesis.wav').read_bytes()).hexdigest()})
            return metadata
        try:
            c.until('!!document.querySelector(".m06-page")');c.prepare();c.click('打开音频目录');c.select('.source-picker');c.until('!!document.querySelector(".m06-page .wave-track svg")')
            c.shot('m06-loaded-current-light')
            c.click('提取参数');finished();c.shot('m06-extract-flow-light')
            c.click('语谱图');c.until('!!document.querySelector(".spectrum-plot")');c.shot('m06-extract-spectrum-current-light');c.click('波形')
            c.click('导出参数');c.until('!!document.querySelector(".m06-page")');c.pause(400)
            csv=next(saved.glob('*.csv'));raw=csv.read_bytes()
            c.fill(aria('phonation_type_preset'),'气声','change');c.click('应用预设')
            c.until('!!document.querySelector("dialog[open]")');c.shot('m06-preset-confirm-current-light',{'dialog':'dialog[open]'})
            original_message=c.js('document.querySelector("dialog[open]").innerText')
            other='ja'if args.locale=='en'else'en';CaptureLocale(other,('speech-synthesis','workbench','components')).prepare_qt(c.js,c.until)
            assert c.js('document.querySelector("dialog[open]").innerText')!=original_message
            c.ui.prepare_qt(c.js,c.until);c.click('应用并覆盖');parameter('AH');c.shot('m06-preset-ah-current-light')
            parameter('F0');duration(2);c.click('合成结果')
            for text,name in [('180','m06-cover-constant-current-light'),('120,250,100','m06-cover-linear-current-light'),('120;250;100','m06-cover-steps-current-light'),('120,240;180,100','m06-cover-combined-current-light')]:
                override(text);c.shot(name,{'override':'.curve-controls','curve':'.curve-editor'})
            c.click('清除覆盖')
            # Draw through the real Qt widget with Shift, then restore with Ctrl.
            box=c.js('(()=>{const r=document.querySelector(".curve-editor svg").getBoundingClientRect();return [r.x,r.y,r.width,r.height]})()');target=c.window.view.focusProxy()or c.window.view
            a=QPoint(round(box[0]+box[2]*.3),round(box[1]+box[3]*.6));b=QPoint(round(box[0]+box[2]*.65),round(box[1]+box[3]*.3))
            QTest.mousePress(target,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.ShiftModifier,a);QTest.mouseMove(target,b,150);QTest.mouseRelease(target,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.ShiftModifier,b);c.pause(250);c.shot('m06-curve-drawn-current-light')
            QTest.mousePress(target,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.ControlModifier,a);QTest.mouseMove(target,b,150);QTest.mouseRelease(target,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.ControlModifier,b);c.pause(250);c.shot('m06-curve-reset-current-light')
            synthesize();c.shot('m06-klatt-flow-r3-light')
            payload=base64.b64encode(raw).decode()
            c.js('(()=>{const e=document.querySelector("input[accept=\\".csv,.json\\"]"),d=new DataTransfer();d.items.add(new File([Uint8Array.from(atob('+json.dumps(payload)+'),c=>c.charCodeAt(0))],"parameters.csv"));e.files=d.files;e.dispatchEvent(new Event("change",{bubbles:true}));})()')
            c.until('!!document.querySelector("dialog[open]")');c.click('应用并覆盖');c.shot('m06-parameter-import-current-light')
            c.click('元音规则');c.until('!!document.querySelector(".vowel-rules")');c.shot('m06-vowel-rules-current-light',{'rules':'.vowel-rules'})
            c.js('const body=document.querySelector("dialog[open] .dialog-body");body.scrollTop=body.scrollHeight;');c.shot('m06-vowel-rules-bottom-current-light')
            c.js('document.querySelector("dialog[open] .dialog-header .icon-button").click()')
            duration(.8);parameter('F0');override('150');phonation('常态浊声');generate('a');synthesize();metadata=export()
            assert metadata['config']['duration']==.8 and metadata['config']['curves']['F0']['override']==150
            c.shot('m06-a-modal-150-light')
            phonation('气声');synthesize();metadata=export();assert metadata['config']['curves']['AH']['override']==35;c.shot('m06-a-breathy-150-light')
            phonation('常态浊声');duration(2);fade(50,100);generate('a-i/- ///u+/e++o/-');synthesize();parameter('F1');c.shot('m06-vowel-syntax-complex-light')
            duration(1);generate('a a');parameter('F0')
            for start,end,name in [(100,100,'m06-fade-100-100-light'),(30,80,'m06-fade-30-80-light')]:
                fade(start,end);synthesize();c.shot(name)
            generate('a i u');c.shot('m06-vowel-generated-current-light');synthesize();export();c.shot('m06-result-saved-current-light')
            assert hashlib.sha256(original.read_bytes()).hexdigest()==original_sha
            c.report['originals']=[{'path':str(original),'sha256':original_sha}];c.report['checks'].append('Actual extraction and Klatt synthesis; CSV roundtrip; open confirmation switches languages; real Shift/Ctrl drawing; 150Hz modal/breathy and complex/fade generation; native save; no playback')
            c.report['success']=True
        finally:c.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--locale',choices=('en','ja'),required=True);p.add_argument('--frontend-dist',type=Path,required=True);run(p.parse_args())
