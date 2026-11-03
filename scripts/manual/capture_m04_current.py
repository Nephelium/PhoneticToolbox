"""Real localized LPC operations with an authorised natural WAV copy."""
from pathlib import Path
import argparse,hashlib,json,shutil,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from verification_artifacts import verification_run
from manual.capture_a_support import Capture

def run(args):
    with verification_run('m04-manual','localized-'+args.locale,'Actual Qt/native LPC; authorised natural WAV copy and previously authored demonstration TextGrid; light/maximized/DPR1.5/page100; no playback')as(out,scratch):
        c=Capture(args,out,scratch,'m04','lpc-spectrum');print(out,flush=True)
        from PyQt6.QtWidgets import QFileDialog
        config=json.loads((ROOT/'local-data/manual-authoring/sources.json').read_text(encoding='utf-8'))
        original=Path(config['single']);inputs=scratch/'inputs';inputs.mkdir();shutil.copyfile(original,inputs/'single-syllable.wav')
        grid=original.with_suffix('.TextGrid')
        if not grid.is_file():grid=ROOT/'output/manual-work/m04-capture/375685caae764e309fb6683a540f1606/inputs/single-syllable.TextGrid'
        shutil.copyfile(grid,inputs/'single-syllable.TextGrid')
        originals={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in (original,grid)}
        saved=scratch/'saved';saved.mkdir();chosen={'directory':inputs}
        QFileDialog.getExistingDirectory=lambda *a,**k:str(chosen['directory'])
        QFileDialog.getSaveFileName=lambda *a,**k:(str(saved/Path(a[2]).name),'')
        start='.lpc-transport .selection-controls label:first-of-type input';end='.lpc-transport .selection-controls label:last-of-type input'
        try:
            c.until('!!document.querySelector(".lpc-page")');c.prepare();c.shot('m04-overview-light')
            c.click('打开音频目录');c.select('.lpc-files select');c.until('!!document.querySelector(".lpc-page .wave-track svg")')
            c.shot('m04-loaded-current-light')
            c.fill(start,.1);c.fill(end,.4);c.shot('m04-selection-current-light',{'wave':' .lpc-page .wave-track','selection':'.lpc-transport .selection-controls'})
            c.shot('m04-parameters-current-light',{'parameters':'.lpc-analysis-settings'})
            c.js('document.querySelector(".view-tabs input[type=checkbox]").click()');c.until('!!document.querySelector(".spectrogram-canvas[aria-busy=false] canvas")')
            c.shot('m04-praat-current-light')
            if c.js('document.querySelectorAll(".lpc-files select")[2].options.length>1'):
                c.js('(()=>{const e=document.querySelectorAll(".lpc-files select")[2];e.value=e.options[1].value;e.dispatchEvent(new Event("change",{bubbles:true}));})()')
            c.click('放大波形');c.shot('m04-textgrid-tier-zoom-round2-light')
            c.fill(start,.1);c.fill(end,.10001);c.click('开始分析');c.until('!!document.querySelector(".lpc-page [role=alert]")');c.pause(200)
            c.shot('m04-short-selection-error-round2-light')
            c.fill(start,.1);c.fill(end,.4);c.click('开始分析');c.until('!!document.querySelector(".lpc-spectrum svg")',120)
            c.shot('m04-analysis-flow-light')
            c.click('动态纵轴');c.shot('m04-dynamic-axis-current-light')
            c.click('固定纵轴');c.fill('[aria-label='+json.dumps(c.ui.key('modules.lpc-spectrum.lpc_minimum_amplitude'),ensure_ascii=False)+']',-90);c.fill('[aria-label='+json.dumps(c.ui.key('modules.lpc-spectrum.lpc_maximum_amplitude'),ensure_ascii=False)+']',40)
            c.shot('m04-fixed-axis-edited-round2-light')
            chosen['directory']=saved;c.click('选择目录保存完整结果')
            c.pause(1000)
            assert list(saved.iterdir()),'Native LPC result files were not saved'
            c.shot('m04-complete-save-current-light')
            c.click('波形');c.fill(start,.2);c.fill(end,.35);c.click('开始分析');c.until('document.querySelectorAll(".history-links button").length>=2',120)
            c.shot('m04-history-fragment-round2-light',{'history':'.lpc-history'})
            assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in originals.items())
            c.report['originals']=[{'path':p,'sha256':sha}for p,sha in originals.items()];c.report['checks'].append('Natural waveform and Praat preview; invalid short selection; two real native LPC analyses; fixed/dynamic axes; complete native result save; originals unchanged')
            c.report['success']=True
        finally:c.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--locale',choices=('en','ja'),required=True);p.add_argument('--frontend-dist',type=Path,required=True);run(p.parse_args())
