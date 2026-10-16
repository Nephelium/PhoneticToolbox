"""Capture an actual synthesized monosyllable example for the landing page."""
from capture_website_modules import Capture, ROOT, verification_run, Path, shutil, json
import hashlib

def main():
    with verification_run('website','pitch-example','User-selected monosyllable directory; F0 +15 Hz and actual synthesis') as (out,scratch):
        folder=scratch/'单音节示例';folder.mkdir()
        src=Path(r'C:\Users\13680\Desktop\project\音频数据\test\dl_m01_0558_ma.wav')
        shutil.copyfile(src,folder/'单音节_ma.wav')
        c=Capture(out,scratch)
        c.report['input_sha256']=hashlib.sha256(src.read_bytes()).hexdigest()
        try:
            c.until("!!document.querySelector('.home-page')")
            c.palette('one','light');c.module('变速变调');c.folder=folder;c.click('打开音频目录');c.select('变速变调音频')
            c.until('document.querySelector("svg.m08-curve .original")?.getAttribute("d").includes("L")',180)
            c.until('document.querySelector(".m08-page")?.getAttribute("aria-busy")==="false"',180)
            # Read the plotted original F0 coordinates, then use the normal import UI.
            values=c.js("(()=>{const s=document.querySelector('svg.m08-curve'),h=s.viewBox.baseVal.height,d=s.querySelector('.original').getAttribute('d');if((d.match(/M/g)||[]).length!==1)throw Error('Expected one continuous voiced segment');return [...d.matchAll(/[ML]([\\d.e+-]+),([\\d.e+-]+)/g)].map(m=>50+((h-40)-Number(m[2]))/(h-60)*300+15)})()")
            assert len(values)>3
            c.click('导入基频序列')
            c.js('(()=>{const e=document.querySelector("textarea[aria-label=基频序列]");e.value='+json.dumps('\n'.join(str(v) for v in values))+';e.dispatchEvent(new Event("input",{bubbles:true}))})()')
            c.click('导入序列');c.until('!document.querySelector("textarea[aria-label=基频序列]")')
            c.click('合成整段')
            c.until('[...document.querySelectorAll("button")].some(b=>b.textContent.trim()==="播放合成音"&&!b.disabled)',180)
            c.until('document.querySelector(".history-plot svg path")?.getAttribute("d").includes("L")',180)
            c.report['f0_adjustment_hz']=15;c.report['voiced_frames']=len(values)
            c.snapshot('pitch-one-light','M08','User-selected monosyllable ma; original F0 shifted +15 Hz through import UI; actual whole-file synthesis and measured output F0')
            assert not c.report['errors'];c.report['success']=True
            p=ROOT/'temporary-site/assets/screens/provenance.json';data=json.loads(p.read_text('utf-8'));data['captures']=[r for r in data['captures'] if r['file']!='pitch-one-light.png']+c.report['captures'];p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n','utf-8')
        except Exception as e:
            c.report['failure']=str(e);c.w.view.grab().save(str(out/'failure.png'));raise
        finally:c.close();print(json.dumps({'out':str(out),'success':c.report['success']}),flush=True)

if __name__=='__main__':main()
