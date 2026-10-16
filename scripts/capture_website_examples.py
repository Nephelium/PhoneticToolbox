"""Refresh the two website examples from actual native application states."""
import base64
from capture_website_modules import Capture, ROOT, verification_run, Path, shutil, json

def main():
    with verification_run('website','examples','White EGG palette and completed image reconstruction') as (out,scratch):
        egg=scratch/'EGG示例';egg.mkdir();speech=scratch/'语句示例';speech.mkdir();images=scratch/'图片示例';images.mkdir()
        shutil.copyfile(Path(r'C:\Users\13680\Desktop\project\音频数据\EGG测试\3.wav'),egg/'EGG示例.wav')
        shutil.copyfile(Path(r'C:\Users\13680\Desktop\project\音频数据\线下录音\女1\原声_平静.wav'),speech/'语句示例.wav')
        c=Capture(out,scratch)
        try:
            c.until("!!document.querySelector('.home-page')")
            c.palette('github','light');c.module('EGG 信号分析');c.folder=egg;c.click('打开音频目录');c.select('EGG 音频文件')
            c.until('document.querySelector(".egg-live-status")?.textContent.includes("实时预览")',180)
            c.js("(()=>{const a=document.querySelectorAll('.egg-bottom-bar .selection-controls input');for(const [i,v] of [[1,41],[0,40]]){a[i].value=v;a[i].dispatchEvent(new Event('input',{bubbles:true}));a[i].dispatchEvent(new Event('change',{bubbles:true}))}})()")
            c.pause(1500);c.until('document.querySelector(".egg-live-status")?.textContent.includes("实时预览")',180)
            c.snapshot('egg-github-light','M03','White GitHub light palette; actual EGG example at 40–41 s')
            c.palette('ayu','dark');c.module('语谱图转音频');c.click('音频藏信息');c.folder=speech;c.click('打开音频目录');c.select('藏信息源音频')
            c.until('document.body.innerText.includes("画布已载入")',120)
            data=c.js('document.querySelector(".spectral-editor canvas").toDataURL("image/png")')
            (images/'语谱图示例.png').write_bytes(base64.b64decode(data.split(',')[1]))
            c.click('图片重建');c.folder=images;c.click('选择图片目录');c.select('语谱图图片')
            c.until('document.querySelector("img[alt=待校正语谱图]")?.naturalWidth>0')
            c.js("(()=>{const a=document.querySelectorAll('.calibration input');for(const [i,v] of [[1,4.671],[3,22050],[4,-60]]){a[i].value=v;a[i].dispatchEvent(new Event('input',{bubbles:true}));a[i].dispatchEvent(new Event('change',{bubbles:true}))}})()")
            c.click('开始重建');c.until('document.querySelector("img[alt=重建音频频谱]")?.naturalWidth>0',240)
            c.snapshot('spectrum-ayu-dark','M09','Completed image reconstruction from a speech spectrogram PNG; Griffin–Lim result and input comparison')
            assert not c.report['errors'];c.report['success']=True
            p=ROOT/'temporary-site/assets/screens/provenance.json';data=json.loads(p.read_text('utf-8'));names={r['file'] for r in c.report['captures']}|{'egg-gruvbox-light.png'}
            data['captures']=[r for r in data['captures'] if r['file'] not in names]+c.report['captures']
            p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n','utf-8')
        except Exception as e:
            c.report['failure']=str(e);c.w.view.grab().save(str(out/'failure.png'));raise
        finally:c.close();print(json.dumps({'out':str(out),'success':c.report['success']}),flush=True)

if __name__=='__main__':main()
