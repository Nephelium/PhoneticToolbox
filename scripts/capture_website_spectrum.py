"""Three distinct completed reconstruction examples, 2560x1440 at 150%."""
from capture_website_stacks import Capture, ROOT, verification_run, Path, shutil, json, base64, draw

def result(c):
    c.until('document.querySelector("img[alt=重建音频频谱]")?.naturalWidth>0',240)
    c.until('[...document.querySelectorAll("button")].some(e=>e.textContent.trim()==="保存重建结果"&&!e.disabled)')
    c.js('document.querySelector(".comparison").closest("section").scrollIntoView({block:"start"})');c.pause(400)

def main():
    with verification_run('website','spectrum-three','Three modes: plain image; hand-painted curves with Griffin–Lim; PTB message with original phase') as (out,scratch):
        speech=scratch/'speech';speech.mkdir();images=scratch/'images';images.mkdir()
        shutil.copyfile(Path(r'C:\Users\13680\Desktop\project\音频数据\线下录音\女1\原声_平静.wav'),speech/'语句示例.wav')
        c=Capture(out,scratch)
        try:
            c.until('!!document.querySelector(".home-page")');c.palette('ayu','dark');c.module('语谱图转音频');c.click('音频藏信息');c.folder=speech;c.click('打开音频目录');c.select('藏信息源音频')
            c.until('document.body.innerText.includes("画布已载入")',120)
            data=c.js('document.querySelector(".spectral-editor canvas").toDataURL("image/png")');(images/'语谱图示例.png').write_bytes(base64.b64decode(data.split(',')[1]))
            draw(c,'ptb');c.click('生成藏信息音频');result(c)
            c.snapshot('spectrum-audio-draw-ayu-dark','M09','PTB message painted using four genuine mouse strokes; original-phase audio generated; target/reconstructed spectrum and waveform shown')
            c.click('图片重建');c.folder=images;c.click('选择图片目录');c.select('语谱图图片');c.until('document.querySelector("img[alt=待校正语谱图]")?.naturalWidth>0')
            for i,v in [(1,4.671),(3,22050),(4,-60)]:
                c.js('(()=>{const e=document.querySelectorAll(".calibration input")['+str(i)+'];e.value='+str(v)+';e.dispatchEvent(new Event("input",{bubbles:true}));e.dispatchEvent(new Event("change",{bubbles:true}))})()')
            c.click('开始重建');result(c);c.snapshot('spectrum-ayu-dark','M09','Plain image reconstruction without drawings; actual Griffin–Lim result and input comparison')
            c.click('图片涂鸦重建');c.js('document.querySelector(".original-card").open=true');c.click('应用校正并绘图');c.until('document.querySelector(".spectral-editor canvas")?.width>0',120)
            draw(c,'curves');c.click('生成涂鸦音频');result(c)
            c.snapshot('spectrum-image-draw-ayu-dark','M09','Three curved bands drawn through mouse input; actual Griffin–Lim output; painted input, output spectrum and waveform shown')
            assert len(c.report['captures'])==3 and not c.report['errors'];c.report['success']=True
        except Exception as e:c.report['failure']=str(e);c.w.view.grab().save(str(out/'failure.png'));raise
        finally:
            p=ROOT/'temporary-site/assets/screens/provenance.json';data=json.loads(p.read_text('utf-8'));names={r['file'] for r in c.report['captures']};data['captures']=[r for r in data['captures'] if r['file'] not in names]+c.report['captures'];p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n','utf-8')
            c.close();print(json.dumps({'out':str(out),'success':c.report['success']}),flush=True)

if __name__=='__main__':main()
