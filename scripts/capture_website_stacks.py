"""Additional real application states for overlapping website screenshots."""
from capture_website_modules import Capture, ROOT, verification_run, Path, shutil, json
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtTest import QTest
import base64, sys

def put(c, selector, value):
    c.js('(()=>{const e=document.querySelector('+json.dumps(selector)+');e.value='+json.dumps(str(value))+';e.dispatchEvent(new Event("input",{bubbles:true}));e.dispatchEvent(new Event("change",{bubbles:true}))})()')

def draw(c,kind='ipa'):
    c.js('document.querySelector(".spectral-editor").scrollIntoView({block:"center"})');c.pause(400)
    rect=c.js('(()=>{const r=document.querySelector(".spectral-editor canvas").getBoundingClientRect();return {x:r.x,y:r.y,w:r.width,h:r.height}})()')
    # Draw IPA through genuine mouse input, preserving the application's stroke data.
    widget=c.w.view.focusProxy()
    strokes=[[(.18,.15),(.18,.42)],[(.13,.15),(.23,.15)],[(.13,.42),(.23,.42)],[(.32,.42),(.32,.15),(.43,.15),(.46,.2),(.43,.28),(.32,.28)],[(.54,.42),(.62,.15),(.70,.42)],[(.58,.3),(.66,.3)]]
    if kind=='curves':
        import math
        strokes=[[(.12+i*.026,y+.045*math.sin(i*.24)) for i in range(28)] for y in [.20,.32,.45]]
    elif kind=='ptb':
        strokes=[[(.16,.4),(.16,.12),(.27,.12),(.30,.17),(.27,.24),(.16,.24)],[(.38,.12),(.54,.12)],[(.46,.12),(.46,.4)],[(.63,.4),(.63,.12),(.74,.12),(.78,.17),(.74,.25),(.63,.25),(.74,.25),(.79,.31),(.74,.4),(.63,.4)]]
    for points in strokes:
        def pos(p):return QPoint(round(rect['x']+p[0]*rect['w']),round(rect['y']+p[1]*rect['h']))
        QTest.mousePress(widget,Qt.MouseButton.LeftButton,pos=pos(points[0]));c.pause(25)
        for a,b in zip(points,points[1:]):
            for i in range(1,11):QTest.mouseMove(widget,pos((a[0]+(b[0]-a[0])*i/10,a[1]+(b[1]-a[1])*i/10)));c.pause(12)
        QTest.mouseRelease(widget,Qt.MouseButton.LeftButton,pos=pos(points[-1]));c.pause(100)
    c.until('document.querySelector(".axis-note")?.textContent.includes('+json.dumps(str(len(strokes))+' 笔')+')')

def main():
    with verification_run('website','additional-stacks','150% zoom: EGG inverse filtering, spectral drawing, 27 full-connected pitch combinations') as (out,scratch):
        folders={name:scratch/name for name in ['egg','speech','images','pitch']}
        for p in folders.values():p.mkdir()
        for src,dst in [(r'C:\Users\13680\Desktop\project\音频数据\EGG测试\3.wav',folders['egg']/'EGG示例.wav'),(r'C:\Users\13680\Desktop\project\音频数据\线下录音\女1\原声_平静.wav',folders['speech']/'语句示例.wav'),(r'C:\Users\13680\Desktop\project\音频数据\test\dl_m01_0558_ma.wav',folders['pitch']/'单音节_ma.wav')]:shutil.copyfile(src,dst)
        c=Capture(out,scratch)
        try:
            c.until('!!document.querySelector(".home-page")');c.palette('github','light')
            for _ in range(10):
                scale=c.js('document.querySelector("[aria-label=页面缩放比例]")?.textContent')
                if scale=='150%':break
                c.click('放大页面' if int(scale.strip('%'))<150 else '缩小页面')
            assert c.js('document.querySelector("[aria-label=页面缩放比例]").textContent')=='150%'
            c.report['page_scale_percent']=150
            if '--skip-egg' not in sys.argv and '--pitch-only' not in sys.argv:
                c.module('EGG 信号分析');c.folder=folders['egg'];c.click('打开音频目录');c.select('EGG 音频文件')
                c.until('document.querySelector(".egg-live-status")?.textContent.includes("实时预览")',180)
                c.js('(()=>{const a=document.querySelectorAll(".egg-bottom-bar .selection-controls input");for(const [i,v] of [[1,40.5],[0,40.4]]){a[i].value=v;a[i].dispatchEvent(new Event("input",{bubbles:true}));a[i].dispatchEvent(new Event("change",{bubbles:true}))}})()')
                c.pause(1000);c.until('document.querySelector(".egg-live-status")?.textContent.includes("实时预览")',180)
                c.click('逆滤波 IF');c.until('document.querySelectorAll(".inverse-grid svg").length>=4',240)
                c.snapshot('egg-inverse-github-light','M03','Actual inverse filtering of stable vowel at 40.4–40.5 s; GitHub white theme, 150% zoom')
                c.click('返回分析')
            if '--pitch-only' not in sys.argv:
                c.palette('ayu','dark');c.module('语谱图转音频');c.click('音频藏信息');c.folder=folders['speech'];c.click('打开音频目录');c.select('藏信息源音频')
                c.until('document.body.innerText.includes("画布已载入")',120)
                data=c.js('document.querySelector(".spectral-editor canvas").toDataURL("image/png")');(folders['images']/'语谱图示例.png').write_bytes(base64.b64decode(data.split(',')[1]))
                draw(c);c.click('生成藏信息音频');c.until('document.querySelector("img[alt=重建音频频谱]")?.naturalWidth>0',240)
                c.snapshot('spectrum-audio-draw-ayu-dark','M09','Actual IPA strokes on speech spectrum; original phase reconstruction completed; 150% zoom')
                c.click('图片涂鸦重建');c.folder=folders['images'];c.click('选择图片目录');c.select('语谱图图片')
                c.until('document.querySelector("img[alt=待校正语谱图]")?.naturalWidth>0')
                for i,v in [(1,4.671),(3,22050),(4,-60)]:
                    c.js('(()=>{const e=document.querySelectorAll(".calibration input")['+str(i)+'];e.value='+str(v)+';e.dispatchEvent(new Event("input",{bubbles:true}));e.dispatchEvent(new Event("change",{bubbles:true}))})()')
                c.click('应用校正并绘图');c.until('document.querySelector(".spectral-editor canvas")?.width>0',120)
                draw(c);c.click('生成涂鸦音频');c.until('document.querySelector("img[alt=重建音频频谱]")?.naturalWidth>0',240)
                c.snapshot('spectrum-image-draw-ayu-dark','M09','Actual six-stroke IPA drawing on calibrated spectrogram; Griffin–Lim synthesis completed; 150% zoom')
            c.palette('one','light');c.module('变速变调');c.folder=folders['pitch'];c.click('打开音频目录');c.select('变速变调音频')
            c.until('document.querySelector("svg.m08-curve .original")?.getAttribute("d").includes("L")',180)
            c.until('document.querySelector(".m08-page")?.getAttribute("aria-busy")==="false"',180)
            c.click('编辑拐点表');c.click('添加行')
            ends=c.js('[...document.querySelectorAll("input[aria-label^=时间]")].map(e=>Number(e.value))');mid=round((ends[0]+ends[-1])/2,3)
            put(c,'input[aria-label="时间 1"]',mid)
            for i,values in enumerate(['90, 110, 130','100, 120, 140','90, 110, 130']):put(c,'input[aria-label="频率 '+str(i)+'"]',values);put(c,'select[aria-label="连接 '+str(i)+'"]','full')
            c.click('保存更改');c.click('批量生成')
            c.until('document.querySelectorAll(".history-plot svg path").length===27',240)
            c.report['batch']={'start':ends[0],'turn':mid,'end':ends[-1],'f0_hz':[[90,110,130],[100,120,140],[90,110,130]],'connection':'full','outputs':27}
            c.snapshot('pitch-batch-one-light','M08','Monosyllable ma; three F0 values at start/turn/end, full cartesian connection; 27 synthesized outputs with measured F0; 150% zoom')
            assert not c.report['errors'];c.report['success']=True
        except Exception as e:
            c.report['failure']=str(e);c.w.view.grab().save(str(out/'failure.png'));raise
        finally:
            p=ROOT/'temporary-site/assets/screens/provenance.json';data=json.loads(p.read_text('utf-8'));names={r['file'] for r in c.report['captures']}
            for row in c.report['captures']:row['page_scale_percent']=150
            data['captures']=[r for r in data['captures'] if r['file'] not in names]+c.report['captures'];p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n','utf-8')
            c.close();print(json.dumps({'out':str(out),'success':c.report['success']}),flush=True)

if __name__=='__main__':main()
