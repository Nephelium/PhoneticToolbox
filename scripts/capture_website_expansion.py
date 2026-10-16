"""Real completed examples, isolated state; 2560x1440 / 150%."""
import os,sys,hashlib
from capture_website_modules import Capture,ROOT,verification_run,Path,shutil,json

def lpc(c,scratch):
    from scipy.io import wavfile
    folder=scratch/'元音示例';folder.mkdir()
    rate,a=wavfile.read(Path(r'C:\Users\13680\Desktop\project\音频数据\test\dl_m02_0558_ma.wav'));wavfile.write(folder/'真实单音节ma.wav',rate,a)
    c.palette('night-owl','dark');c.module('LPC 谱图');c.folder=folder;c.click('打开音频目录');c.select('LPC 音频文件');c.click('开始分析')
    c.until('document.querySelector(".lpc-results")!==null',180);c.click('LPC 频谱');c.click('动态纵轴')
    c.pause(900)
    bounds=c.js('(()=>{const keys=[".lpc-spectrum",".lpc-page .workbench-center",".lpc-card",".scientific-plot"];return {scale:getComputedStyle(document.documentElement).getPropertyValue("--page-scale"),innerHeight,items:keys.map(k=>{const e=document.querySelector(k);return {key:k,rect:e.getBoundingClientRect().toJSON(),height:getComputedStyle(e).height,scroll:e.scrollHeight}})}})()')
    c.report['lpc_bounds']=bounds;print(json.dumps(bounds),flush=True)
    c.w.view.grab().save(str(c.out/'lpc-layout.png'))
    assert bounds['items'][0]['rect']['bottom']<=bounds['items'][1]['rect']['bottom'], 'LPC axis and controls must fit the visible panel'
    c.snapshot('lpc-night-owl-dark','M04','Completed actual LPC analysis of a real /ma/ recording; dynamic y axis keeps all spectral peaks visible')


def apply_pinyin_resources(c):
    from PyQt6.QtWidgets import QFileDialog
    dictionary=Path(r'C:\Users\13680\Desktop\PhoneticToolbox\mfa_models\dictionary\mandarin_pinyin_tab.dict')
    model=dictionary.parent.parent/'acoustic/mandarin.zip'
    for label,file in [('声学模型',model),('发音词典',dictionary)]:
        QFileDialog.getOpenFileName=lambda *a,_file=file,**k:(str(_file),'')
        c.js('(()=>{const e=[...document.querySelectorAll(".mfa-page label")].find(e=>e.textContent.trim().startsWith('+json.dumps(label)+')).querySelector("select");e.value="import";e.dispatchEvent(new Event("change",{bubbles:true}))})()')
        c.until('document.querySelector(".mfa-page")?.getAttribute("aria-busy")==="false"');c.pause(500)
    c.click('检查并应用所选资源');c.until('document.body.innerText.includes("已检查并应用：mandarin.zip")',180)


def mfa_only():
    from PyQt6.QtWidgets import QFileDialog
    with verification_run('website','mfa-pinyin','Original pinyin transcript and user-specified tab dictionary; isolated alignment, original manual labels unchanged') as (out,scratch):
        component=scratch/'components';component.mkdir()
        shutil.copyfile(ROOT/'output/validation/m11/qt-642b638d008c497abb576e5db63dcbe7/components/registry.json',component/'registry.json')
        os.environ['PTB_M11_COMPONENT_ROOT']=str(component)
        natural=Path(r'C:\Users\13680\Desktop\project\音频数据\线下录音\女1')
        dictionary=Path(r'C:\Users\13680\Desktop\PhoneticToolbox\mfa_models\dictionary\mandarin_pinyin_tab.dict')
        model=dictionary.parent.parent/'acoustic/mandarin.zip'
        originals=[natural/('女童_愉快2'+suffix) for suffix in ['.wav','.lab','.TextGrid','_自动保存.TextGrid']]+[dictionary,model]
        hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in originals}
        corpus=scratch/'MFA独立副本';corpus.mkdir()
        for p in originals[:2]:shutil.copyfile(p,corpus/p.name)
        c=Capture(out,scratch)
        try:
            c.until("!!document.querySelector('.home-page')");c.palette('monokai','dark');c.module('MFA 自动标注')
            apply_pinyin_resources(c)
            c.folder=corpus;c.click('打开音频目录');c.click('开始对齐')
            c.until('document.querySelectorAll(".mfa-page .result-row").length>0',300)
            assert 'mandarin_pinyin_tab.dict' in c.js('document.querySelector(".applied-resources").textContent')
            c.snapshot('mfa-monokai-dark','M11','Completed alignment with original pinyin LAB, user-specified mandarin_pinyin_tab.dict and compatible mandarin.zip; all original WAV/LAB/manual TextGrids/dictionary/model unchanged')
            c.report['originals_unchanged']=True;c.report['original_hashes']=hashes;c.report['success']=True
            p=ROOT/'temporary-site/assets/screens/provenance.json';data=json.loads(p.read_text('utf-8'));data['captures']=[r for r in data['captures'] if r['file']!='mfa-monokai-dark.png']+c.report['captures'];p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n','utf-8')
        except Exception as e:
            c.report['failure']=str(e);c.w.view.grab().save(str(out/'failure.png'));raise
        finally:
            assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
            c.close();print(str(out),flush=True)

def lpc_only():
    with verification_run('website','lpc-view','LPC dynamic axis re-capture at 2560x1440 and 150%, user-selected real /ma/') as (out,scratch):
        c=Capture(out,scratch)
        try:
            c.until("!!document.querySelector('.home-page')");lpc(c,scratch)
            c.palette('github','light');c.module('LPC 谱图');c.pause(900)
            assert c.js('document.querySelector(".lpc-spectrum").getBoundingClientRect().bottom<=document.querySelector(".lpc-page .workbench-center").getBoundingClientRect().bottom')
            c.w.view.grab().save(str(out/'lpc-light-verification.png'));c.report['light_theme_axis_visible']=True;c.report['success']=True
            p=ROOT/'temporary-site/assets/screens/provenance.json';data=json.loads(p.read_text('utf-8'));data['captures']=[r for r in data['captures'] if r['file']!='lpc-night-owl-dark.png']+c.report['captures'];p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n','utf-8')
        finally:c.close();print(str(out),flush=True)

def main():
    with verification_run('website','expansion','User-selected real recordings; original manual TextGrid preserved; isolated native screenshots') as (out,scratch):
        component=scratch/'components';component.mkdir()
        registry=ROOT/'output/validation/m11/qt-642b638d008c497abb576e5db63dcbe7/components/registry.json'
        shutil.copyfile(registry,component/'registry.json');os.environ['PTB_M11_COMPONENT_ROOT']=str(component)
        inputs=scratch/'合成示例';inputs.mkdir()
        for i,name in enumerate(['dl_m02_0558_ma.wav','dl_m04_0558_ma.wav'],1):shutil.copyfile(Path(r'C:\Users\13680\Desktop\project\音频数据\test')/name,inputs/f'真实端点{i}.wav')
        natural=Path(r'C:\Users\13680\Desktop\project\音频数据\线下录音\女1')
        originals=[natural/('女童_愉快2'+suffix) for suffix in ['.wav','.lab','.TextGrid','_自动保存.TextGrid']]
        hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in originals}
        corpus=scratch/'MFA独立副本';corpus.mkdir();audio=corpus/'女童_愉快2.wav';shutil.copyfile(originals[0],audio)
        shutil.copyfile(originals[1],audio.with_suffix('.lab'))
        manual=scratch/'手工标注副本';manual.mkdir()
        for p in originals[:3]:shutil.copyfile(p,manual/p.name)
        saved=scratch/'标注结果';saved.mkdir()
        c=Capture(out,scratch)
        def fill(selector,value):
            c.js('(()=>{const e=document.querySelector('+json.dumps(selector)+');e.value='+json.dumps(str(value))+';e.dispatchEvent(new Event("input",{bubbles:true}));e.dispatchEvent(new Event("change",{bubbles:true}));})()');c.pause(200)
        try:
            c.until("!!document.querySelector('.home-page')")
            c.palette('gruvbox','dark');c.module('发声类型合成');c.folder=inputs;c.click('打开音频目录');c.select('源音频',1);c.select('目标音频',2)
            c.click('提取 F0');c.until('document.body.innerText.includes("分析完成，控制点尚未修改")',180)
            fill('[aria-label="连续统步数"]',7)
            fill('[aria-label="连续统类型"]',1)
            c.click('生成当前');c.until('document.querySelectorAll(".result-audios button").length===8',240)
            c.snapshot('phonation-gruvbox-dark','M07','Completed 7-step phonation-only continuum; real ma recordings dl_m02/dl_m04 from the user-selected test directory')
            c.palette('monokai','dark');c.module('MFA 自动标注');apply_pinyin_resources(c);c.folder=corpus;c.click('打开音频目录')
            c.until('document.querySelector(".mfa-page")?.getAttribute("aria-busy")==="false"')
            c.click('开始对齐');c.until('document.querySelectorAll(".mfa-page .result-row").length>0',300)
            c.snapshot('mfa-monokai-dark','M11','Actual MFA 3.3.8 completed alignment of user-selected female speech and original pinyin transcript, mandarin_pinyin_tab.dict and mandarin.zip; input/output isolated from original manual annotation; automatic boundaries need checking')
            c.folder=saved;c.click('选择输出目录');c.click('保存完整结果到输出目录');c.pause(1800)
            grids=list(saved.rglob('*.TextGrid'));assert grids, c.js('document.body.innerText')
            assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
            c.palette('material','light');c.module('TextGrid标注');c.folder=manual;c.click('打开音频目录')
            c.until('document.querySelector(".annotation-files button")!==null')
            c.js('document.querySelector(".annotation-files button").click()');c.pause(1800)
            c.until('document.body.innerText.includes("已加载：女童_愉快2.TextGrid")',60)
            c.select('词层名',1);c.select('音素层名',2)
            c.snapshot('textgrid-material-light','M12','User-provided manual TextGrid paired with original female speech; distinct from automatic MFA result; original files unchanged')
            lpc(c,scratch)
            c.palette('tokyo-night','light');c.click('外观');c.snapshot('appearance-tokyo-night-light','settings-appearance','Current appearance settings, Tokyo Night light palette, 150% app zoom')
            assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items());c.report['originals_unchanged']=True;c.report['original_hashes']=hashes
            assert not c.report['errors'];c.report['success']=True
            p=ROOT/'temporary-site/assets/screens/provenance.json';data=json.loads(p.read_text('utf-8'));names={r['file'] for r in c.report['captures']};data['captures']=[r for r in data['captures'] if r['file'] not in names]+c.report['captures'];p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n','utf-8')
        except Exception as e:
            c.report['failure']=str(e);c.w.view.grab().save(str(out/'failure.png'));raise
        finally:
            assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
            c.close();print(json.dumps({'out':str(out),'success':c.report['success']}),flush=True)
if __name__=='__main__':
    if '--lpc-only' in sys.argv:lpc_only()
    elif '--mfa-only' in sys.argv:mfa_only()
    else:main()
