"""Actual maximized M12 manual screenshots, isolated synthetic stereo and native saves."""
from pathlib import Path
import base64, hashlib, json, math, os, struct, subprocess, sys, time, wave

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from workbench_source import configure
from verification_artifacts import verification_run
os.environ['QT_QPA_PLATFORM']='windows'
os.environ['QTWEBENGINE_CHROMIUM_FLAGS']='--disable-gpu --mute-audio'
for key in ('QT_ENABLE_HIGHDPI_SCALING','QT_SCALE_FACTOR','QT_SCREEN_SCALE_FACTORS'):os.environ.pop(key,None)

def main():
 configure()
 from PyQt6.QtCore import QEventLoop,Qt,QTimer
 from PyQt6.QtWidgets import QApplication,QFileDialog
 from PyQt6 import sip
 from ptb_desktop.host import Workbench,register_scheme
 from ptb_worker.local_workspace import prepare_workspace
 register_scheme();app=QApplication(['M12 manual screenshots'])
 recipe='Native Windows Qt DPR1.5, page100, light, maximized and collapsed navigation; own 3.2s PCM16 stereo harmonic pulses, three interval tiers and one retained point tier; no natural speech or physical devices'
 with verification_run('m12-manual','screenshots-20261021',recipe) as (out,scratch):
  print(out,flush=True);db,cache=prepare_workspace(scratch/'state',ROOT/'backend/migrations');inputs=scratch/'inputs';inputs.mkdir()
  for name in ['标注示例','新建示例','多候选']:
   with wave.open(str(inputs/(name+'.wav')),'wb') as stream:
    stream.setparams((2,2,16000,0,'NONE','not compressed'))
    stream.writeframes(b''.join(struct.pack('<hh',round(2500*math.sin(i*.0785)) if i%16000<8000 else 0,round(7000*math.sin(i*.0942)+1800*math.sin(i*.1884)) if any(a<=i/16000<b for a,b in [(.2,.7),(.9,1.4),(1.6,2.1),(2.3,2.9)]) else 0) for i in range(51200)))
  rows=[(0,.2,''),(.2,.7,'zhe4'),(.7,.9,''),(.9,1.4,'shi4'),(1.4,1.6,''),(1.6,2.1,'zhe4'),(2.1,2.3,''),(2.3,2.9,'hua4'),(2.9,3.2,'')]
  text='"ooTextFile short" "TextGrid" 0 3.2 <exists> 4 '+' '.join('"IntervalTier" "'+n+'" 0 3.2 '+str(len(rows))+' '+' '.join(f'{a} {b} "{label if n=="words" else ("4" if n=="tones" and label else label)}"' for a,b,label in rows) for n in ['words','phones','tones'])+' "TextTier" "events" 0 3.2 1 0.1 "保留点层"'
  for name in ['标注示例','多候选_方案甲','多候选_方案乙']:(inputs/(name+'.TextGrid')).write_text(text,encoding='utf-8')
  originals={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.iterdir()}
  window=Workbench(ROOT/'frontend/dist',test=True,jobs_path=db,local_files_root=cache,vocal_profile=scratch/'vocal',start_module='M12')
  window.resize(window.screen().geometry().width(),window.screen().geometry().height());window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen);window.showMaximized();window.view.setZoomFactor(1);window.page.setAudioMuted(True)
  picker=QFileDialog.getExistingDirectory;QFileDialog.getExistingDirectory=lambda *a,**k:str(inputs)
  report=dict(success=False,captures=[],checks=[],scope=recipe,inputs=originals)
  def pause(ms=100):
   loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
  def js(code):
   wrapped='(()=>{try{return {ok:true,value:eval('+json.dumps(code)+')??null};}catch(e){return {ok:false,error:String(e)};}})()';loop=QEventLoop();values=[]
   window.page.runJavaScript(wrapped,lambda v:(values.append(v),loop.quit()));QTimer.singleShot(6000,loop.quit);loop.exec()
   if not values or not values[0] or not values[0].get('ok'):raise RuntimeError(str(values)+' '+code)
   return values[0].get('value')
  def until(code):
   deadline=time.monotonic()+40
   while time.monotonic()<deadline:
    if js(code):return
    pause()
   raise AssertionError(code+'\n'+str(js('document.body.innerText.slice(-3500)')))
  def click(name):
   e='[...document.querySelectorAll("button")].find(b=>b.offsetParent&&(b.getAttribute("aria-label")==='+json.dumps(name)+'||b.textContent.trim().replace(/\\s*\\*$/,"")==='+json.dumps(name)+'))'
   until('!!('+e+')&&!('+e+').disabled');js(e+'.click()');pause(120)
  def fill(label,value,event='input'):
   js('(()=>{const e=document.querySelector('+json.dumps('[aria-label="'+label+'"]')+');e.value='+json.dumps(value)+';e.dispatchEvent(new Event('+json.dumps(event)+',{bubbles:true}));})()');pause()
  def choose(name):
   js('[...document.querySelectorAll(".annotation-file-list button")].find(b=>b.textContent.trim()==='+json.dumps(name)+').click()');until('document.querySelector(".annotation-page").getAttribute("aria-busy")==="false"');pause(200)
  def grid(time_value,row=0,double=False):
   js('document.querySelector(".annotation-grid").scrollIntoView({block:"center"})');pause()
   x,y=js('(()=>{const r=document.querySelector(".annotation-grid").getBoundingClientRect();return [r.x+r.width*'+str(time_value/3.2)+',r.y+r.height*'+str((row+.5)/3)+']})()')
   # Hidden native Qt windows do not receive system double-click activation.
   # Dispatch the real canvas dblclick handler using its measured CSS position.
   js('document.querySelector(".annotation-grid").dispatchEvent(new MouseEvent("dblclick",{bubbles:true,button:0,clientX:'+str(x)+',clientY:'+str(y)+'}))')
   pause(150)
  def capture(name,detail=False):
   until('document.fonts.status==="loaded"');
   if name.startswith('save-') or name=='close-protection':js('document.querySelector(".annotation-settings").scrollTop=0')
   js('document.querySelector(".annotation-editor").scrollTop=0');pause(350);window.view.repaint();pause(120)
   image=window.view.grab().toImage();raw=[image.width(),image.height()];dpr=image.devicePixelRatio();assert raw in ([2560,1600],[2561,1601]) and abs(dpr-1.5)<.01,(raw,dpr)
   assert window.isMaximized() and window.view.zoomFactor()==1 and js('document.documentElement.dataset.theme')=='light';assert js('document.querySelector(".app-shell").classList.contains("collapsed")')
   if raw!=[2560,1600]:image=image.copy(0,0,2560,1600)
   path=out/(name+'.png');assert image.save(str(path));row=dict(name=name,file=str(path),image=[2560,1600],raw_image=raw,dpr=dpr,pageZoom=1,maximized=True,theme='light',navigationCollapsed=True,sha256=hashlib.sha256(path.read_bytes()).hexdigest());report['captures'].append(row)
   if detail:
    r=js('(()=>{const r=document.querySelector("dialog[open]").getBoundingClientRect();return [r.x,r.y,r.width,r.height]})()');x=max(0,int(r[0]*dpr)-18);y=max(0,int(r[1]*dpr)-18);w=min(2560-x,int(r[2]*dpr)+36);h=min(1600-y,int(r[3]*dpr)+36);crop=image.copy(x,y,w,h);target=out/(name+'-detail.png');assert crop.save(str(target));report['captures'].append(dict(row,name=name+'-detail',file=str(target),image=[w,h],crop=[x,y,w,h],parent=name,sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
   print('Captured '+name,flush=True)
  try:
   until('!!document.querySelector(".annotation-page")');click('设置');click('浅色');click('TextGrid标注');click('收起侧栏');click('打开音频目录');until('document.querySelectorAll(".annotation-file-list button").length===3');choose('标注示例.wav')
   assert js('document.querySelector(".annotation-grid").dataset.tierCount')=='3'
   assert js('[...document.querySelectorAll(".module-toolbar button")].some(b=>b.textContent.trim()==="音标键盘")&&document.querySelector(".label-editor button[aria-pressed]")===null')
   capture('overview')
   click('吸附与关联');capture('links');click('完成')
   grid(.4);click('强度贴合');assert js('document.querySelector("[aria-label=强度贴合声道]").value')=='';fill('强度贴合声道','1','change');capture('intensity');click('关闭')
   click('搜索替换');fill('搜索词层文本','zhe4');fill('替换文本','zhe2');assert js('document.querySelector("[aria-label=搜索匹配位置]").textContent.trim()')=='1 / 2';capture('search');click('下一个');assert js('document.querySelector("[aria-label=搜索匹配位置]").textContent.trim()')=='2 / 2';click('全部替换');capture('replace-confirm');click('取消');until('!!document.querySelector("[aria-label=搜索词层文本]")');assert js('document.querySelector("[aria-label=替换文本]").value')=='zhe2';click('关闭')
   grid(.4);click('音标键盘');until('document.querySelectorAll(".ipa-key").length>300');capture('keyboard');click('关闭音标键盘')
   report['checks'].append('Native current top-toolbar search, confirm/cancel, matching navigation, optional intensity channel and keyboard dialogs')
   choose('多候选.wav');until('document.querySelector("dialog[open]")?.textContent.includes("选择关联 TextGrid")');fill('选择关联标注文件',js('document.querySelector("[aria-label=选择关联标注文件]").options[1].value'),'change');capture('association',detail=True);click('打开标注');until('document.querySelector(".annotation-page").getAttribute("aria-busy")==="false"')
   choose('新建示例.wav');click('创建标注层')
   for label,value in [('新建音节层名','音节'),('新建音素层名','音素'),('新建第三层名','声调')]:fill(label,value)
   capture('create-layers',detail=True);click('创建层');until('document.querySelector(".annotation-grid").dataset.tierCount==="3"');click('粘贴词表');fill('拼音词表','zhe4 shi4 zhe4 hua4');click('应用词表');js('[...document.querySelectorAll(".sequence-toggle input")][0].click()');pause();capture('dictionary-wordlist')
   grid(.2,double=True);until('document.querySelector(".sequence-hint").textContent.includes("待定起点")');capture('sequence-start');grid(.7,double=True);grid(.4,1,double=True);until('document.querySelector(".annotation-settings").textContent.includes("首个")||document.querySelector(".annotation-settings").textContent.includes("边界")');capture('sequence-phones')
   choose('标注示例.wav');raw=base64.b64encode(text.encode()).decode();js('(()=>{const e=[...document.querySelectorAll("input[type=file]")].find(e=>e.accept===".TextGrid,.textgrid"),d=new DataTransfer();d.items.add(new File([Uint8Array.from(atob('+json.dumps(raw)+'),c=>c.charCodeAt(0))],"参考示例.TextGrid",{type:"text/plain"}));e.files=d.files;e.dispatchEvent(new Event("change",{bubbles:true}));})()');pause();fill('参考起点','0.5');fill('参考终点','1');fill('参考复用模式','inside','change');js('document.querySelector("[aria-label=参考复用模式]").scrollIntoView({block:"center"})');capture('reference-unpaired-lip')
   grid(.4);fill('编辑选中标注文本','音标 æ');js('document.querySelector("[aria-label=编辑选中标注文本]").dispatchEvent(new Event("change",{bubbles:true}))');fill('TextGrid 保存后缀','_手册示例');click('保存 TextGrid');until('document.querySelector(".annotation-page").getAttribute("aria-busy")==="false"');assert (inputs/'标注示例_手册示例.TextGrid').is_file();capture('save-success')
   fill('TextGrid 保存后缀','/错误');click('保存 TextGrid');until('!!document.querySelector(".module-status-error")');capture('save-error');fill('TextGrid 保存后缀','_手册示例');click('保存 TextGrid');until('document.querySelector(".annotation-page").getAttribute("aria-busy")==="false"');fill('编辑选中标注文本','未保存的演示');click('关闭 TextGrid标注');until('document.querySelector("dialog[open]")?.textContent.includes("保存修改并关闭")');capture('close-protection');click('取消关闭')
   report['checks'].append('Actual file association, three custom layers, sequence endpoints/phones, reference selection, native save/error and unsaved-close protection')
   assert all(hashlib.sha256((inputs/n).read_bytes()).hexdigest()==h for n,h in originals.items());report['checks'].append('Original owned WAV/TextGrid hashes unchanged');report['success']=True
  except Exception as exc:
   report['error']=repr(exc);report['page']=js('document.body.innerText');window.view.grab().save(str(out/'failure.png'));raise
  finally:
   report['sources']={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['frontend/src/modules/annotation/AnnotationPage.vue','frontend/dist/index.html',str(Path(__file__).relative_to(ROOT))]};(out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');QFileDialog.getExistingDirectory=picker;window.closing=True;window.close();app.processEvents();sip.delete(window);app.processEvents();sip.delete(app)
  print(json.dumps(dict(success=True,out=str(out),captures=len(report['captures']))),flush=True)

if __name__=='__main__':
 if '--child' in sys.argv:main()
 else:
  p=subprocess.run([sys.executable,'-B','-X','utf8',__file__,'--child'],capture_output=True,text=True,encoding='utf-8',errors='replace');print(p.stdout,flush=True);print(p.stderr,flush=True)
  prefix=str(ROOT/'output/validation/m12-manual/screenshots-20261021-');paths=[Path(line.strip()) for line in p.stdout.splitlines() if line.strip().startswith(prefix)]
  if paths:
   clean=subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',str(ROOT/'scripts/cleanup_test_scratch.ps1'),'-RunDirectory',str(paths[0])],capture_output=True,text=True,encoding='utf-8',errors='replace');print(clean.stdout+clean.stderr,flush=True);(paths[0]/'console.log').write_text(p.stdout+p.stderr+clean.stdout+clean.stderr,encoding='utf-8')
   if clean.returncode:raise RuntimeError('Owned screenshot scratch retained')
  raise SystemExit(p.returncode)
