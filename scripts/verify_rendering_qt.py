"""Visible default-GPU regression: real Qt pixels, focus, fonts and scrolling."""
import json,os,sqlite3,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from workbench_source import configure
configure()
os.environ['QT_QPA_PLATFORM']='windows'
os.environ['QTWEBENGINE_CHROMIUM_FLAGS']='--mute-audio'
from PyQt6.QtCore import QEventLoop,QTimer
from PyQt6.QtGui import QImage
from PyQt6.QtWidgets import QApplication
from ptb_desktop.host import Workbench,register_scheme
from ptb_worker.local_acoustic_files import initialize_local_files
from verification_artifacts import verification_run
import numpy as np

def main():
 with verification_run('rendering','visible-qt','Visible Windows GPU defaults; generated checkerboard; Qt framebuffer text pixels and scroll return; 3 fonts x 7 sizes x 3 windows; sidebar style matrix') as (out,scratch):
  db=scratch/'jobs.sqlite3'
  with sqlite3.connect((ROOT/'output/validation/p06/local-state.sqlite3').as_uri()+'?mode=ro',uri=True) as src,sqlite3.connect(db) as dst:src.backup(dst)
  cache=scratch/'cache';cache.mkdir();initialize_local_files(cache)
  register_scheme();app=QApplication(['visible-rendering-owned-QA'])
  w=Workbench(ROOT/'frontend/dist',test=True,jobs_path=db,local_files_root=cache,vocal_profile=scratch/'vocal')
  w.resize(1440,900);w.show()
  report={'success':False,'styles':[],'frames':[],'errors':[],'mode':'visible windows; default GPU; mute audio only'}
  w.page.renderProcessTerminated.connect(lambda status,code:report['errors'].append([status.name,code]))
  def pause(ms=120):
   loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
  def js(code):
   loop=QEventLoop();values=[];w.page.runJavaScript(code,lambda value:(values.append(value),loop.quit()));QTimer.singleShot(10000,loop.quit);loop.exec()
   assert values,'JS timeout';return values[0]
  def until(code):
   end=time.monotonic()+60
   while time.monotonic()<end:
    if js(code):return
    pause()
   raise AssertionError(code)
  def click(label):
   exp='[...document.querySelectorAll("button")].find(b=>b.offsetParent&&(b.getAttribute("aria-label")||b.textContent.trim())==='+json.dumps(label)+')'
   until('!!('+exp+')');js(exp+'.click()');pause()
  def pixels(label):
   # Fixed sidebar text must remain in the actual Qt framebuffer during scrolling.
   boxes=js("[...document.querySelectorAll('.nav-item,.m17-chart-viewport button')].filter(e=>{const r=e.getBoundingClientRect();return r.top>100&&r.bottom<innerHeight-100&&e.offsetParent&&(!e.closest('.m17-chart-viewport')||(r.top>e.closest('.m17-chart-viewport').getBoundingClientRect().top&&r.bottom<e.closest('.m17-chart-viewport').getBoundingClientRect().bottom))}).filter((e,i,a)=>e.matches('.nav-item')?a.filter(x=>x.matches('.nav-item')).indexOf(e)<6:a.filter(x=>!x.matches('.nav-item')).indexOf(e)<6).map(e=>{const r=e.getBoundingClientRect(),c=document.createElement('canvas').getContext('2d');c.fillStyle=getComputedStyle(e.querySelector('.m17-ipa')||e).color;c.fillRect(0,0,1,1);return {box:[r.x,r.y,r.width,r.height],color:[...c.getImageData(0,0,1,1).data].slice(0,3)}})")
   image=w.view.grab().toImage().convertToFormat(QImage.Format.Format_RGBA8888);buf=image.bits();buf.setsize(image.sizeInBytes());array=np.frombuffer(buf,dtype=np.uint8).reshape(image.height(),image.bytesPerLine()//4,4)
   scale=image.width()/w.view.width();counts=[]
   for item in boxes:
    x,y,bw,bh=item['box'];x0,y0,x1,y1=[round(v*scale) for v in (x,y,x+bw,y+bh)]
    crop=array[y0:y1,x0:x1,:3].astype(float);counts.append(int((np.max(np.abs(crop-np.array(item['color'])),axis=2)<38).sum()))
   record={'case':label,'text_pixels':counts};report['frames'].append(record)
   if len(counts)<3 or min(counts)<8:image.save(str(out/'missing-text.png'));raise AssertionError(record)
   return image
  try:
   until("!!document.querySelector('#tool-sidebar')&&document.fonts.status==='loaded'")
   click('设置')
   js("(()=>{const c=document.createElement('canvas');c.width=1200;c.height=800;const x=c.getContext('2d');for(let y=0;y<800;y+=30)for(let v=0;v<1200;v+=30){x.fillStyle=(v+y)%60?'#4477aa':'#eebbaa';x.fillRect(v,y,30,30)}c.toBlob(blob=>{const d=new DataTransfer();d.items.add(new File([blob],'checker.png',{type:'image/png'}));const e=document.querySelector('input[aria-label=上传背景图]');e.files=d.files;e.dispatchEvent(new Event('change',{bubbles:true}));});})()")
   until("getComputedStyle(document.documentElement).getPropertyValue('--button-glass-image').includes('blob:')")
   click('国际音标表Plus');pause(300)
   for theme in ('light','dark'):
    for material in ('solid','glass'):
     for mode in ('auto','all','plain'):
      for effects in ('on','off'):
       js('Object.assign(document.documentElement.dataset,'+json.dumps(dict(theme=theme,buttonMaterial=material,buttonStyle=mode,buttonEffects=effects))+')');pause(150)
       # Programmatic focus with focusVisible models the keyboard selector explicitly.
       result=js("(()=>{const e=document.querySelector('.nav-item.selected');e.focus({focusVisible:true});const c=getComputedStyle(e);return {outline:c.outlineStyle,border:c.borderColor,shadow:c.boxShadow,background:c.backgroundColor,decoration:c.textDecorationLine,filtered:[...document.querySelectorAll('*')].filter(n=>getComputedStyle(n).backdropFilter!=='none').length}})()")
       assert result['outline']=='none' and result['border']=='rgba(0, 0, 0, 0)' and result['shadow']=='none' and result['filtered']==0,result
       report['styles'].append(dict(theme=theme,material=material,mode=mode,effects=effects,**result));pixels(f'{theme}-{material}-{mode}-{effects}')
   js("Object.assign(document.documentElement.dataset,{theme:'light',buttonMaterial:'glass',buttonStyle:'auto',buttonEffects:'on'})")
   for font in ('SimSun','Microsoft YaHei','KaiTi'):
    for size in (10,12,14,15,18,20,24):
     for width,height in ((1280,800),(1440,900),(1706,1018)):
      js('document.documentElement.style.setProperty("--font",'+json.dumps('"Times New Roman","'+font+'"')+');document.documentElement.style.setProperty("--body-size",'+json.dumps(str(size)+'px')+')');w.resize(width,height);pause(200)
      for direction in (1,-1):
       scroll=js("(()=>{const s=document.querySelector('.m17-chart-viewport');s.scrollTop+="+str(direction*350)+";return s.scrollTop})()");assert (scroll>0 if direction==1 else scroll==0),scroll;pause(60);pixels(f'{font}-{size}-{width}-{direction}')
   pixels('final').save(str(out/'final-visible.png'))
   assert not report['errors'],report['errors'];report['success']=True
  finally:
   w.closing=True;w.close();pause(800);app.quit()
   (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({'success':report['success'],'report':str(out/'report.json')}),flush=True)

if __name__=='__main__':main()
