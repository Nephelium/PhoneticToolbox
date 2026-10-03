"""P19-R3 built actual Qt: UI buttons and M10 theme bridge, no device actions."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS','--disable-gpu --mute-audio')
import json,time
from pathlib import Path
from uuid import uuid4
from PyQt6.QtCore import QEventLoop,QTimer
from PyQt6.QtWidgets import QApplication
from ptb_desktop.host import Workbench,register_scheme
ROOT=Path(__file__).resolve().parents[1]

def main():
 out=ROOT/'output/validation/p19-r3'/('qt-'+uuid4().hex);out.mkdir(parents=True)
 register_scheme();app=QApplication(['P19-R3-owned-offscreen-QA'])
 w=Workbench(ROOT/'frontend/dist',test=True,vocal_profile=out/'vocal');w.resize(1920,1080);w.show()
 report={'success':False,'checks':[],'scope':'Windows built actual Qt offscreen; no physical devices/DPI or scientific acceptance'}
 def pause(ms=180):
  loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
 def js(code):
  loop=QEventLoop();values=[];w.page.runJavaScript(code,lambda v:(values.append(v),loop.quit()));QTimer.singleShot(5000,loop.quit);loop.exec()
  if not values:raise RuntimeError('JS callback timeout')
  return values[0]
 def until(code):
  end=time.monotonic()+35
  while time.monotonic()<end:
   if js(code):return
   pause()
  raise RuntimeError('UI timeout: '+code)
 def click(label):
  assert js('(()=>{const b=[...document.querySelectorAll("button")].find(b=>b.offsetParent&&b.textContent.trim()==='+json.dumps(label,ensure_ascii=False)+');if(!b)return false;b.click();return true;})()'),label
  pause()
 def set_buttons(mode,effects):
  click('设置')
  js('(()=>{const s=document.querySelector("#button-style-choice");s.value='+json.dumps(mode)+';s.dispatchEvent(new Event("change",{bubbles:true}));const b=document.querySelector(".button-effects-toggle input");b.checked='+json.dumps(effects)+';b.dispatchEvent(new Event("change",{bubbles:true}));})()');pause()
 def audit(selector,frame=False):
  doc='document.querySelector("iframe").contentDocument' if frame else 'document'
  return js('(()=>{const doc='+doc+',root=doc.documentElement,b=doc.querySelector('+json.dumps(selector)+'),s=getComputedStyle(b),r=getComputedStyle(root),p=doc.createElement("span");doc.body.append(p);const color=k=>{p.style.color=r.getPropertyValue(k);return getComputedStyle(p).color;};const result={mode:root.dataset.buttonStyle,effects:root.dataset.buttonEffects,bg:s.backgroundColor,color:s.color,shadow:s.boxShadow,accent:color("--accent"),onAccent:color("--on-accent"),panel:color("--panel"),outline:s.outlineStyle};p.remove();return result;})()')
 try:
  until('!!document.querySelector(".app-shell")');click('设置');until('!!document.querySelector("#button-style-choice")')
  assert js('document.querySelector("#button-style-choice").value')=='auto'
  for theme in ['light','dark']:
   click('设置')
   click('浅色' if theme=='light' else '深色')
   for mode in ['auto','all','plain']:
    for effects in [True,False]:
     set_buttons(mode,effects);click('语音合成')
     result=audit('.m06-page .workbench-right button.primary')
     assert result['mode']==mode and result['effects']==('on' if effects else 'off'),result
     assert result['bg']==result['panel' if mode=='plain' else 'accent'],result
     auxiliary=audit('.m06-page .module-toolbar-actions button')
     assert auxiliary['shadow']!='none' if effects else auxiliary['shadow']=='none',auxiliary
     report['checks'].append({'module':'M06','theme':theme,'mode':mode,'effects':effects})
  set_buttons('auto',True);click('语音合成');w.view.grab().save(str(out/'m06-auto.png'))
  click('声道工作台');until('!!document.querySelector("iframe")?.contentDocument?.querySelector("#liveButton")')
  until('document.querySelector("iframe").contentDocument.documentElement.dataset.buttonStyle==="auto"')
  for mode in ['auto','all','plain']:
   for effects in [True,False]:
    set_buttons(mode,effects);click('声道工作台')
    until('document.querySelector("iframe").contentDocument.documentElement.dataset.buttonStyle==='+json.dumps(mode))
    result=audit('#oneSecondButton',True);assert result['bg']==result['panel' if mode=='plain' else 'accent'],result
    auxiliary=audit('#aboutButton',True);assert auxiliary['bg']==auxiliary['accent'] if mode=='all' else auxiliary['bg']!=auxiliary['accent'],auxiliary
    assert auxiliary['shadow']!='none' if effects else auxiliary['shadow']=='none',auxiliary
    report['checks'].append({'module':'M10','mode':mode,'effects':effects})
  set_buttons('plain',False);click('声道工作台')
  js('document.querySelector("iframe").contentDocument.querySelector("#liveButton").classList.add("active")')
  assert audit('#liveButton',True)['outline']=='solid'
  w.view.grab().save(str(out/'m10-plain.png'));report['checks'].append('M10 live active indicator preserved in plain mode without shadow')
  for width,height in [(1440,900),(1024,768)]:
   w.resize(width,height);pause();click('设置');until('!!document.querySelector("#button-style-choice")')
   assert js('document.documentElement.scrollWidth<=document.documentElement.clientWidth+2')
   js('document.querySelector("#button-style-choice").scrollIntoView()');w.view.grab().save(str(out/f'settings-{width}.png'))
   report['checks'].append({'settings_width':width,'height':height})
  report['success']=True
 except Exception as error:
  report['error']=str(error);w.view.grab().save(str(out/'failed.png'));raise
 finally:
  (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(out,flush=True)
  w.closing=True;w.close();app.processEvents()

if __name__=='__main__':main()
