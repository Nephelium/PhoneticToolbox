"""P19-R4 built Qt settings and M10 color bridge, no hardware actions."""
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
 out=ROOT/'output/validation/p19-r4'/('qt-'+uuid4().hex);out.mkdir(parents=True)
 register_scheme();app=QApplication(['P19-R4-owned-offscreen-QA'])
 w=Workbench(ROOT/'frontend/dist',test=True,vocal_profile=out/'vocal');w.resize(1440,900);w.show()
 report={'success':False,'checks':[],'scope':'Windows built actual Qt offscreen; settings and M10 bridge only'}
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
 def color(mode):
  js('(()=>{const s=document.querySelector("#waveform-color-choice");s.value='+json.dumps(mode)+';s.dispatchEvent(new Event("change",{bubbles:true}));})()');pause()
 def custom(value):
  js('(()=>{const s=document.querySelector(".wave-color-controls input:not([type=color])");s.value='+json.dumps(value)+';s.dispatchEvent(new Event("input",{bubbles:true}));s.dispatchEvent(new Event("change",{bubbles:true}));})()');pause()
 try:
  until('!!document.querySelector(".app-shell")');click('设置');until('!!document.querySelector("#waveform-color-choice")')
  for theme in ['light','dark']:
   click('浅色' if theme=='light' else '深色')
   for mode in ['blue','theme','custom']:
    color(mode)
    if mode=='custom':custom('#b52f91')
    result=js('(()=>{const root=document.documentElement,p=document.createElement("span");p.style.color="var(--waveform-color)";document.body.append(p);const expected=getComputedStyle(p).color;p.remove();return {color:getComputedStyle(document.querySelector(".wave-color-preview path")).stroke,expected,mode:root.dataset.waveformColorMode};})()')
    assert result['color']==result['expected'] and result['mode']==mode,result
    report['checks'].append({'theme':theme,'mode':mode,'preview':result['color']})
  custom('bad');assert js('getComputedStyle(document.querySelector(".wave-color-preview path")).stroke')=='rgb(181, 47, 145)'
  assert js('document.querySelector(".appearance-settings [role=status]").textContent.includes("有效的 HEX")')
  custom('#b52f91');js('document.querySelector("#waveform-color-choice").scrollIntoView()');w.view.grab().save(str(out/'settings-custom.png'))
  click('声道工作台');until('!!document.querySelector("iframe")?.contentDocument?.querySelector("#liveButton")')
  until('getComputedStyle(document.querySelector("iframe").contentDocument.documentElement).getPropertyValue("--waveform-color").trim()==="#b52f91"')
  for theme in ['light','dark']:
   click('设置');color('theme');click('浅色' if theme=='light' else '深色');click('声道工作台')
   until('getComputedStyle(document.querySelector("iframe").contentDocument.documentElement).getPropertyValue("--waveform-color").trim()===getComputedStyle(document.documentElement).getPropertyValue("--accent").trim()')
   report['checks'].append({'bridge_theme':theme})
  click('设置');color('custom');assert js('document.querySelector(".wave-color-controls input:not([type=color])").value')=='#b52f91'
  w.resize(1024,768);pause();js('document.querySelector("#waveform-color-choice").scrollIntoView()')
  assert js('document.documentElement.scrollWidth<=document.documentElement.clientWidth+2');w.view.grab().save(str(out/'settings-narrow.png'))
  report['checks'].append('custom bridge, invalid HEX keeps prior color, retained custom after switching and 1024-wide settings')
  report['success']=True
 except Exception as error:
  report['error']=str(error);w.view.grab().save(str(out/'failed.png'));raise
 finally:
  (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(out,flush=True)
  w.closing=True;w.close();app.processEvents()

if __name__=='__main__':main()
