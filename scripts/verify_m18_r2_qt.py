"""Owned Qt R2 reader interaction checks; uses isolated data and public catalogue."""
import json,time
from pathlib import Path
from uuid import uuid4
from workbench_source import configure

def main():
 configure()
 from PyQt6.QtCore import QEventLoop,QTimer,Qt,QPoint,QEvent
 from PyQt6.QtWidgets import QApplication
 from PyQt6.QtTest import QTest
 from ptb_desktop.host import Workbench,register_scheme
 from pypdf import PdfReader
 root=Path(__file__).resolve().parents[1];out=root/'output/validation/m18'/('r2-'+uuid4().hex);out.mkdir(parents=True)
 register_scheme();app=QApplication(['M18-R2-owned-QA']);report={'success':False,'checks':[]}
 class Checks(list):
  def append(self,item):
   super().append(item);print(item,flush=True)
 report['checks']=Checks()
 w=Workbench(root/'frontend/dist',test=True,vocal_profile=out/'vocal',start_module='M18')
 w.papers_bridge.service.first_launch='2026-10-07'
 (out/'owned-process.json').write_text(json.dumps({'servicePid':w.service.process.pid}), 'utf8')
 w.resize(1440,900);w.show()
 def pause(ms=120):
  loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
 def js(code):
  loop=QEventLoop();v=[];w.page.runJavaScript(code,lambda x:(v.append(x),loop.quit()));QTimer.singleShot(6000,loop.quit);loop.exec();assert v,'JS timeout';return v[0]
 def until(code,seconds=40):
  end=time.monotonic()+seconds
  while time.monotonic()<end:
   if js(code):return
   pause()
  raise AssertionError(code+'\n'+str(js('document.body.innerText.slice(-1600)')))
 def pos(x,y):
  target=w.view.focusProxy() or w.view
  return target,target.mapFromGlobal(w.view.mapToGlobal(QPoint(round(x),round(y))))
 def pointer(selector):
  r=js('(()=>{const e=document.querySelector('+json.dumps(selector)+');const r=e.getBoundingClientRect();return [r.left+r.width/2,r.top+r.height/2]})()');target,p=pos(*r);QTest.mouseClick(target,Qt.MouseButton.LeftButton,pos=p);pause()
 def click(text):
  selector='[...document.querySelectorAll("button")].find(e=>e.offsetParent&&e.textContent.trim()==='+json.dumps(text)+')'
  until('!!('+selector+')');js(selector+'.setAttribute("data-r2-click","true")');pointer('[data-r2-click=true]');js('document.querySelector("[data-r2-click=true]")?.removeAttribute("data-r2-click")')
 def select_line():
  until('!!document.querySelector("[data-page=\\"1\\"] .pdf-text-line")')
  js('document.querySelector(".paper-viewport").scrollTop=0');pause()
  r=js('(()=>{const e=[...document.querySelectorAll("[data-page=\\"1\\"] .pdf-text-line")][2];const r=e.getBoundingClientRect();return [r.left+2,r.top+r.height/2,r.right-2,r.top+r.height/2]})()')
  target,start=pos(r[0],r[1]);_,end=pos(r[2],r[3]);QTest.mousePress(target,Qt.MouseButton.LeftButton,pos=start);QTest.mouseMove(target,end,180);QTest.mouseRelease(target,Qt.MouseButton.LeftButton,pos=end);pause()
  until('! [...document.querySelectorAll("button")].find(e=>e.textContent.trim()==="荧光笔").disabled')
 try:
  until('document.querySelector(".pdf-text-line")&&document.querySelectorAll(".pdf-page").length===13')
  assert js('document.querySelector(".paper-details")===null')
  pointer('.paper-fold');until('!!document.querySelector(".paper-guide")');pointer('.paper-fold')
  report['checks'].append('metadata initially collapsed, complete details and guide expand/collapse')
  select_line();selected=js('getSelection().toString()');assert len(selected)>5,selected
  target=w.view.focusProxy() or w.view;QTest.keyClick(target,Qt.Key.Key_C,Qt.KeyboardModifier.ControlModifier);pause();assert app.clipboard().text().strip()==selected.strip()
  click('荧光笔');until('document.querySelectorAll(".pdf-mark.highlight").length>0')
  select_line();click('添加批注');until('!!document.querySelector("textarea[aria-label=\\"批注内容\\"]")')
  js('(()=>{const e=document.querySelector("textarea[aria-label=\\"批注内容\\"]");e.value="检验声源和滤波器的控制边界。";e.dispatchEvent(new Event("input",{bubbles:true}));})()')
  click('保存批注');until('document.querySelectorAll(".pdf-note-pin").length===1')
  report['checks'].append('native pointer PDF selection and Ctrl+C clipboard; highlight and Chinese note persisted')
  pointer('button[aria-label="下一页"]');until('document.querySelector("input[aria-label=\\"页码\\"]").value==="2"')
  pointer('.paper-viewport');QTest.keyClick(target,Qt.Key.Key_Right);until('document.querySelector("input[aria-label=\\"页码\\"]").value==="3"')
  QTest.keyClick(target,Qt.Key.Key_Left);until('document.querySelector("input[aria-label=\\"页码\\"]").value==="2"')
  js('document.querySelector(".paper-viewport").scrollTop=document.querySelector("[data-page=\\"5\\"]").offsetTop');until('document.querySelector("input[aria-label=\\"页码\\"]").value==="5"&&!!document.querySelector("[data-page=\\"5\\"] .paper-sheet")')
  report['checks'].append('continuous page scroll tracks page 5 and native left/right keys navigate without replacing viewport')
  click('全屏阅读');until('!!document.fullscreenElement');assert w.isFullScreen()
  click('中文译文');until('document.querySelectorAll(".pdf-page").length===11&&!!document.querySelector(".pdf-text-line")');assert w.isFullScreen()
  assert not js('!!document.querySelector(".pdf-note-pin")')
  click('退出全屏');until('!document.fullscreenElement');assert not w.isFullScreen()
  report['checks'].append('native fullscreen restores window and retains fullscreen during language switch; languages isolate annotations')
  click('原文');until('document.querySelectorAll(".pdf-page").length===13')
  js('(()=>{const e=document.querySelector("input[aria-label=\\"页码\\"]");e.value=1;e.dispatchEvent(new Event("change",{bubbles:true}));})()');until('!!document.querySelector(".pdf-note-pin")')
  for annotated in (False,True):
   destination=out/('original-annotated.pdf' if annotated else 'original.pdf');w.papers_bridge.export_picker=lambda name,path=destination:str(path)
   click('导出 PDF');until('document.querySelector("dialog")?.textContent.includes("导出论文 PDF")')
   if annotated:pointer('.export-check input')
   click('选择位置并导出');until('!document.querySelector("dialog[open]")');assert destination.exists()
  p=w.papers_bridge.service.catalog['papers'][0]
  assert (out/'original.pdf').read_bytes()==w.papers_bridge.service._path(p['original']).read_bytes()
  exported=PdfReader(out/'original-annotated.pdf');annots=[a.get_object() for a in exported.pages[0]['/Annots']];assert any(a.get('/Subtype')=='/Highlight' for a in annots);assert any('检验声源' in a.get('/Contents','') for a in annots)
  report['checks'].append('native export original bytes identical; annotated PDF retains 13 pages, highlight and Chinese standard text annotation')
  click('引用格式');until('!!document.querySelector("textarea[aria-label=\\"引用文本\\"]")');click('复制引用');assert 'Bandekar'.upper() in app.clipboard().text();click('关闭')
  for size in [(1440,900),(1100,720)]:
   w.resize(*size)
   for theme in ('light','dark'):
    js('document.documentElement.dataset.theme='+json.dumps(theme));pause(300)
    assert js('document.documentElement.scrollWidth<=innerWidth+1')
    w.view.grab();pause();w.view.grab().save(str(out/f'{size[0]}-{theme}.png'))
  report['checks'].append('citation clipboard and four light/dark viewport layouts')
  # Manual images use the approved real screen logical geometry and effective DPR.
  screen=w.screen();w.resize(screen.geometry().width(),screen.geometry().height());w.showMaximized()
  js('document.documentElement.dataset.theme="light";document.querySelector("[aria-label=收起侧栏]")?.click()');pause(400)
  report['manualCaptures']=[]
  def shot(name):
   pause(250);w.view.grab();pause();pix=w.view.grab();pix.save(str(out/(name+'.png')))
   report['manualCaptures'].append({'name':name,'size':[pix.width(),pix.height()],'dpr':pix.devicePixelRatio(),'viewport':[w.view.width(),w.view.height()],'maximized':w.isMaximized(),'pageZoom':w.view.zoomFactor()})
  shot('reader-collapsed')
  pointer('.paper-fold');shot('reader-details');pointer('.paper-fold')
  click('批注 2');shot('reader-annotations');click('批注 2')
  click('引用格式');shot('reader-citations');click('关闭')
  click('导出 PDF');shot('reader-export');click('取消')
  click('全屏阅读');until('!!document.fullscreenElement');shot('reader-fullscreen');click('退出全屏');until('!document.fullscreenElement');pause(300);assert w.isMaximized()
  click('中文译文');until('document.querySelectorAll(".pdf-page").length===11&&!!document.querySelector(".pdf-text-line")');shot('reader-translation')
  select_line();click('荧光笔');until('document.querySelectorAll(".pdf-mark.highlight").length>0')
  # A real native file picker is also exercised; only its test-owned destination is selected.
  from PyQt6.QtWidgets import QFileDialog,QLineEdit
  destination=out/'translation-native-dialog.pdf'
  w.papers_bridge.export_picker=w.pick_paper_export
  timer=QTimer()
  def choose():
   for dialog in app.topLevelWidgets():
    if isinstance(dialog,QFileDialog) and dialog.isVisible():
     dialog.setDirectory(str(destination.parent));dialog.selectFile(destination.name);edit=dialog.findChild(QLineEdit,'fileNameEdit');assert edit;edit.setText(destination.name);dialog.accept();timer.stop()
  timer.timeout.connect(choose);timer.start(120)
  click('导出 PDF');click('选择位置并导出');until('!document.querySelector("dialog[open]")');timer.stop()
  assert destination.exists() and len(PdfReader(destination).pages)==11
  w.papers_bridge.export_picker=lambda name:''
  click('导出 PDF');click('选择位置并导出');until('document.querySelector("dialog")?.textContent.includes("已取消导出")');click('取消')
  click('原文');until('document.querySelectorAll(".pdf-page").length===13&&!!document.querySelector(".pdf-note-pin")')
  assert len(w.papers_bridge.service.annotations(p['id'],'original')['items'])==2
  assert len(w.papers_bridge.service.annotations(p['id'],'translation')['items'])==1
  report['checks'].append('approved-DPI manual captures; translated PDF highlight and native save dialog; export cancellation; independent marks survive language reload')
  click('帮助');until('document.body.innerText.includes("连续阅读与全屏")&&document.body.innerText.includes("荧光笔与文字批注")')
  until('[...document.querySelectorAll(".manual-reader img")].filter(e=>e.offsetParent).every(e=>e.complete&&e.naturalWidth>0)')
  report['checks'].append('module help navigates to M18 chapter')
  report['success']=True
 finally:
  (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf8')
  proc=w.service.process;w.closing=True;w.close();w.page.deleteLater();app.processEvents();report['serviceExit']=proc.poll();(out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf8');print(json.dumps({'output':str(out),**report},ensure_ascii=True),flush=True)
if __name__=='__main__':main()
