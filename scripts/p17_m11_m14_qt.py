"""Actual maximized Qt host, natural M12 copy and no writes to source recordings."""
import argparse
import os
import hashlib
import json
from pathlib import Path
import shutil
import time
from uuid import uuid4
from PyQt6.QtCore import Qt,QTimer
from PyQt6.QtWidgets import QApplication,QFileDialog
from phonetic_core.annotation import parse_document
from ptb_desktop.host import register_scheme,Workbench
from ptb_worker.local_workspace import prepare_workspace

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--visible',action='store_true');parser.add_argument('--m11-controls',action='store_true');args=parser.parse_args()
    out=ROOT/'output/validation/p17/qt-c'/uuid4().hex;inputs=out/'inputs';inputs.mkdir(parents=True)
    source=json.loads((ROOT/'output/validation/p17/natural-inventory.json').read_text('utf8'))['selected']['medium']
    wav=Path(source['path']);grid=wav.with_suffix('.TextGrid');originals=[]
    for p in [wav,grid]:
        originals.append((p,hashlib.sha256(p.read_bytes()).hexdigest()));shutil.copyfile(p,inputs/p.name)
    raw=grid.read_bytes();expected=parse_document(raw.decode('utf-16' if raw[:2] in (b'\xff\xfe',b'\xfe\xff') else 'utf-8-sig'))
    db,cache=prepare_workspace(out/'workspace',ROOT/'backend/migrations')
    picks=[];pick_counts={}
    if args.m11_controls:
        components=out/'components';components.mkdir();shutil.copyfile(ROOT/'output/validation/m11/qt-642b638d008c497abb576e5db63dcbe7/components/registry.json',components/'registry.json');os.environ['PTB_M11_COMPONENT_ROOT']=str(components)
        registry=json.loads((components/'registry.json').read_text('utf8'))
    register_scheme();app=QApplication(['P17 Qt M11-M14']);window=Workbench(ROOT/'frontend/dist',test=True,jobs_path=db,local_files_root=cache)
    window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen,not args.visible);window.showMaximized()
    QFileDialog.getExistingDirectory=lambda *a,**k:str(inputs)
    QFileDialog.getSaveFileName=lambda parent,title,name,filter,**kwargs:(str(out/Path(name).name),filter)
    if args.m11_controls:
        def choose_directory(parent,title,*a,**k):
            pick_counts[title]=pick_counts.get(title,0)+1
            if 'runtime' in title:value=registry['runtimes'][0]['path']
            elif ('MFA' in title or '结果' in title) and pick_counts[title]==1:value=''
            else:value=str(inputs if ('corpus' in title or '音频' in title) else out)
            picks.append({'kind':'directory','title':title,'cancelled':not bool(value)});return value
        def choose_file(parent,title,*a,**k):
            value=registry['models'][0]['model'] if 'model' in title else registry['models'][0]['dictionary'] if 'dictionary' in title else ''
            picks.append({'kind':'file','title':title,'cancelled':not bool(value)});return value,''
        QFileDialog.getExistingDirectory=choose_directory;QFileDialog.getOpenFileName=choose_file
    click=lambda text:"[...document.querySelectorAll('button')].find(b=>b.offsetParent&&b.textContent.trim()==="+json.dumps(text,ensure_ascii=False)+")?.click()"
    nav=lambda text:"[...document.querySelectorAll('.nav-item')].find(b=>b.textContent.includes("+json.dumps(text)+"))?.click()"
    stages=[]
    for id,name,selector in [('M11','MFA 自动标注','[aria-label="MFA 自动标注工作区"]'),('M12','语音标注对齐','.annotation-page'),('M13','普通话转 IPA','.mandarin-ipa-page'),('M14','音系归纳','[aria-label="音系归纳工作区"]')]:
        stages.extend([("document.querySelector('.host-badge')?.textContent==='本地桌面'",nav(name)),('!!document.querySelector('+json.dumps(selector)+')','capture:'+id)])
    stages.extend([('true',nav('普通话转 IPA')),('!!document.querySelector(".mandarin-ipa-page")',"(()=>{const e=document.querySelector('[aria-label=\"待转换汉字文本\"]');e.value='银行花';e.dispatchEvent(new Event('input',{bubbles:true}));})()"),('document.querySelectorAll(".m13-mapped").length===3','capture:M13-loaded'),('true',nav('语音标注对齐')),('!!document.querySelector(".annotation-page")',click('选择语料文件夹')),('document.querySelectorAll(".annotation-file-list button").length===1','document.querySelector(".annotation-file-list button").click()'),("document.querySelector('.annotation-page')?.getAttribute('aria-busy')==='false'&&!!document.querySelector('.annotation-grid')",'capture:M12-loaded'),('true',click('保存 TextGrid')),("document.body.textContent.includes('已保存：')",'capture:M12-saved')])
    if args.m11_controls:
        stages.extend([('true',nav('MFA 自动标注')),('!!document.querySelector(".component-details")',"document.querySelector('.component-details summary').click()"),('!!document.querySelector(".component-details[open]")','capture:M11-component-info'),('true',"document.querySelector('.component-details summary').click()"),('true',click('选择语料目录')),('true',click('选择语料目录')),('document.querySelectorAll(".corpus-list li").length===1',click('选择输出目录')),('true',click('选择输出目录')),('true',click('组件安装与环境检查'))])
        for label in ['已有环境 / auto_alignment','声学模型 ZIP','配套词典','离线组件 ZIP','可信发布清单 JSON']:
            action="[...document.querySelectorAll('.component-manager .resource-line')].find(e=>e.textContent.includes("+json.dumps(label)+")).querySelector('button').click()"
            stages.append(("document.querySelector('.mfa-page')?.getAttribute('aria-busy')==='false'&&!!document.querySelector('.component-manager')",action))
        stages.append(("document.querySelector('.mfa-page')?.getAttribute('aria-busy')==='false'",'capture:M11-native-selections'))
    report={'native_dialog_returns':picks,'success':False,'stages':[],'geometry':{},'screen':{'size':[app.primaryScreen().size().width(),app.primaryScreen().size().height()],'available':[app.primaryScreen().availableGeometry().width(),app.primaryScreen().availableGeometry().height()],'dpr':app.primaryScreen().devicePixelRatio()},'maximized':window.isMaximized(),'visibility':'visible' if args.visible else 'WA_DontShowOnScreen','frontend':'production dist; final P17 source build, not frozen EXE','frontend_index_sha256':hashlib.sha256((ROOT/'frontend/dist/index.html').read_bytes()).hexdigest()}
    index=0;pending=False;started=time.monotonic();done=False
    def finish(ok,error=None):
        nonlocal done
        if done:return
        done=True;timer.stop();report.update(success=ok,error=error)
        try:
            if ok:
                saved=parse_document((inputs/(wav.stem+'_自动保存.TextGrid')).read_text('utf8'))
                assert [t['name'] for t in saved['tiers']]==[t['name'] for t in expected['tiers']]
                assert sum(len(t.get('intervals',[])) for t in saved['tiers'])==sum(len(t.get('intervals',[])) for t in expected['tiers'])
                assert all(hashlib.sha256(p.read_bytes()).hexdigest()==sha for p,sha in originals)
                report['readback']={'tiers':len(saved['tiers']),'all_intervals_preserved':True,'originals_unchanged':True}
        except Exception as exc:report.update(success=False,error=str(exc))
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf8');window.closing=True;window.close()
    def tick():
        nonlocal pending,index
        if done or pending:return
        if time.monotonic()-started>90:finish(False,'timeout '+str(index));return
        pending=True
        def ready(value):
            nonlocal index,pending
            if not value:pending=False;return
            condition,action=stages[index];index+=1;report['stages'].append(index)
            if action.startswith('capture:'):
                label=action.split(':')[1]
                def capture():
                    window.view.grab().save(str(out/(label+'.png')))
                    def geometry(data):
                        nonlocal pending
                        report['geometry'][label]=data;pending=False
                        if index==len(stages):finish(True)
                    window.page.runJavaScript("(()=>({inner:[innerWidth,innerHeight],screen:[screen.width,screen.height],dpr:devicePixelRatio,frames:[...document.querySelectorAll('.module-frame')].filter(e=>e.offsetParent).map(e=>({class:e.className,client:e.clientHeight,scroll:e.scrollHeight}))}))()",geometry)
                window.page.runJavaScript("requestAnimationFrame(()=>requestAnimationFrame(()=>window.__p17Paint=true))")
                QTimer.singleShot(350,capture)
            else:window.page.runJavaScript(action);pending=False
        window.page.runJavaScript(stages[index][0],ready)
    timer=QTimer();timer.timeout.connect(tick);timer.start(120);app.exec();print(out/'report.json')
    return 0 if report['success'] else 1

if __name__=='__main__':raise SystemExit(main())
