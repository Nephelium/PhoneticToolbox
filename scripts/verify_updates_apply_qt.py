"""Actual maximized hidden Qt update-close wiring; local fixture and dry helper only."""
import json
import os
from pathlib import Path
import sys
import threading
import time
from uuid import uuid4

ROOT=Path(__file__).resolve().parents[1]
for folder in ('packages/phonetic_core/src','backend/src','desktop/src'):
    sys.path.insert(0,str(ROOT/folder))
os.environ['PYTHONPATH']=os.pathsep.join(str(ROOT/p) for p in ('packages/phonetic_core/src','backend/src','desktop/src'))
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS','--mute-audio --disable-gpu --disable-gpu-compositing')
from PyQt6.QtCore import QEventLoop,QTimer,Qt,QThread
from PyQt6.QtWidgets import QApplication,QFileDialog
from ptb_desktop.host import Workbench,register_scheme
from ptb_desktop.updates import Package,_atomic_json


def main():
    out=ROOT/'output/validation/updates-apply'/('qt-'+uuid4().hex);out.mkdir(parents=True)
    QFileDialog.getSaveFileName=lambda parent,title,name,kind,**kwargs:(str(out/Path(name).name),kind)
    register_scheme();app=QApplication(['PTB-update-close-owned-QA']);app.setApplicationName('PTB-updates-owned-QA')
    w=Workbench(ROOT/'frontend/dist',test=True,jobs_path=None,vocal_profile=out/'vocal')
    w.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen,True);w.showMaximized()
    report={'success':False,'scope':'actual Windows maximized hidden Qt; source apply disabled; owned synthetic download metadata; dry prepare/helper adapters only in this test process; no physical capture or install','checks':[],'responses':[],'helper_calls':[]}
    service=w.updates_bridge.service
    assert service.preferences()['applyAvailable'] is False
    def prepare(*args):
        assert QThread.currentThread()!=app.thread()
        request=out/'dry-apply'/uuid4().hex/'request.json';plan={'token':uuid4().hex}
        _atomic_json(request,{'fixture':True});return request,plan
    def launch(request,plan):
        assert QThread.currentThread()==app.thread()
        report['helper_calls'].append({'request':str(request),'window_closing':w.closing})
    w.update_coordinator.prepare=prepare;w.update_coordinator.launch=launch
    service.apply_handler=w.update_coordinator.apply
    path=service.root/'downloads'/('a'*32)/'local-fixture.zip';path.parent.mkdir(parents=True);body=b'owned-Qt-close-fixture';path.write_bytes(body)
    import hashlib
    service._downloads['a'*32]=(path,Package('portable',path.name,'https://www.phonetictoolbox.com/local-fixture',len(body),hashlib.sha256(body).hexdigest(),'server'))
    service._download_versions['a'*32]='3.0.0-preview.2'
    w.updates_bridge.ready.connect(lambda rid,payload:report['responses'].append({'id':rid,**json.loads(payload)}))
    close_requests=[]
    real_close=w.request_update_close
    w.request_update_close=lambda:close_requests.append('requested')
    def pause(ms=50):
        loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
    def js(code):
        loop=QEventLoop();box=[];w.page.runJavaScript(code,lambda value:(box.append(value),loop.quit()));QTimer.singleShot(5000,loop.quit);loop.exec()
        assert box,'JS timeout';return box[0]
    def until(code):
        end=time.monotonic()+30
        while time.monotonic()<end:
            if js(code):return
            pause()
        raise AssertionError(code+' '+str(js('document.body.innerText.slice(-1800)')))
    def button(text):
        assert js('(()=>{const text='+json.dumps(text)+';const e=[...document.querySelectorAll("button")].find(e=>e.offsetParent&&(e.textContent.trim()===text||e.getAttribute("aria-label")===text||(e.matches(".nav-item")&&e.querySelector(":scope>span:not([aria-hidden])")?.textContent.trim()===text)));if(!e||e.disabled)return false;e.click();return true})()'),text
        pause()
    def fill(label,value):
        assert js('(()=>{const e=document.querySelector('+json.dumps('[aria-label="'+label+'"]')+');if(!e)return false;e.value='+json.dumps(value)+';e.dispatchEvent(new Event("input",{bubbles:true}));return true})()')
        pause()
    def apply_request(name):
        w.updates_bridge.request(name,json.dumps({'operation':'apply','args':{'downloadId':'a'*32,'confirmed':True}}))
        pause(150)
    def response(name):
        end=time.monotonic()+10
        while time.monotonic()<end:
            found=[r for r in report['responses'] if r['id']==name]
            if found:return found[-1]
            pause()
        raise AssertionError('native response '+name)
    try:
        until('document.querySelector(".host-badge")?.textContent==="本地桌面"')
        assert w.isMaximized();report['checks'].append('source apply unavailable; maximized native workbench handshake')
        button('汉字转国际音标');until('!!document.querySelector("[aria-label=待转换汉字文本]")');fill('待转换汉字文本','秋叶 ŋ')
        apply_request('cancel-draft');until('document.body.innerText.includes("保存转换草稿？")')
        assert not report['helper_calls'];button('取消关闭');assert response('cancel-draft')['error']['code']=='APPLY_CANCELLED'
        assert js('document.querySelector("[aria-label=待转换汉字文本]").value==="秋叶 ŋ"')
        report['checks'].append('cancel existing dirty-close modal preserves text and never launches helper')
        apply_request('save-draft');until('document.body.innerText.includes("保存转换草稿？")');button('保存草稿并关闭')
        assert response('save-draft')['value']['started'];pause(250)
        assert close_requests==['requested'] and not report['helper_calls']
        assert js('Object.values(localStorage).some(v=>v.includes("秋叶 ŋ"))')
        w.update_coordinator.cancel();pause();report['checks'].append('existing module save writes actual localStorage; close request alone never launches helper')
        # An actual local experiment remains protected even while paused.
        button('感知实验');until('!!document.querySelector("[aria-label=实验范式]")')
        js(r'(()=>{const dt=new DataTransfer();dt.items.add(new File(["Qt 刺激 a̠ ŋ"],"刺激1.txt",{type:"text/plain"}));const e=document.querySelector("[aria-label=\"导入刺激 X\"]");e.files=dt.files;e.dispatchEvent(new Event("change",{bubbles:true}));})()')
        until('document.querySelectorAll(".m15-asset").length===1&&document.querySelector(".perception-page")?.getAttribute("aria-busy")==="false"')
        button('预检与试音');until('document.body.textContent.includes("预检通过")')
        assert js('(()=>{for(const text of ["已听见试音，设备正确","已暂停其他录制、播放及重型任务"]){const e=[...document.querySelectorAll("label")].find(l=>l.textContent.trim()===text)?.querySelector("input");if(!e||e.disabled)return false;if(!e.checked)e.click();}return true})()')
        button('填写被试信息');fill('姓名/编号','Qt 更新保护');js('document.querySelector("input[type=radio][value=女]").click()');button('建立独立会话');until('document.querySelector(".m15-run")?.dataset.phase==="intro"')
        apply_request('experiment');assert response('experiment')['error']['code']=='APPLY_CANCELLED';assert not report['helper_calls']
        report['checks'].append('actual synthetic perception session blocks update without disposal or helper')
        # Keep experimental test data intact; final accepted native closure is
        # exercised after explicitly declining this update and closing test UI.
        w.update_coordinator.cancel();button('结束实验');until('!document.querySelector(".m15-run")')
        button('关闭 感知实验');until('!!document.querySelector(".dialog-actions")')
        button('放弃草稿并关闭')
        w.request_update_close=real_close
        apply_request('accepted-close')
        end=time.monotonic()+10
        while w.isVisible() and time.monotonic()<end:pause()
        assert not w.isVisible() and len(report['helper_calls'])==1 and report['helper_calls'][0]['window_closing']
        report['checks'].append('native window accepted close invokes dry helper once only after all guards')
        report['success']=True
    finally:
        if w.isVisible():w.update_coordinator.cancel();w.accept_close()
        w.service.close();w.provider.close();w.vocal.close();w.updates_bridge.close()
        report['service_exit']=w.service.exit_code
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf-8')
        print(json.dumps({'output':str(out),'success':report['success'],'checks':len(report['checks'])},ensure_ascii=False))
    return 0


if __name__=='__main__':raise SystemExit(main())
