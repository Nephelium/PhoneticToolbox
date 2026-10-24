"""Actual Qt bridge, Cocoa Dock image and isolated per-icon persistence on Mac."""
import base64,ctypes,hashlib,json,sys,time
from pathlib import Path

def main(output):
    from PyQt6.QtCore import QEventLoop,QTimer,Qt
    from PyQt6.QtGui import QImage,QIcon,QPixmap,QCursor
    from PyQt6.QtWidgets import QApplication
    from ptb_desktop.web_assets import register_scheme
    from ptb_desktop.host import Workbench
    from ptb_desktop.icon_preferences import IconPreferences,decode_icon
    from ptb_worker.native.darwin import bundle_root
    from ptb_worker.local_workspace import prepare_workspace
    root=bundle_root();out=Path(output);out.mkdir(parents=True,exist_ok=False)
    report=dict(success=False,frozen=bool(getattr(sys,'frozen',False)),checks=[],scope='Actual Mac Qt WebChannel and Cocoa application Dock image; isolated preferences; user icons unchanged; no global Dock settings changed')
    register_scheme();QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    app=QApplication.instance() or QApplication(sys.argv[:1]);app.setApplicationName('PhoneticToolbox-macOS-Icon-QA');app.setQuitOnLastWindowClosed(False)
    def pause(ms):
        loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
    def js(code):
        result=[];loop=QEventLoop();timer=QTimer();timer.setSingleShot(True);timer.timeout.connect(loop.quit)
        window.page.runJavaScript(code,lambda value:(result.append(value),loop.quit()))
        timer.start(15000);loop.exec();timer.stop()
        if not result:raise TimeoutError('Qt JavaScript callback')
        return result[0]
    def wait(predicate,timeout=30):
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            if predicate():return
            pause(50)
        raise AssertionError('Qt icon condition timed out')
    def bounds(image):
        assert not image.isNull()
        points=[(x,y) for y in range(image.height()) for x in range(image.width()) if image.pixelColor(x,y).alpha()>=32]
        return [min(x for x,y in points),min(y for x,y in points),max(x for x,y in points),max(y for x,y in points)]
    def native_image():
        # Read the NSImage set by Qt in this exact owned GUI process.
        objc=ctypes.CDLL('/usr/lib/libobjc.A.dylib')
        objc.objc_getClass.argtypes=[ctypes.c_char_p];objc.objc_getClass.restype=ctypes.c_void_p
        objc.sel_registerName.argtypes=[ctypes.c_char_p];objc.sel_registerName.restype=ctypes.c_void_p
        address=ctypes.cast(objc.objc_msgSend,ctypes.c_void_p).value
        send=ctypes.CFUNCTYPE(ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p)(address)
        size=ctypes.CFUNCTYPE(ctypes.c_ulong,ctypes.c_void_p,ctypes.c_void_p)(address)
        sel=lambda name:objc.sel_registerName(name.encode())
        nsapp=send(objc.objc_getClass(b'NSApplication'),sel('sharedApplication'))
        image=send(nsapp,sel('applicationIconImage'));data=send(image,sel('TIFFRepresentation'))
        n=size(data,sel('length'));assert 0<n<20_000_000
        raw=ctypes.string_at(send(data,sel('bytes')),n)
        return QImage.fromData(raw)
    def check_native(tag,expected_percent=None):
        pause(150);image=native_image();b=bounds(image);image.save(str(out/(tag+'.png')))
        ratio=max(b[2]-b[0]+1,b[3]-b[1]+1)/max(image.width(),image.height())
        if expected_percent is not None:assert abs(ratio-expected_percent/100)<.035,(tag,ratio)
        report.setdefault('native_dock_images',[]).append(dict(tag=tag,size=[image.width(),image.height()],bounds=b,occupied_ratio=ratio))
    def create(name):
        db,files=prepare_workspace(out/'scratch'/name,root/'backend/migrations')
        w=Workbench(root/'frontend/dist',test=True,vocal_profile=out/'scratch'/'vocal-profile',jobs_path=db,local_files_root=files,reaper_binary=root/'resources/research/reaper',vocal_resources=root/'resources/vocal_tract/native')
        w.icon_preferences=IconPreferences(out/'scratch'/'icon-preferences',w.windowIcon(),w,root/'frontend/dist')
        w.icon_preferences.apply(w.icon_preferences.read());w.show();w.raise_();QCursor.setPos(w.mapToGlobal(w.rect().center()))
        return w
    def settings():
        wait(lambda:js('!![...document.querySelectorAll(".sidebar-bottom button")].find(e=>e.title==="设置")'))
        js('[...document.querySelectorAll(".sidebar-bottom button")].find(e=>e.title==="设置").click()')
        wait(lambda:js('!![...document.querySelectorAll("button")].find(e=>e.textContent.trim()==="图标")'))
        js('[...document.querySelectorAll("button")].find(e=>e.textContent.trim()==="图标").click()')
        wait(lambda:js('!!document.querySelector("#dock-icon-size")'))
    def select(identity):
        if identity.startswith('custom-'):code='[...document.querySelectorAll(".custom-icon-item .icon-choice")][0].click()'
        else:code='document.querySelector('+json.dumps('[data-icon-id="'+identity+'"]')+').click()'
        js(code);wait(lambda:window.icon_preferences.read()['selected']==identity)
        wait(lambda:js('!document.querySelector("#dock-icon-size").disabled'))
    def resize(percent):
        before=window.icon_preferences.path.read_bytes()
        js('(()=>{const e=document.querySelector("#dock-icon-size");e.value='+str(percent)+';e.dispatchEvent(new Event("input",{bubbles:true}));})()')
        pause(80);assert window.icon_preferences.path.read_bytes()==before,'Preview must not save on every pointer movement'
        js('document.querySelector("#dock-icon-size").dispatchEvent(new Event("change",{bubbles:true}))')
        wait(lambda:window.icon_preferences.read()['dockSize']['value']==percent)
        wait(lambda:js('!document.querySelector("#dock-icon-size").disabled'))
    def close():
        window.accept_close();window.page.deleteLater();window.view.deleteLater();pause(200)
    window=None
    try:
        window=create('workspace-1');store=window.icon_preferences
        # Public synthetic shape only; never read the user's imported avatar.
        from PyQt6.QtCore import QBuffer,QIODevice
        from PyQt6.QtGui import QPainter,QColor
        image=QImage(256,256,QImage.Format.Format_ARGB32);image.fill(0);p=QPainter(image);p.fillRect(50,12,156,232,QColor('#aabbdd'));p.end()
        buf=QBuffer();buf.open(QIODevice.OpenModeFlag.WriteOnly);image.save(buf,'PNG');source='data:image/png;base64,'+base64.b64encode(bytes(buf.data())).decode()
        custom=store.request(dict(action='import',name='透明头像验收.png',dataUrl=source))
        custom_id=custom['selected'];custom_data=custom['custom'][0]['dataUrl'];store.request(dict(action='select',id='default'))
        settings();assert js('document.querySelector("#dock-icon-size").value')=='82';check_native('builtin-default-82',82)
        resize(64);check_native('builtin-size-64',64)
        js('document.documentElement.dataset.theme="light"');pause(120);window.grab().save(str(out/'settings-light.png'))
        js('document.documentElement.dataset.theme="dark"');pause(120);window.grab().save(str(out/'settings-dark.png'))
        select('builtin-033');assert store.read()['dockSize']['value']==82
        resize(100);select('default');assert store.read()['dockSize']['value']==64
        js('document.querySelector(".dock-icon-size-setting button").click()');wait(lambda:store.read()['dockSize']['value']==82)
        select(custom_id);assert store.read()['dockSize']['value']==100
        expected=QPixmap.fromImage(decode_icon(custom_data)[1]).toImage()
        assert bounds(window.windowIcon().pixmap(256,256).toImage())==bounds(expected)
        check_native('custom-original-100');resize(70);check_native('custom-size-70')
        select('default');select(custom_id);assert store.read()['dockSize']['value']==70
        assert store.read()['custom'][0]['dataUrl']==custom_data
        report['checks']+=['Cocoa_builtin_default_gutter','real_slider_preview_then_save','independent_icon_sizes_and_default_restore','custom_transparency_preserved_and_resize']
        close();window=None;window=create('workspace-2');settings()
        assert window.icon_preferences.read()['selected']==custom_id
        wait(lambda:js('document.querySelector("#dock-icon-size").value')=='70')
        js('document.documentElement.dataset.theme="dark"');pause(120);window.grab().save(str(out/'settings-reopened.png'))
        report['checks'].append('actual_new_workbench_reopens_saved_icon_and_size')
        report.update(success=True,fixture='generated transparent rectangle; no user avatar read')
    finally:
        if window is not None:close()
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return 0

if __name__=='__main__':raise SystemExit(main(*sys.argv[1:]))
