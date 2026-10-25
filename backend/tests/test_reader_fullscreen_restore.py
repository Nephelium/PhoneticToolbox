"""Execute the actual host handler against delayed native-window transitions."""
import ast
from enum import IntFlag
from pathlib import Path
from types import SimpleNamespace,ModuleType
import sys


class State(IntFlag):
    WindowNoState=0
    WindowMaximized=2
    WindowFullScreen=4


def handler(monkeypatch,platform='darwin',previous=State.WindowNoState):
    source=(Path(__file__).resolve().parents[2]/'desktop/src/ptb_desktop/host.py').read_text('utf8')
    tree=ast.parse(source);node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='reader_fullscreen')
    queue=[]
    class Window:
        closing=False
        state=previous
        native=bool(previous & State.WindowFullScreen)
        saves=0
        restores=0
        toggles=0
        def windowState(self):return self.state
        def saveGeometry(self):self.saves+=1;return b'original geometry'
        def showFullScreen(self):self.state=State.WindowFullScreen;self.native=True
        def isFullScreen(self):return bool(self.state & State.WindowFullScreen)
        def showNormal(self):
            self.state=State.WindowNoState
            if platform!='darwin':self.native=False
        def showMaximized(self):self.state=State.WindowMaximized;self.native=False
        def winId(self):return 42
        def restoreGeometry(self,value):
            assert not self.native,'Geometry must wait for Cocoa to leave its native Space'
            assert value==b'original geometry';self.restores+=1
    window=Window()
    def native(handle,*,leave=False):
        assert handle==42
        active=window.native
        if leave:window.toggles+=1;window.native=False
        return active
    fake=ModuleType('ptb_desktop.window_chrome');fake.macos_window_fullscreen=native;monkeypatch.setitem(sys.modules,'ptb_desktop.window_chrome',fake)
    namespace={'__package__':'ptb_desktop','sys':SimpleNamespace(platform=platform),'Qt':SimpleNamespace(WindowState=State),'QTimer':SimpleNamespace(singleShot=lambda delay,callback:queue.append(callback))}
    exec(compile(ast.get_source_segment(source,node),'<actual host full-screen handler>','exec'),namespace)
    def request(on):
        accepted=[];namespace['reader_fullscreen'](window,SimpleNamespace(toggleOn=lambda:on,accept=lambda:accepted.append(True)));assert accepted==[True]
    def settle():
        for _ in range(400):
            if not queue:return
            queue.pop(0)()
        raise AssertionError('Full-screen restoration did not settle')
    return window,request,settle,queue


def test_native_enter_finishing_after_qt_exit_does_not_restore_fullscreen_geometry(monkeypatch):
    window,request,settle,queue=handler(monkeypatch)
    request(True);request(False)
    assert window.native and not window.isFullScreen() and not window.restores
    queue.pop(0)();assert not window.restores
    settle();assert not window.native and window.restores==1 and window.toggles==1 and not window._reader_return_active


def test_quick_reentry_cancels_stale_restore_without_overwriting_original_geometry(monkeypatch):
    window,request,settle,queue=handler(monkeypatch)
    request(True);request(False);request(True);settle()
    assert window.native and window.restores==0 and window.saves==1
    request(False);settle();assert window.restores==1 and not window.native


def test_user_preexisting_native_fullscreen_is_preserved(monkeypatch):
    window,request,settle,queue=handler(monkeypatch,previous=State.WindowFullScreen)
    request(True);request(False);settle();assert window.native and not window.toggles and not window.restores


def test_windows_normal_and_maximized_states_still_restore(monkeypatch):
    for previous in [State.WindowNoState,State.WindowMaximized]:
        window,request,settle,queue=handler(monkeypatch,'win32',previous)
        request(True);request(False);settle();assert window.state==previous and not window.native
        assert window.restores==(previous==State.WindowNoState)


def test_cocoa_adapter_checks_native_window_mask_and_only_toggles_fullscreen(monkeypatch):
    import importlib.util
    path=Path(__file__).resolve().parents[2]/'desktop/src/ptb_desktop/window_chrome.py'
    spec=importlib.util.spec_from_file_location('owned_window_chrome_test',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.sys=SimpleNamespace(platform='darwin');calls=[];state=[1<<14]
    monkeypatch.setattr(module,'_cocoa_window_messages',lambda:(None,lambda view,selector:99,lambda window,selector:state[0],lambda *args:calls.append(args),11,12,13))
    assert module.macos_window_fullscreen(42) and not calls
    assert module.macos_window_fullscreen(42,leave=True) and calls==[(99,13,None)]
    state[0]=1<<10;assert not module.macos_window_fullscreen(42,leave=True);assert len(calls)==1
    module.sys=SimpleNamespace(platform='win32');monkeypatch.setattr(module,'_cocoa_window_messages',lambda:(_ for _ in ()).throw(AssertionError('Cocoa must not load on Windows')));assert not module.macos_window_fullscreen(42,leave=True)
