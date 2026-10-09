"""Capture only the owned native preparation window, positioned offscreen."""
import ctypes as c
from ctypes import wintypes as w
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'desktop/src'))
from ptb_desktop.startup_progress import PreparationWindow
from PIL import Image


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    parser.add_argument('--icon',type=Path)
    args=parser.parse_args()
    out=args.output.absolute();out.mkdir(parents=True,exist_ok=False)
    if not getattr(sys,'frozen',False) and not args.icon:
        parser.error('Source preview requires --icon with the same packaged application artwork')
    os.environ.pop('PTB_OWNED_BOOTSTRAP_LOG',None)
    user=c.WinDLL('user32');gdi=c.WinDLL('gdi32')
    kernel=c.WinDLL('kernel32')
    user.SetThreadDpiAwarenessContext.argtypes=(c.c_void_p,)
    user.SetThreadDpiAwarenessContext.restype=c.c_void_p
    prior_dpi=user.SetThreadDpiAwarenessContext(c.c_void_p(-4))
    kernel.GetCurrentProcess.restype=w.HANDLE
    user.GetGuiResources.argtypes=(w.HANDLE,w.DWORD)
    user.SendMessageW.argtypes=(w.HWND,w.UINT,w.WPARAM,w.LPARAM)
    user.SendMessageW.restype=c.c_ssize_t
    user.GetWindowDC.argtypes=(w.HWND,);user.GetWindowDC.restype=w.HDC
    user.GetWindowRect.argtypes=(w.HWND,c.POINTER(w.RECT))
    user.GetWindowRgn.argtypes=(w.HWND,w.HRGN)
    user.PrintWindow.argtypes=(w.HWND,w.HDC,w.UINT)
    user.ReleaseDC.argtypes=(w.HWND,w.HDC)
    gdi.CreateCompatibleDC.argtypes=(w.HDC,);gdi.CreateCompatibleDC.restype=w.HDC
    gdi.CreateCompatibleBitmap.argtypes=(w.HDC,c.c_int,c.c_int);gdi.CreateCompatibleBitmap.restype=w.HBITMAP
    gdi.SelectObject.argtypes=(w.HDC,w.HGDIOBJ);gdi.SelectObject.restype=w.HGDIOBJ
    gdi.DeleteObject.argtypes=(w.HGDIOBJ,);gdi.DeleteDC.argtypes=(w.HDC,)
    gdi.CreateRectRgn.argtypes=(c.c_int,c.c_int,c.c_int,c.c_int);gdi.CreateRectRgn.restype=w.HRGN
    gdi.GetRgnBox.argtypes=(w.HRGN,c.POINTER(w.RECT))
    gdi.PtInRegion.argtypes=(w.HRGN,c.c_int,c.c_int)
    class Header(c.Structure):
        _fields_=[('size',w.DWORD),('width',w.LONG),('height',w.LONG),('planes',w.WORD),('bits',w.WORD),('compression',w.DWORD),('image',w.DWORD),('x',w.LONG),('y',w.LONG),('used',w.DWORD),('important',w.DWORD)]
    gdi.GetDIBits.argtypes=(w.HDC,w.HBITMAP,w.UINT,w.UINT,c.c_void_p,c.c_void_p,w.UINT)
    def capture(progress,path):
        rectangle=w.RECT();assert user.GetWindowRect(progress.hwnd,c.byref(rectangle))
        width,height=rectangle.right-rectangle.left,rectangle.bottom-rectangle.top
        region=gdi.CreateRectRgn(0,0,0,0)
        try:
            assert user.GetWindowRgn(progress.hwnd,region)
            bounds=w.RECT();assert gdi.GetRgnBox(region,c.byref(bounds))
            assert (bounds.right,bounds.bottom)==(width,height),(bounds.right,bounds.bottom,width,height)
            assert not gdi.PtInRegion(region,0,0)
            assert gdi.PtInRegion(region,width//2,height-1)
        finally:gdi.DeleteObject(region)
        dc=user.GetWindowDC(progress.hwnd);target=gdi.CreateCompatibleDC(dc)
        bitmap=gdi.CreateCompatibleBitmap(dc,width,height);old=gdi.SelectObject(target,bitmap)
        try:
            assert user.PrintWindow(progress.hwnd,target,2)
            gdi.SelectObject(target,old)
            header=Header(c.sizeof(Header),width,-height,1,32,0,0,0,0,0,0)
            buffer=c.create_string_buffer(width*height*4)
            assert gdi.GetDIBits(target,bitmap,0,height,buffer,c.byref(header),0)==height
            picture=Image.frombuffer('RGB',(width,height),buffer,'raw','BGRX',0,1)
            picture.save(path)
            assert picture.crop((width//5,height//10,width*9//10,height//3)).getextrema()[0][0]<100
        finally:
            gdi.DeleteObject(bitmap);gdi.DeleteDC(target);user.ReleaseDC(progress.hwnd,dc)
        return width,height
    counts=lambda:[user.GetGuiResources(kernel.GetCurrentProcess(),kind) for kind in (0,1)]
    cold_counts=counts();cases=[]
    try:
        for dpi in (96,120,144,192,240):
            progress=PreparationWindow(offscreen=True,dpi=dpi,icon_path=args.icon)
            try:
                progress.update('science',0,100)
                assert progress.visible.wait(10) and progress.hwnd,repr(progress.failure)
                for stage,done,total in [('science',0,100),('science',43,100),('science',97,100),
                                         ('host',100,100),('apps',45,100),('waiting',0,0),('repair',0,0),
                                         ('starting',0,0),('checking',0,0),('verify-science',23,100),
                                         ('verify-host',60,100),('verify-apps',100,100),('loading',0,0),('slow-loading',0,0)]:
                    progress.update(stage,done,total);time.sleep(.15)
                    dimensions=capture(progress,out/f'{dpi}-{stage}-{done}.png')
                    assert dimensions==(round(620*dpi/96),round(312*dpi/96)),dimensions
                    assert not progress.failure,repr(progress.failure)
                    assert progress.resolved_fonts=={'KaiTi':'楷体','Times New Roman':'Times New Roman'},progress.resolved_fonts
                    cases.append({'dpi':dpi,'stage':stage,'percent':done,'dimensions':dimensions,'fonts':progress.resolved_fonts})
            finally:progress.close()
        # Exercise the production DPI-change path and the new rounded region.
        progress=PreparationWindow(offscreen=True,icon_path=args.icon)
        try:
            progress.update('apps',75,100)
            assert progress.visible.wait(10) and progress.hwnd,repr(progress.failure)
            for dpi in (144,192,96):
                bounds=w.RECT(-10000,0,-10000+round(620*dpi/96),round(312*dpi/96))
                user.SendMessageW(progress.hwnd,0x2E0,dpi|(dpi<<16),c.addressof(bounds))
                time.sleep(.1)
                dimensions=capture(progress,out/f'dpi-change-{dpi}.png')
                assert dimensions==(round(620*dpi/96),round(312*dpi/96)),dimensions
                assert progress.dpi==dpi and not progress.failure,repr(progress.failure)
        finally:progress.close()
        # USER/GDI initialize process-wide font/icon/theme caches lazily. Compare
        # steady-state cycles after warming and waiting for deferred destruction,
        # rather than treating that first-use allocation as a per-window leak.
        for _ in range(3):
            progress=PreparationWindow(offscreen=True,dpi=96,icon_path=args.icon)
            progress.update('science',50,100)
            assert progress.visible.wait(10) and progress.hwnd,repr(progress.failure)
            progress.close();time.sleep(.6)
        before=counts()
        for _ in range(15):
            progress=PreparationWindow(offscreen=True,dpi=96,icon_path=args.icon)
            try:
                progress.update('science',50,100)
                assert progress.visible.wait(10) and progress.hwnd,repr(progress.failure)
                for value in range(101):progress.update('science',value,100)
            finally:progress.close()
        time.sleep(.6)
        after=counts()
        # Deferred Windows cleanup may reduce the count. Only growth indicates
        # unreleased objects; a decrease is not a drawing or resource failure.
        assert all(end<=start for start,end in zip(before,after)),{'before':before,'after':after}
    finally:
        if prior_dpi:user.SetThreadDpiAwarenessContext(prior_dpi)
    (out/'report.json').write_text(json.dumps({'success':True,'scope':'actual Win32 window positioned offscreen; PrintWindow capture',
        'frozen':bool(getattr(sys,'frozen',False)),'cases':cases,'simulated_dpi_changes':[144,192,96],
        'gdi_user_objects_cold':cold_counts,'gdi_user_objects_before':before,'gdi_user_objects_after':after,
        'repeated_window_cycles':15},ensure_ascii=False,indent=2),'utf8')
    print(str(out))


if __name__=='__main__':
    try:main()
    except Exception:
        target=Path(sys.argv[1]) if len(sys.argv)>1 else Path.cwd()
        target.mkdir(parents=True,exist_ok=True)
        (target/'failure.txt').write_text(traceback.format_exc(),'utf8')
        raise SystemExit(1)
