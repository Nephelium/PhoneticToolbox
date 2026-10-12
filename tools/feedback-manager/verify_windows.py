"""Actual frozen EXE, outside checkout, private HTTPS fixture and owned state."""
from pathlib import Path
import base64
import ctypes
from ctypes import wintypes
from datetime import datetime,timedelta,timezone
import hashlib
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import ipaddress
import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from workbench_source import bind_sources
from verification_artifacts import verification_run,write_json
bind_sources(ROOT)
from ptb_api.feedback import FeedbackService,create_feedback_app
from ptb_desktop.feedback import FeedbackClient
import uvicorn

def port():
    with socket.socket() as s:s.bind(('127.0.0.1',0));return s.getsockname()[1]

def owned_pids(pid):
    class Entry(ctypes.Structure):
        _fields_=[('size',wintypes.DWORD),('cnt',wintypes.DWORD),('pid',wintypes.DWORD),('heap',ctypes.c_size_t),('module',wintypes.DWORD),('threads',wintypes.DWORD),('parent',wintypes.DWORD),('priority',wintypes.LONG),('flags',wintypes.DWORD),('name',wintypes.WCHAR*260)]
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateToolhelp32Snapshot.restype=wintypes.HANDLE
    kernel.Process32FirstW.argtypes=[wintypes.HANDLE,ctypes.POINTER(Entry)]
    kernel.Process32NextW.argtypes=[wintypes.HANDLE,ctypes.POINTER(Entry)]
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    handle=kernel.CreateToolhelp32Snapshot(2,0);entry=Entry();entry.size=ctypes.sizeof(entry);rows=[]
    okay=kernel.Process32FirstW(handle,ctypes.byref(entry))
    while okay:rows.append((entry.pid,entry.parent));okay=kernel.Process32NextW(handle,ctypes.byref(entry))
    kernel.CloseHandle(handle);result={pid}
    for _ in range(6):result.update(child for child,parent in rows if parent in result)
    return result

def owned_window(pid):
    user=ctypes.WinDLL('user32',use_last_error=True);owners=owned_pids(pid);windows=[]
    callback=ctypes.WINFUNCTYPE(wintypes.BOOL,wintypes.HWND,wintypes.LPARAM)
    user.GetWindowThreadProcessId.argtypes=[wintypes.HWND,ctypes.POINTER(wintypes.DWORD)]
    user.GetWindowTextW.argtypes=[wintypes.HWND,wintypes.LPWSTR,ctypes.c_int]
    def check(hwnd,data):
        owner=wintypes.DWORD();user.GetWindowThreadProcessId(hwnd,ctypes.byref(owner))
        if owner.value in owners:
            title=ctypes.create_unicode_buffer(512);user.GetWindowTextW(hwnd,title,512)
            if title.value.startswith('PhoneticToolbox 反馈管理'):windows.append(hwnd)
        return True
    user.EnumWindows(callback(check),0)
    if len(windows)!=1:raise AssertionError('Expected one owned manager window')
    return user,windows[0]

def main():
    with verification_run('feedback-manager','windows-final','Final one-file EXE alone outside checkout; Windows-only PATH; owned DPAPI profile; private TLS certificate; actual native bridge and tray') as (out,scratch):
        print(json.dumps({'out':str(out)}),flush=True)
        # Existing private helper dependency is used only for test certificates.
        sys.path.insert(0,str(Path.home()/'.codex/server-access/packages'))
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes,serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        key=rsa.generate_private_key(public_exponent=65537,key_size=2048)
        name=x509.Name([x509.NameAttribute(NameOID.COMMON_NAME,'Owned feedback fixture')])
        now=datetime.now(timezone.utc)
        certificate=x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key()).serial_number(x509.random_serial_number()).not_valid_before(now-timedelta(minutes=1)).not_valid_after(now+timedelta(days=1)).add_extension(x509.BasicConstraints(ca=True,path_length=None),critical=True).add_extension(x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address('127.0.0.1'))]),critical=False).sign(key,hashes.SHA256())
        cert=scratch/'tls.pem';private=scratch/'tls.key'
        cert.write_bytes(certificate.public_bytes(serialization.Encoding.PEM));private.write_bytes(key.private_bytes(serialization.Encoding.PEM,serialization.PrivateFormat.PKCS8,serialization.NoEncryption()))
        master='owned-fixture-master-key-12345678901234567890'
        service=FeedbackService(scratch/'feedback.sqlite',secret='owned-fixture-hmac-key-12345678901234567890',admin_token=master,allow_unverified=True)
        identity=service.open_device('fixture@example.org',secrets.token_urlsafe(32))
        from PIL import Image
        png=scratch/'fixture.png';Image.new('RGB',(100,60),(0,120,126)).save(png)
        image={'data':base64.b64encode(png.read_bytes()).decode()}
        service.send(identity['owner'],'user','Windows fixture feedback',[image],'windows-initial-fixture')
        api_port,debug_port=port(),port()
        base=f'https://127.0.0.1:{api_port}/api/v1/feedback'
        api=uvicorn.Server(uvicorn.Config(create_feedback_app(service),host='127.0.0.1',port=api_port,ssl_certfile=str(cert),ssl_keyfile=str(private),access_log=False,log_level='critical',timeout_graceful_shutdown=1))
        thread=threading.Thread(target=api.run,daemon=True);thread.start()
        destination=Path('D:/PTB-Feedback-Manager-QA')/uuid.uuid4().hex;destination.mkdir(parents=True,exist_ok=False)
        write_json(destination/'owner.json',{'scope':'Owned final manager EXE acceptance','state':'active'})
        executable=destination/'反馈收件箱.exe';source=ROOT/'output/feedback-manager/1.0.1/反馈收件箱.exe';shutil.copyfile(source,executable)
        profile=destination/'profile';temp=destination/'temp';profile.mkdir();temp.mkdir()
        client=FeedbackClient(profile/'PhoneticToolbox/v3/feedback-manager',admin=True,base_url=base)
        client.state={'token':master,'email':'','cursor':0};client.save()
        env={k:v for k,v in os.environ.items() if not k.startswith(('PTB_','PYTHON','CONDA','_PYI')) and k not in ['VIRTUAL_ENV','QT_PLUGIN_PATH','QT_QPA_PLATFORM_PLUGIN_PATH','QTWEBENGINEPROCESS_PATH','QTWEBENGINE_RESOURCES_PATH','QTWEBENGINE_LOCALES_PATH']}
        env.update(LOCALAPPDATA=str(profile),TEMP=str(temp),TMP=str(temp),PATH=os.pathsep.join([os.environ['SystemRoot']+'/System32',os.environ['SystemRoot']]),QTWEBENGINE_REMOTE_DEBUGGING=str(debug_port),QTWEBENGINE_CHROMIUM_FLAGS='--disable-gpu --mute-audio',SSL_CERT_FILE=str(cert),PTB_OWNED_BOOTSTRAP_LOG=str(out/'bootstrap-error.log'))
        process=None;controller=None
        try:
            end=time.monotonic()+15
            while not api.started and time.monotonic()<end:time.sleep(.05)
            assert api.started
            log=(out/'windows-process.log').open('wb')
            process=subprocess.Popen([str(executable)],cwd=destination,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            class Controller(BaseHTTPRequestHandler):
                def log_message(self,*args):pass
                def do_POST(self):
                    try:
                        raw=self.rfile.read(min(int(self.headers.get('Content-Length','0')),4096));body=json.loads(raw or b'{}')
                        if self.path=='/received':
                            last=service.messages(identity['owner'])['messages'][-1];value={'text':last['text'],'images':len(last['images'])}
                        elif self.path=='/new-feedback':
                            service.send(identity['owner'],'user','Immediate server event',[],'windows-immediate-fixture');value={'ok':True}
                        elif self.path=='/pair':
                            device=service.admin_devices.complete_pairing(body['code'],'fixture phone');service.authenticate(device['token'],True);service.admin_devices.revoke(device['deviceId']);value={'paired':True}
                        elif self.path=='/window-lifecycle':
                            user,hwnd=owned_window(process.pid);user.IsWindowVisible.argtypes=[wintypes.HWND];user.PostMessageW.argtypes=[wintypes.HWND,wintypes.UINT,wintypes.WPARAM,wintypes.LPARAM]
                            user.PostMessageW(hwnd,0x0010,0,0);time.sleep(.6);hidden=not user.IsWindowVisible(hwnd)
                            second=subprocess.run([str(executable)],cwd=destination,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=30,creationflags=subprocess.CREATE_NO_WINDOW)
                            time.sleep(.5);value={'trayHidden':hidden,'reactivated':bool(user.IsWindowVisible(hwnd)),'secondInstanceExited':second.returncode==0}
                        else:raise ValueError('Unknown test operation')
                        payload=json.dumps(value).encode();self.send_response(200);self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
                    except Exception:self.send_error(500,'Fixture operation failed')
            controller=ThreadingHTTPServer(('127.0.0.1',0),Controller);ct=threading.Thread(target=controller.serve_forever,daemon=True);ct.start()
            plan=scratch/'plan.json';write_json(plan,{'port':debug_port,'control':f'http://127.0.0.1:{controller.server_port}','out':str(out),'image':str(png)})
            subprocess.run([r'C:/Program Files/nodejs/node.exe',str(ROOT/'tools/feedback-manager/verify_windows.mjs'),str(plan)],check=True,timeout=220)
            assert process.wait(20)==0
            assert not list(temp.glob('_MEI*'))
            result={'success':True,'exeSha256':hashlib.sha256(source.read_bytes()).hexdigest(),'outsideCheckout':True,'windowsOnlyPath':True,'finalBytes':source.stat().st_size,'cleanExit':True,'tempBootDirsRemain':False,'scope':'Windows 11 actual final EXE; no physical separate computer or tray toast observation'}
            write_json(out/'windows-final.json',result);print(json.dumps(result),flush=True)
        finally:
            if process and process.poll() is None:
                # Stop only this QA launch and its recorded descendants.
                kernel=ctypes.WinDLL('kernel32');kernel.OpenProcess.restype=wintypes.HANDLE;kernel.TerminateProcess.argtypes=[wintypes.HANDLE,wintypes.UINT];kernel.CloseHandle.argtypes=[wintypes.HANDLE]
                for pid in sorted(owned_pids(process.pid),reverse=True):
                    handle=kernel.OpenProcess(1,False,pid)
                    if handle:kernel.TerminateProcess(handle,1);kernel.CloseHandle(handle)
                process.wait(10)
            if controller:controller.shutdown();controller.server_close()
            api.should_exit=True;thread.join(8);client.close()
            write_json(destination/'owner.json',{'scope':'Owned final manager EXE acceptance','state':'stopped','report':str(out)})
if __name__=='__main__':main()
