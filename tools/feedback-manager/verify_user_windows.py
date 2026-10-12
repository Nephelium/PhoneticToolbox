"""Actual final main EXE feedback UI, owned HTTPS fixture and protected user profile."""
from pathlib import Path
import hashlib,ipaddress,json,os,secrets,shutil,socket,subprocess,sys,threading,time,uuid
from datetime import datetime,timedelta,timezone
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from verify_windows import ROOT,port,owned_pids,FeedbackService,FeedbackClient,create_feedback_app,uvicorn,verification_run,write_json

def main():
 with verification_run('feedback-manager','user-final-r3','Final main EXE alone outside checkout; private HTTPS and DPAPI user; no real SMTP or production messages') as (out,scratch):
  print(json.dumps({'out':str(out)}),flush=True)
  sys.path.insert(0,str(Path.home()/'.codex/server-access/packages'))
  from cryptography import x509
  from cryptography.x509.oid import NameOID
  from cryptography.hazmat.primitives import hashes,serialization
  from cryptography.hazmat.primitives.asymmetric import rsa
  key=rsa.generate_private_key(public_exponent=65537,key_size=2048);name=x509.Name([x509.NameAttribute(NameOID.COMMON_NAME,'Owned user feedback fixture')]);now=datetime.now(timezone.utc)
  certificate=x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key()).serial_number(x509.random_serial_number()).not_valid_before(now-timedelta(minutes=1)).not_valid_after(now+timedelta(days=1)).add_extension(x509.BasicConstraints(ca=True,path_length=None),critical=True).add_extension(x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address('127.0.0.1'))]),critical=False).sign(key,hashes.SHA256())
  cert=scratch/'tls.pem';private=scratch/'tls.key';cert.write_bytes(certificate.public_bytes(serialization.Encoding.PEM));private.write_bytes(key.private_bytes(serialization.Encoding.PEM,serialization.PrivateFormat.PKCS8,serialization.NoEncryption()))
  class Mail:
   def __init__(self):self.sent=[]
   def send(self,*args):self.sent.append(args)
  mail=Mail();service=FeedbackService(scratch/'feedback.sqlite',secret='owned-user-fixture-key-12345678901234567890',admin_token='owned-user-admin-key-12345678901234567890',mailer=mail,allow_unverified=True)
  identity=service.open_device('fixture@example.org',secrets.token_urlsafe(32))
  with service.db() as d:d.execute('UPDATE identities SET email_verified=1 WHERE id=?',(identity['owner'],))
  identity['emailVerified']=True
  api_port,debug_port=port(),port();base=f'https://127.0.0.1:{api_port}/api/v1/feedback'
  api=uvicorn.Server(uvicorn.Config(create_feedback_app(service),host='127.0.0.1',port=api_port,ssl_certfile=str(cert),ssl_keyfile=str(private),access_log=False,log_level='critical',timeout_graceful_shutdown=1));thread=threading.Thread(target=api.run,daemon=True);thread.start()
  destination=Path('D:/PTB-Feedback-Manager-QA')/uuid.uuid4().hex;destination.mkdir();write_json(destination/'owner.json',{'scope':'Owned final user EXE acceptance','state':'active'})
  source=ROOT/'dist/PhoneticToolbox-v3-Feedback-20261011-R3/PhoneticToolbox-windows-x64.exe';executable=destination/source.name;shutil.copyfile(source,executable)
  profile=destination/'profile';temp=destination/'temp';profile.mkdir();temp.mkdir();updates=profile/'PhoneticToolbox/v3/updates';updates.mkdir(parents=True);(updates/'state.json').write_text('{"autoCheck":false}','utf-8')
  client=FeedbackClient(profile/'PhoneticToolbox/v3/feedback',base_url=base);client.state.update(identity);client.save()
  env={k:v for k,v in os.environ.items() if not k.startswith(('PTB_','PYTHON','CONDA','_PYI')) and k not in ['VIRTUAL_ENV','QT_PLUGIN_PATH','QT_QPA_PLATFORM_PLUGIN_PATH','QTWEBENGINEPROCESS_PATH','QTWEBENGINE_RESOURCES_PATH','QTWEBENGINE_LOCALES_PATH']}
  env.update(LOCALAPPDATA=str(profile),TEMP=str(temp),TMP=str(temp),PATH=os.pathsep.join([os.environ['SystemRoot']+'/System32',os.environ['SystemRoot']]),QT_QPA_PLATFORM='offscreen',QTWEBENGINE_REMOTE_DEBUGGING=str(debug_port),QTWEBENGINE_CHROMIUM_FLAGS='--disable-gpu --mute-audio',SSL_CERT_FILE=str(cert),PTB_OWNED_BOOTSTRAP_LOG=str(out/'bootstrap-error.log'))
  process=controller=None
  try:
   end=time.monotonic()+15
   while not api.started and time.monotonic()<end:time.sleep(.05)
   assert api.started
   log=(out/'process.log').open('wb');process=subprocess.Popen([str(executable)],cwd=destination,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
   class Controller(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def do_POST(self):
     try:
      body=json.loads(self.rfile.read(min(int(self.headers.get('Content-Length','0')),4096)) or b'{}')
      if self.path=='/reply':service.send(identity['owner'],'admin',body['text'],[],body['id']);value={'ok':True}
      elif self.path=='/state':
       c=service.messages(identity['owner']);value={'mailNotifications':c['mailNotifications'],'mailCount':len(mail.sent),'messages':c['messages']}
      else:raise ValueError('Unknown owned operation')
      raw=json.dumps(value).encode();self.send_response(200);self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
     except Exception:self.send_error(500)
   controller=ThreadingHTTPServer(('127.0.0.1',0),Controller);threading.Thread(target=controller.serve_forever,daemon=True).start()
   plan=scratch/'plan.json';write_json(plan,{'port':debug_port,'control':f'http://127.0.0.1:{controller.server_port}','out':str(out)})
   subprocess.run([r'C:/Program Files/nodejs/node.exe',str(ROOT/'tools/feedback-manager/verify_user_windows.mjs'),str(plan)],check=True,timeout=600)
   assert process.wait(30)==0
   write_json(out/'final.json',{'success':True,'exeSha256':hashlib.sha256(source.read_bytes()).hexdigest(),'finalBytes':source.stat().st_size,'outsideCheckout':True,'windowsOnlyPath':True,'cleanExit':True,'remainingBootDirectories':[str(p) for p in temp.glob('_MEI*')],'scope':'Windows 11 final main EXE and synthetic HTTPS user feedback; no real user mail'})
  finally:
   if process and process.poll() is None:
    import ctypes
    from ctypes import wintypes
    kernel=ctypes.WinDLL('kernel32');kernel.OpenProcess.restype=wintypes.HANDLE;kernel.TerminateProcess.argtypes=[wintypes.HANDLE,wintypes.UINT];kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    for pid in sorted(owned_pids(process.pid),reverse=True):
     handle=kernel.OpenProcess(1,False,pid)
     if handle:kernel.TerminateProcess(handle,1);kernel.CloseHandle(handle)
    process.wait(10)
   if controller:controller.shutdown();controller.server_close()
   api.should_exit=True;thread.join(8);client.close();write_json(destination/'owner.json',{'scope':'Owned final user EXE acceptance','state':'stopped','report':str(out)})
if __name__=='__main__':main()
