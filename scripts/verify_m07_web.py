"""Fresh owned PG + authenticated API + actual Chrome. No existing database."""
import sys,os,json,secrets,socket,subprocess,time,urllib.request,zipfile,hashlib
from pathlib import Path
from uuid import uuid4
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'backend/tests'))
from test_p07_policy_postgres import cluster,legacy,migrate
from ptb_api.account_store import PostgresAccountStore
os.environ['PTB_POLICY_FRESH_PG']='1'
def main():
 out=ROOT/'output/validation/m07/web'/uuid4().hex;out.mkdir(parents=True);print(out,flush=True)
 cg=cluster.__wrapped__();c=next(cg);eg=legacy.__wrapped__(c);e=next(eg);server=worker=None
 try:
  migrate(e);accounts=PostgresAccountStore(e.dsn);people=[]
  for i in range(2):
   username='m07_'+uuid4().hex[:12];password=secrets.token_urlsafe(32);user=accounts.create_user(username,password);project=accounts.create_project(str(user['id']),'M07 网页验证');people.append(dict(username=username,password=password,id=str(user['id']),project=str(project['id'])))
  with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
  origin=f'http://127.0.0.1:{port}';opts=dict(dsn=e.dsn,storage_root=str(e.root),enable_acoustic_batches=True)
  kw=dict(stdin=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,encoding='utf8',creationflags=subprocess.CREATE_NO_WINDOW)
  server=subprocess.Popen([sys.executable,'-m','ptb_api.server','--port',str(port),'--frontend',str(ROOT/'frontend/dist'),'--managed'],stdout=subprocess.DEVNULL,**kw);server.stdin.write(json.dumps(opts|dict(signing_key=secrets.token_urlsafe(48),enable_jobs=True,enable_file_jobs=True))+'\n');server.stdin.flush()
  opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));end=time.monotonic()+20
  while True:
   try:
    with opener.open(origin+'/api/v1/health',timeout=1) as r:assert json.load(r)['mode']=='server'
    break
   except OSError:
    if time.monotonic()>end:raise RuntimeError('owned server timeout')
    time.sleep(.1)
  worker=subprocess.Popen([sys.executable,'-m','ptb_worker.cli'],stdout=subprocess.PIPE,**kw);worker.stdin.write(json.dumps(opts|dict(kind='postgres',reaper_binary=str(ROOT/'phonetic_toolbox/core/acoustic/reaper.exe')))+'\n');worker.stdin.flush();assert json.loads(worker.stdout.readline())=={'ready':True}
  runtime=Path.home()/'.cache/codex-runtimes/codex-primary-runtime/dependencies/node'
  config=dict(origin=origin,people=people,playwright=str(runtime/'node_modules/playwright'),output=str(out),uploads=[str(ROOT/f'output/validation/m07/baseline/round1/input{i}.wav') for i in (0,1)])
  result=subprocess.run([str(runtime/'bin/node.exe'),str(ROOT/'tests/e2e/m07-web.cjs')],input=json.dumps(config)+'\n',capture_output=True,text=True,encoding='utf8',creationflags=subprocess.CREATE_NO_WINDOW,timeout=240)
  log=result.stdout+'\n'+result.stderr
  for person in people:log=log.replace(person['password'],'[redacted]')
  (out/'browser.log').write_text(log,'utf8');assert result.returncode==0,'See browser.log'
  report=json.loads((out/'web-report.json').read_text('utf8'))
  with zipfile.ZipFile(out/'complete.zip') as archive:
   assert archive.testzip() is None
   for f in report['manifest']:assert hashlib.sha256(archive.read(f['name'])).hexdigest()==f['sha256']
  report['zip_all_hashes_verified']=True;(out/'web-report.json').write_text(json.dumps(report,indent=2),'utf8')
 finally:
  for process in (worker,server):
   if process:
    process.stdin.close()
    try:process.wait(timeout=15)
    except subprocess.TimeoutExpired:process.terminate();process.wait(timeout=5)
  eg.close();cg.close()
if __name__=='__main__':main()
