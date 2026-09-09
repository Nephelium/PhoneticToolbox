"""M01-E actual HTTP/auth/shared UI with in-memory store doubles, not PG evidence."""
from pathlib import Path
from uuid import uuid4
import json
import io
import wave
import math
import struct
import secrets
import socket
import subprocess
import sys
import threading
import time
import uvicorn
from starlette.staticfiles import StaticFiles
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests/support'))
from account_double import MemoryAccountStore
from m01_preview_double import PreviewStorage,GRID
from ptb_api.main import create_app
from ptb_api.auth import AuthSettings


def main():
    output=ROOT/'output/playwright'/('m01-e-'+uuid4().hex);output.mkdir(parents=True)
    accounts=MemoryAccountStore();storage=PreviewStorage();people=[]
    raw=(ROOT/'frontend/src/assets/SYN-EGG-44100.wav').read_bytes()
    with wave.open(io.BytesIO(raw),'rb') as source:
        buffer=io.BytesIO()
        with wave.open(buffer,'wb') as target:
            target.setparams(source.getparams());target.setframerate(22050);target.writeframes(source.readframes(source.getnframes()))
        slower=buffer.getvalue()
    long_buffer=io.BytesIO()
    with wave.open(long_buffer,'wb') as target:
        target.setnchannels(1);target.setsampwidth(2);target.setframerate(16000)
        second=b''.join(struct.pack('<h',int(12000*math.sin(2*math.pi*1000*i/16000))) for i in range(16000))
        target.writeframes(second*600)
    long_audio=long_buffer.getvalue()
    for username in ('research_a','research_b'):
        password=secrets.token_urlsafe(20);accounts.create_user(username,password);owner=accounts.users[username]['id']
        projects=[]
        for name in (['测试项目甲','测试项目乙'] if username=='research_a' else ['测试项目丙']):
            project=accounts.create_project(owner,name);projects.append(project)
            for i in range(17 if name=='测试项目甲' else 1):storage.add(owner,project['id'],f'测试声调{i:02}.wav',slower if i==1 else long_audio if i==16 else raw)
            storage.add(owner,project['id'],'测试声调00.TextGrid',GRID)
        people.append(dict(username=username,password=password,owner=owner))
    sock=socket.socket();sock.bind(('127.0.0.1',0));origin='http://127.0.0.1:'+str(sock.getsockname()[1])
    app=create_app(account_store=accounts,auth_settings=AuthSettings(origin=origin,signing_key=secrets.token_urlsafe(48),allow_insecure_loopback=True),storage=storage)
    app.mount('/server',StaticFiles(directory=ROOT/'frontend/dist',html=True))
    server=uvicorn.Server(uvicorn.Config(app,log_level='warning',access_log=False))
    thread=threading.Thread(target=lambda:server.run(sockets=[sock]),daemon=True);thread.start()
    try:
        deadline=time.monotonic()+10
        while not server.started:
            if time.monotonic()>deadline:raise RuntimeError('Test HTTP host did not start')
            time.sleep(.05)
        runtime=Path.home()/'.cache/codex-runtimes/codex-primary-runtime/dependencies/node'
        config=dict(origin=origin,people=people,output=str(output),playwright=str(runtime/'node_modules/playwright'),browser='C:/Program Files/Google/Chrome/Application/chrome.exe')
        result=subprocess.run([str(runtime/'bin/node.exe'),'tests/e2e/m01-workspace.cjs'],input=json.dumps(config)+'\n',text=True,encoding='utf-8',capture_output=True,timeout=100,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        log=result.stdout+'\n'+result.stderr
        for person in people:log=log.replace(person['password'],'[redacted]')
        (output/'runner.log').write_text(log,encoding='utf-8')
        print(json.dumps({'output':str(output),'exit_code':result.returncode,'scope':'HTTP/UI with memory doubles; no PG or scientific job claim'},ensure_ascii=False))
        if result.returncode:raise RuntimeError('See test runner.log')
    finally:
        server.should_exit=True;thread.join(10);sock.close()
        assert not thread.is_alive()


if __name__=='__main__':main()
