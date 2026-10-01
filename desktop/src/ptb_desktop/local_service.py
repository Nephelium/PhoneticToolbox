"""Own a single API child through pipes; no backend imports or global processes."""
import json
import queue
import secrets
import subprocess
import sys
import threading
import urllib.request


class LocalService:
    def __init__(self, jobs_path=None, *, local_files_root=None,reaper_binary=None):
        self.token = secrets.token_urlsafe(32)
        self.process = None
        self.url = None
        self.exit_code = None
        self.jobs_path = str(jobs_path) if jobs_path is not None else None
        self.local_files_root=str(local_files_root) if local_files_root is not None else None
        self.reaper_binary=str(reaper_binary) if reaper_binary is not None else None

    def start(self):
        if self.process is not None:
            raise RuntimeError('Service already started')
        self.process = subprocess.Popen(
            ([sys.executable, '--local-service', '--mode', 'local'] if getattr(sys,'frozen',False) else [sys.executable, '-m', 'ptb_api.cli', '--mode', 'local']),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding='utf-8', creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        try:
            self.process.stdin.write(json.dumps({'token': self.token, 'jobs_path':self.jobs_path,
                'local_files_root':self.local_files_root,'reaper_binary':self.reaper_binary}) + '\n')
            self.process.stdin.flush()
            lines = queue.Queue()
            threading.Thread(target=lambda: lines.put(self.process.stdout.readline()), daemon=True).start()
            ready = json.loads(lines.get(timeout=15))
            self.url = ready['url']
            if not self.url.startswith('http://127.0.0.1:'):
                raise RuntimeError('Invalid local service address')
            health = self.get('/api/v1/health')
            if health['mode'] != 'local':
                raise RuntimeError('Unexpected service mode')
            return health
        except Exception:
            self.close()
            raise

    def get(self, path):
        return self.request(path)

    def request(self, path, method='GET', body=None):
        if not path.startswith('/api/v1/') or '://' in path:
            raise ValueError('Only local API paths are supported')
        data=json.dumps(body).encode('utf-8') if body is not None else None
        request = urllib.request.Request(self.url + path, method=method, data=data,
            headers={'Authorization': 'Bearer ' + self.token, 'Origin':self.url, 'Content-Type':'application/json'})
        # Local traffic does not inherit a user's HTTP proxy.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=900 if path=='/api/v1/jobs/m11/component' else 35 if path=='/api/v1/jobs/egg/fonts' else 5) as response:
            return json.load(response)

    def close(self):
        process = self.process
        if process is None:
            return
        try:
            if process.poll() is None:
                try:
                    process.stdin.write('shutdown\n')
                    process.stdin.flush()
                except (BrokenPipeError, OSError):
                    pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait(timeout=5)
            self.exit_code = process.returncode
        finally:
            process.stdin.close()
            process.stdout.close()

    def import_input(self,payload,name,role):
        from urllib.parse import urlencode
        path='/api/v1/jobs/local-inputs?'+urlencode(dict(name=name,role=role))
        return json.loads(self.binary(path,'POST',payload))

    def binary(self,path,method='GET',payload=None,*,max_bytes=1_048_576):
        if not path.startswith('/api/v1/jobs/') or '://' in path:raise ValueError('Invalid local task path')
        request=urllib.request.Request(self.url+path,data=payload,method=method,
            headers={'Authorization':'Bearer '+self.token,'Origin':self.url,'Content-Type':'application/octet-stream'})
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request,timeout=30) as response:
            raw=response.read(max_bytes+1)
            if len(raw)>max_bytes:raise ValueError('Oversized task response')
            return raw

    def parameters(self,payload,name):
        from urllib.parse import urlencode
        request=urllib.request.Request(self.url+'/api/v1/preview/parameters?'+urlencode({'name':name}),data=payload,
            headers={'Authorization':'Bearer '+self.token,'Origin':self.url,'Content-Type':'application/octet-stream'})
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request,timeout=30) as response:
            raw=response.read(16_000_001)
            if len(raw)>16_000_000:raise ValueError('Parameter response budget')
            return json.loads(raw)

    def egg_preview(self, action, payload=None, session=None):
        from uuid import UUID
        from urllib.error import HTTPError
        path='/api/v1/preview/egg'+('/'+str(UUID(session)) if session else '')
        binary=action=='open'
        body=payload if binary else json.dumps(payload).encode() if payload is not None else None
        request=urllib.request.Request(self.url+path,data=body,method='DELETE' if action=='close' else 'POST',
            headers={'Authorization':'Bearer '+self.token,'Origin':self.url,
                     'Content-Type':'application/octet-stream' if binary else 'application/json'})
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(request,timeout=35) as response:
                raw=response.read(64_000_001)
                if len(raw)>64_000_000: raise ValueError('egg_preview_failed')
                return json.loads(raw)
        except HTTPError as error:
            try: code=json.load(error).get('detail','egg_preview_failed')
            except (ValueError,TypeError): code='egg_preview_failed'
            raise ValueError(code if isinstance(code,str) else 'egg_preview_failed') from None

    def preview(self,payload,query):
        from urllib.parse import urlencode
        from urllib.error import HTTPError
        request=urllib.request.Request(self.url+'/api/v1/preview/spectrogram?'+urlencode(query),data=payload,
            headers={'Authorization':'Bearer '+self.token,'Origin':self.url,'Content-Type':'application/octet-stream'})
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(request,timeout=30) as response:return json.load(response)
        except HTTPError as error:
            raise ValueError(json.load(error).get('detail','preview_failed')) from None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.close()
