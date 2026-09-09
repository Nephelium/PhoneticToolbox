"""Own a single API child through pipes; no backend imports or global processes."""
import json
import queue
import secrets
import subprocess
import sys
import threading
import urllib.request


class LocalService:
    def __init__(self, jobs_path=None):
        self.token = secrets.token_urlsafe(32)
        self.process = None
        self.url = None
        self.exit_code = None
        self.jobs_path = str(jobs_path) if jobs_path is not None else None

    def start(self):
        if self.process is not None:
            raise RuntimeError('Service already started')
        self.process = subprocess.Popen(
            [sys.executable, '-m', 'ptb_api.cli', '--mode', 'local'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding='utf-8', creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        try:
            self.process.stdin.write(json.dumps({'token': self.token, 'jobs_path':self.jobs_path}) + '\n')
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
        with opener.open(request, timeout=5) as response:
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

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.close()
