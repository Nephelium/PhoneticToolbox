"""Application-owned vocal-tract worker. Never attaches to an unrelated server."""
import atexit
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import threading
import time
import urllib.request

from phonetic_toolbox.models.vocal_tract_models import VocalTractLaunchResult
from .vocal_tract.process_guard import OwnedJob


class VocalTractService:
    def __init__(self, *, profile_dir=None, silent=False, command=None):
        if profile_dir is None:
            profile_dir=Path(os.environ.get('LOCALAPPDATA',Path.home()/'.local/share'))/'PhoneticToolbox/vocal_tract'
        self.profile_dir=Path(profile_dir)
        self.silent=silent;self.command=command
        self.lock=threading.RLock();self.start_lock=threading.Lock()
        self.process=None;self.job=None;self.ready=None;self.closing=False

    def launch(self):
        with self.start_lock:
            try:
                with self.lock:
                    if self.closing:raise RuntimeError('主应用正在退出')
                    if self.process and self.process.poll() is None and self.ready:
                        return VocalTractLaunchResult(True,'服务已复用',self.ready['url'],self.process.pid)
                    if self.job:self.job.close();self.job=None
                    self.profile_dir.mkdir(parents=True,exist_ok=True)
                    instance=secrets.token_hex(12)
                    handshake=self.profile_dir/f'worker-{os.getpid()}.json'
                    if self.command:command=list(self.command)
                    elif getattr(sys,'frozen',False):command=[sys.executable,'--vocal-tract-worker']
                    else:command=[sys.executable,str(Path(__file__).resolve().parents[2]/'run.py'),'--vocal-tract-worker']
                    command+=['--parent-pid',str(os.getpid()),'--handshake',str(handshake),'--instance',instance,'--profile',str(self.profile_dir)]
                    if self.silent:command.append('--silent')
                    # PYINSTALLER_RESET_ENVIRONMENT is deliberately absent: this child
                    # shares its parent's extraction directory and cannot outlive it.
                    env=os.environ.copy();env.pop('PYINSTALLER_RESET_ENVIRONMENT',None)
                    with (self.profile_dir/f'worker-{os.getpid()}.log').open('w',encoding='utf-8') as log:
                        self.process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0,env=env)
                    self.ready=None
                    self.job=OwnedJob(self.process)
                    process=self.process
                deadline=time.monotonic()+45
                while time.monotonic()<deadline:
                    with self.lock:
                        if self.closing:raise RuntimeError('主应用正在退出')
                        if process.poll() is not None:raise RuntimeError(f'工作进程退出 ({process.returncode})；请查看 {self.profile_dir}/worker-{os.getpid()}.log')
                    try:
                        ready=json.loads(handshake.read_text(encoding='utf-8'))
                        if ready.get('instance')==instance and ready.get('pid')==process.pid and ready.get('parent_pid')==os.getpid():
                            with self.lock:
                                if self.closing:raise RuntimeError('主应用正在退出')
                                self.ready=ready
                            return VocalTractLaunchResult(True,'服务已启动',ready['url'],process.pid)
                    except (OSError,ValueError):pass
                    time.sleep(.05)
                raise TimeoutError('声道工作台启动超时')
            except Exception as exc:
                self._stop_owned()
                return VocalTractLaunchResult(False,str(exc))

    def _stop_owned(self):
        with self.lock:
            process=self.process;ready=self.ready;job=self.job
            self.process=None;self.ready=None;self.job=None
        if process and process.poll() is None:
            if ready:
                try:
                    request=urllib.request.Request(ready['url']+'/api/shutdown',data=b'{}',headers={'X-Session':ready['token'],'Content-Type':'application/json'})
                    with urllib.request.urlopen(request,timeout=.8) as response:response.read()
                except (OSError,ValueError):pass
            try:process.wait(timeout=1.5)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:process.wait(timeout=2)
                except subprocess.TimeoutExpired:process.kill();process.wait(timeout=2)
        if job:job.close()

    def shutdown(self):
        with self.lock:self.closing=True
        self._stop_owned()


_service=VocalTractService()
atexit.register(_service.shutdown)


def launch_vocal_tract():
    return _service.launch()


def shutdown_vocal_tract():
    _service.shutdown()
