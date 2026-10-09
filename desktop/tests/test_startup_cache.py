import hashlib
import io
import json
import os
from pathlib import Path
import tarfile

import pytest
from ptb_desktop import startup_cache as cache

pytestmark = pytest.mark.skipif(os.name != 'nt', reason='Windows native file leases')


def record(data): return {'size':len(data),'sha256':hashlib.sha256(data).hexdigest()}


class Fixture:
    def __init__(self, revision=b'APP1', corrupt=False):
        self.config={'schema':'ptb-cache-payload/1','sharedFiles':{'shared.dll':'runtimes/egg/library.dll'}}
        sources={'science':{'runtimes/egg/python.exe':b'PYTHON','runtimes/egg/library.dll':b'SCIENCE'},
                 'host':{'qt.dll':b'QT'},'apps':{'PhoneticToolbox.exe':revision,'_internal/frontend/index.html':b'PAGE'}}
        science={n:record(v) for n,v in sources['science'].items()}
        host={n:record(v) for n,v in sources['host'].items()}|{'shared.dll':record(b'SCIENCE')}
        app={n:record(v) for n,v in sources['apps'].items()}
        combined=app|{'_internal/'+n:r for group in (science,host) for n,r in group.items()}
        self.config['components']={name:{'id':hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),'files':files} for name,files in [('science',science),('host',host),('apps',combined)]}
        self.config['applicationFiles']=app
        self.raw={};self.opens=[]
        for name,files in sources.items():
            out=io.BytesIO()
            with tarfile.open(fileobj=out,mode='w:xz') as archive:
                for path,data in files.items():
                    info=tarfile.TarInfo(path);info.size=len(data)
                    archive.addfile(info,io.BytesIO(data))
            self.raw[name]=out.getvalue()
        if corrupt:self.raw['science']=b'broken'

    def open(self,name): self.opens.append(name);return io.BytesIO(self.raw[name])
    def check(self,name): pass


def test_cold_warm_and_cross_version_reuse(tmp_path):
    root=tmp_path/'cache';first=Fixture()
    app,live,cold=cache.prepare(first,root);live.close()
    assert cold['prepared']==['science','host','apps']
    assert app.read_bytes()==b'APP1'
    assert os.path.samefile(app.parent/'_internal/shared.dll',app.parent/'_internal/runtimes/egg/library.dll')
    app2,live,warm=cache.prepare(first,root);live.close()
    assert app==app2 and warm['prepared']==[] and first.opens==['science','host','apps']
    second=Fixture(b'APP2')
    app3,live,upgrade=cache.prepare(second,root);live.close()
    assert app3!=app and app3.read_bytes()==b'APP2'
    assert upgrade['reused']==['science','host'] and second.opens==['apps']


def test_same_size_corruption_is_detected_and_repaired(tmp_path):
    root=tmp_path/'cache';payload=Fixture()
    app,live,_=cache.prepare(payload,root);live.close()
    target=app.parent/'_internal/runtimes/egg/library.dll'
    old=target.stat();target.write_bytes(b'BROKEN!');os.utime(target,ns=(old.st_atime_ns,old.st_mtime_ns))
    app,live,report=cache.prepare(payload,root);live.close()
    assert 'science' in report['prepared'] and target.read_bytes()==b'SCIENCE'


def test_active_application_prevents_cleanup_and_repairs(tmp_path):
    root=tmp_path/'cache';payload=Fixture()
    app,live,_=cache.prepare(payload,root)
    cache.request_clear(root)
    assert cache.clear_if_idle(root)=={'complete':False,'active':1}
    assert app.exists()
    with pytest.raises(RuntimeError,match='预约清除'):cache.prepare(payload,root)
    live.close()
    assert cache.clear_if_idle(root,requested_only=True)['complete'] and not root.exists()


def test_copy_fallback_keeps_bytes(tmp_path,monkeypatch):
    def denied(*args):raise OSError('volume cannot link')
    monkeypatch.setattr(os,'link',denied)
    payload=Fixture();app,live,report=cache.prepare(payload,tmp_path/'cache');live.close()
    assert report['scienceAttachment']['copied']==2 and report['hostAttachment']['copied']==2
    assert (app.parent/'_internal/shared.dll').read_bytes()==b'SCIENCE'


def test_incomplete_extraction_never_marked_ready_and_restarts(tmp_path):
    root=tmp_path/'cache'
    with pytest.raises(tarfile.ReadError):cache.prepare(Fixture(corrupt=True),root)
    assert not list(root.glob('science/*/ready.json'))
    app,live,_=cache.prepare(Fixture(),root);live.close()
    assert app.is_file() and not list((root/'staging').iterdir())


def test_unknown_root_never_deleted(tmp_path):
    root=tmp_path/'cache';root.mkdir();(root/'research.wav').write_bytes(b'KEEP')
    with pytest.raises(ValueError,match='ownership'):cache.clear_if_idle(root)
    assert (root/'research.wav').read_bytes()==b'KEEP'


def test_invalid_windows_paths_rejected():
    for name in ('../escape','C:/escape','x/CON','x./file','x\\file'):
        with pytest.raises(ValueError):cache.validate_records({name:record(b'X')})


def test_lease_lock_is_real_across_processes(tmp_path):
    import subprocess,sys
    root=tmp_path/'cache';_,live,_=cache.prepare(Fixture(),root)
    env=os.environ.copy()
    source='from pathlib import Path;from ptb_desktop.startup_cache import clear_if_idle;import sys,json;print(json.dumps(clear_if_idle(Path(sys.argv[1]))))'
    result=subprocess.run([sys.executable,'-B','-c',source,str(root)],env=env,capture_output=True,text=True,check=True)
    assert json.loads(result.stdout)['active']==1
    live.close();assert cache.clear_if_idle(root)['complete']


def test_two_fresh_processes_prepare_one_complete_cache(tmp_path):
    import subprocess,sys
    source='import runpy,sys,json;from pathlib import Path;from ptb_desktop import startup_cache as c;f=runpy.run_path(sys.argv[1])["Fixture"]();app,lease,report=c.prepare(f,Path(sys.argv[2]));lease.close();print(json.dumps(report))'
    args=[sys.executable,'-B','-c',source,str(Path(__file__)),str(tmp_path/'cache')]
    processes=[subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) for _ in range(2)]
    reports=[]
    for process in processes:
        stdout,stderr=process.communicate(timeout=20);assert process.returncode==0,stderr;reports.append(json.loads(stdout))
    assert sorted(len(r['prepared']) for r in reports)==[0,3]


def test_junction_cannot_redirect_verification_or_cleanup(tmp_path):
    import subprocess
    root=tmp_path/'cache';app,live,_=cache.prepare(Fixture(),root);live.close()
    original=app.parent/'_internal/frontend';original.rename(original.with_name('frontend-saved'))
    outside=tmp_path/'original-data';outside.mkdir();(outside/'index.html').write_bytes(b'PAGE')
    result=subprocess.run(['cmd','/c','mklink','/J',str(original),str(outside)],capture_output=True)
    assert result.returncode==0,result.stderr
    with pytest.raises(ValueError,match='directory'):cache.prepare(Fixture(),root)
    with pytest.raises(ValueError,match='reparse'):cache.clear_if_idle(root)
    assert (outside/'index.html').read_bytes()==b'PAGE'


def test_space_failure_cannot_publish_partial_files(tmp_path,monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(cache.shutil,'disk_usage',lambda root:SimpleNamespace(free=0))
    with pytest.raises(OSError,match='空间不足'):cache.prepare(Fixture(),tmp_path/'cache')
    assert not list((tmp_path/'cache').glob('science/*/ready.json'))


def test_warm_launch_reports_all_three_checks_and_detects_preserved_timestamp_damage(tmp_path):
    root=tmp_path/'cache';payload=Fixture()
    app,live,_=cache.prepare(payload,root);live.close()
    updates=[]
    _,live,warm=cache.prepare(payload,root,lambda *row:updates.append(row));live.close()
    assert warm['prepared']==[] and set(warm['componentSeconds'])=={'science','host','apps'}
    for family in ('science','host','apps'):
        values=[(done,total) for stage,done,total in updates if stage=='verify-'+family]
        assert values and values[0][0]==0 and values[-1][0]==values[-1][1]
        assert all(0<=done<=total for done,total in values)
        assert [done for done,total in values]==sorted(done for done,total in values)
    target=app.parent/'_internal/frontend/index.html';old=target.stat()
    target.write_bytes(b'FAIL');os.utime(target,ns=(old.st_atime_ns,old.st_mtime_ns))
    _,live,repair=cache.prepare(payload,root);live.close()
    assert repair['prepared']==['apps'] and target.read_bytes()==b'PAGE'


def test_parallel_verification_reads_each_hard_link_once_per_launch(tmp_path,monkeypatch):
    payload=Fixture();app,live,_=cache.prepare(payload,tmp_path/'cache');live.close()
    original=cache.digest;reads=[]
    def counted(path):
        info=path.stat();reads.append((info.st_dev,info.st_ino))
        return original(path)
    monkeypatch.setattr(cache,'digest',counted)
    for _ in range(2):
        reads.clear();_,live,_=cache.prepare(payload,tmp_path/'cache');live.close()
        assert len(reads)==len(set(reads))==5


def test_parallel_read_failure_finishes_readers_before_return(tmp_path,monkeypatch):
    import threading,time
    folder=tmp_path/'files';folder.mkdir();records={}
    for n in range(8):
        (folder/str(n)).write_bytes(b'DATA');records[str(n)]=record(b'DATA')
    original=cache.digest;active=set();lock=threading.Lock()
    def denied(path):
        with lock:active.add(path.name)
        try:
            time.sleep(.01)
            if path.name=='0':raise PermissionError('owned test denial')
            return original(path)
        finally:
            with lock:active.remove(path.name)
    monkeypatch.setattr(cache,'digest',denied)
    with pytest.raises(PermissionError):cache.verify_tree(folder,records,{})
    assert not active
