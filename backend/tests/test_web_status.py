"""Public load counters must stay aggregate, bounded, cached and truthful."""
from pathlib import Path
from types import SimpleNamespace
from ptb_api.web_status import host_counters,PublicLoad


def test_host_counters_missing_and_linux_accounting(tmp_path):
    assert host_counters(tmp_path)==(None,None)
    (tmp_path/'stat').write_text('cpu 100 0 50 800 50 0 0 0 20 0\n')
    (tmp_path/'meminfo').write_text('MemTotal: 1000 kB\nMemAvailable: 600 kB\n')
    assert host_counters(tmp_path)==((1000,850),40.)
    (tmp_path/'meminfo').write_text('MemTotal: 0 kB\nMemAvailable: 600 kB\n')
    assert host_counters(tmp_path)[1] is None


def test_load_is_cached_and_cpu_uses_deltas():
    from contextlib import contextmanager
    reads=[];now=[0.];samples=iter([((1000,850),40.),((1200,900),50.),((1,1),None)])
    class Conn:
        def execute(self,query):
            return SimpleNamespace(fetchone=lambda:dict(running=1,waiting=2,batch=3))
    @contextmanager
    def read():reads.append(True);yield Conn()
    probe=PublicLoad(SimpleNamespace(_read=read,now=lambda c:123),SimpleNamespace(max_running=2),clock=lambda:now[0],probe=lambda:next(samples))
    first=probe.sample();assert first.cpu_percent is None and first.waiting_tasks==5
    assert probe.sample() is first and len(reads)==1
    now[0]=1.1;assert probe.sample().cpu_percent==75.
    now[0]=2.2;assert probe.sample().cpu_percent is None
    assert set(first.model_dump())=={'running_tasks','waiting_tasks','task_slots','cpu_percent','memory_percent','measured_at'}


def test_task_poll_does_not_sample_host_and_shares_one_read_per_second():
    from contextlib import contextmanager
    now=[0.];reads=[];probes=[]
    class Conn:
        def execute(self,query):return SimpleNamespace(fetchone=lambda:dict(running=len(reads),waiting=2,batch=3))
    @contextmanager
    def read():reads.append(now[0]);yield Conn()
    service=SimpleNamespace(_read=read,now=lambda c:now[0])
    def host():probes.append(now[0]);return ((1000+int(now[0]*100),500),40.)
    load=PublicLoad(service,SimpleNamespace(max_running=1),clock=lambda:now[0],probe=host)
    for i in range(40):assert load.sample(details=False).cpu_percent is None
    assert len(reads)==1 and not probes
    now[0]=1.01;assert load.sample(details=False).running_tasks==2
    assert not probes
    first=load.sample();assert first.cpu_percent is None
    assert len(reads)==2 and len(probes)==1
    for i in range(40):load.sample()
    assert len(probes)==1
    now[0]=2.02;assert load.sample().cpu_percent==100.
    now[0]=30;assert load.sample(details=False).cpu_percent is None
    assert len(probes)==2
    assert load.sample().cpu_percent is None  # do not average the whole closed interval
