"""Independent assertions for latency summaries and bounded test stop policy."""
import importlib.util
from pathlib import Path


path=Path(__file__).resolve().parents[2]/'scripts/benchmark_modules.py'
spec=importlib.util.spec_from_file_location('p11_benchmark',path)
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def test_percentiles_use_nearest_rank_and_missing_is_not_zero():
    assert module.percentile(list(range(1,101)),95)==95
    assert module.percentile(list(range(1,101)),99)==99
    assert module.distribution([])==dict(n=0,p50=None,p95=None,p99=None,maximum=None)
    assert module.percentile([8,1,2],95)==8


def sample():
    return dict(memory={'MemAvailable':1024**3},disk_free_bytes=20*1024**3,
                pressure={'memory':'some avg10=0.00 avg60=0.00 total=0\nfull avg10=0.00 avg60=0.00 total=0'},
                processes={'1':{'Pss':100}},groups=[])


def test_pressure_must_persist_and_recovery_resets_timer():
    policy=module.StopPolicy(); value=sample()
    value['memory']['MemAvailable']=699*1024**2
    assert policy.reason(value,0) is None
    assert policy.reason(value,9) is None
    assert policy.reason(sample(),9.5) is None
    assert policy.reason(value,10) is None
    assert policy.reason(value,20)=='sustained_memory_pressure'


def test_oom_disk_and_unknown_ownership_stop_immediately():
    value=sample(); value['groups']=[{'memory.events':'oom 1\noom_kill 0'}]
    assert module.StopPolicy().reason(value,0)=='owned_group_oom'
    value=sample(); value['disk_free_bytes']=9*1024**3
    assert module.StopPolicy().reason(value,0)=='disk_free_below_10_GiB'
    value=sample(); value['unknown_owner']=True
    assert module.StopPolicy().reason(value,0)=='resource_ownership_unknown'
