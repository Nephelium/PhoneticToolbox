import pytest
from ptb_worker import vocal_child


def test_host_affinity_is_applied_before_native_thread_start(monkeypatch):
    called=[];monkeypatch.setattr(vocal_child.sys,'platform','linux')
    monkeypatch.setenv('PTB_WEB_COMPUTE_CPUS','1')
    monkeypatch.setattr(vocal_child.os,'sched_getaffinity',lambda pid:{0,1},raising=False)
    monkeypatch.setattr(vocal_child.os,'sched_setaffinity',lambda pid,cpus:called.append((pid,cpus)),raising=False)
    vocal_child.configure_cpu_affinity();assert called==[(0,{1})]


@pytest.mark.parametrize('configured',['','2','x'])
def test_invalid_host_cpu_selection_fails_before_native_work(monkeypatch,configured):
    monkeypatch.setattr(vocal_child.sys,'platform','linux');monkeypatch.setenv('PTB_WEB_COMPUTE_CPUS',configured)
    monkeypatch.setattr(vocal_child.os,'sched_getaffinity',lambda pid:{0,1},raising=False)
    with pytest.raises(RuntimeError):vocal_child.configure_cpu_affinity()


def test_native_platform_without_server_setting_retains_its_cpu_set(monkeypatch):
    monkeypatch.setattr(vocal_child.sys,'platform','linux');monkeypatch.delenv('PTB_WEB_COMPUTE_CPUS',raising=False)
    monkeypatch.setattr(vocal_child.os,'sched_setaffinity',lambda *args:pytest.fail('Unexpected affinity change'),raising=False)
    vocal_child.configure_cpu_affinity()
