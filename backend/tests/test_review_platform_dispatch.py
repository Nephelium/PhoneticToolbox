from types import SimpleNamespace
import pytest
from ptb_worker import acoustic_executor
from ptb_worker.io.limits import FormatError


@pytest.mark.parametrize('platform',['darwin','unsupported'])
def test_unknown_science_host_never_falls_through_to_windows(monkeypatch,platform):
    monkeypatch.setattr(acoustic_executor.sys,'platform',platform)
    with pytest.raises(FormatError,match='scientific_platform_unavailable'):
        acoustic_executor.collect_scientific('acoustic',None,SimpleNamespace(root=None),None,lambda:False)
