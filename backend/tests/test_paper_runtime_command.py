from pathlib import Path
import sys
import pytest
from ptb_worker.native import linux_runtime
from ptb_worker.io.limits import FormatError


def test_paper_local_libraries_do_not_change_scientific_commands(tmp_path, monkeypatch):
    profile={'python':sys.executable,'paper_libraries':str(tmp_path.resolve())}
    monkeypatch.setattr(linux_runtime,'load_profile',lambda:(tmp_path/'profile.json',profile))
    assert linux_runtime.command('papers','request')[:2]==['/usr/bin/env','LD_LIBRARY_PATH='+str(tmp_path.resolve())]
    assert linux_runtime.command('lpc','request')[0]==sys.executable
    assert linux_runtime.command('egg','request')[0]==sys.executable


def test_missing_or_relative_paper_libraries_are_rejected(tmp_path, monkeypatch):
    for directory in ('relative-library',str(tmp_path/'missing')):
        monkeypatch.setattr(linux_runtime,'load_profile',lambda:(tmp_path/'profile.json',{'python':sys.executable,'paper_libraries':directory}))
        with pytest.raises(FormatError):linux_runtime.command('papers','request')
