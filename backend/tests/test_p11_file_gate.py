"""Unbounded archive paths cannot bypass the server scientific lane."""
from contextlib import nullcontext
import json
import sys
from types import SimpleNamespace
import threading

import pytest

from ptb_worker import file_executor


@pytest.mark.parametrize('operation', ['archive_zip', 'extract_zip'])
@pytest.mark.parametrize('profile', ['server-small', 'trusted-worker', 'invalid'])
def test_server_rejects_before_any_storage_read(monkeypatch, operation, profile):
    monkeypatch.setattr(sys, 'platform', 'linux')
    monkeypatch.setenv('PTB_RESOURCE_PROFILE', profile)
    failures=[]
    # No storage object: touching an input would fail this test.
    files=SimpleNamespace(fail=lambda identity, code: failures.append((identity, code)))
    claim=dict(id='job',generation=1,snapshot=json.dumps(dict(operation=operation,config={})))
    file_executor.execute_file_claim(SimpleNamespace(files=files),claim,'worker',threading.Event())
    assert failures==[(('job','worker',1),'server_export_unavailable')]


@pytest.mark.parametrize('operation', ['archive_zip', 'extract_zip'])
def test_desktop_retains_existing_archive_execution(monkeypatch, operation):
    monkeypatch.setenv('PTB_RESOURCE_PROFILE','desktop-local')
    seen=[]
    files=SimpleNamespace(storage=SimpleNamespace(_locked=lambda: nullcontext(None)),
                          _fence=lambda conn,identity: (None,[{'id':'input'}]),
                          complete=lambda identity: seen.append('complete'))
    monkeypatch.setattr(file_executor,'archive',lambda *args: seen.append('archive_zip'))
    monkeypatch.setattr(file_executor,'extract',lambda *args: seen.append('extract_zip'))
    claim=dict(id='job',generation=1,snapshot=json.dumps(dict(operation=operation,config={'max_output_bytes':100})))
    file_executor.execute_file_claim(SimpleNamespace(files=files),claim,'worker',threading.Event())
    assert seen==[operation,'complete']
