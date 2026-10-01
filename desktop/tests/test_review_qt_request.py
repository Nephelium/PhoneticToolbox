"""Actual Qt slot/signal transport without a window, database or user file."""
import base64
import hashlib
import json
from PyQt6.QtCore import QCoreApplication
from PyQt6.QtTest import QSignalSpy
from ptb_desktop.file_provider import FileProvider
from ptb_desktop.host import Bridge


def test_mfa_large_payload_reaches_adapter_through_qt_slot():
    application=QCoreApplication.instance() or QCoreApplication([])
    content=b'0'*6_100_000
    class Service:
        def import_input(self,raw,name,role):
            assert raw==content and name=='transport.wav' and role=='audio'
            return {'size':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
    bridge=Bridge(FileProvider(),Service(),None)
    spy=QSignalSpy(bridge.taskReady)
    bridge.task('large-mfa',json.dumps(dict(op='m11_import',role='audio',name='transport.wav',
        base64=base64.b64encode(content).decode('ascii'))))
    assert len(spy) or spy.wait(5000)
    assert spy[0][0]=='large-mfa'
    assert json.loads(spy[0][1])=={'ok':True,'value':{'size':len(content),
        'sha256':hashlib.sha256(content).hexdigest()}}
    bridge.task('bad-role',json.dumps(dict(op='m11_import',role='other',base64='AA==')))
    assert len(spy)==2 and not json.loads(spy[1][1])['ok']
