"""Regression for Qt page-scoped cached media decisions and explicit retry."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from PyQt6.QtCore import QUrl
from PyQt6.QtWidgets import QMessageBox
from ptb_desktop.m05_permissions import MediaPermission


class Request:
    def __init__(self, origin='ptbapp://app/', kind='MediaVideoCapture'):
        self.url=QUrl(origin);self.kind=kind;self.decision=None;self.resets=0;self.valid=True
    def origin(self):return self.url
    def permissionType(self):return SimpleNamespace(name=self.kind)
    def isValid(self):return self.valid
    def reset(self):self.resets+=1;self.decision=None
    def grant(self):self.decision='granted'
    def deny(self):self.decision='denied'


class MediaPermissionTest(unittest.TestCase):
    def setUp(self):
        self.permission=MediaPermission(None)
        self.copy=patch('ptb_desktop.m05_permissions.QWebEnginePermission',side_effect=lambda request:request)
        self.copy.start();self.addCleanup(self.copy.stop)

    def test_denial_retry_is_ask_not_automatic_grant(self):
        self.permission.arm();first=Request()
        with patch.object(QMessageBox,'question',return_value=QMessageBox.StandardButton.No):
            self.permission.requested(first)
        self.assertEqual(first.decision,'denied')
        self.permission.arm()
        self.assertEqual(first.resets,1);self.assertIsNone(first.decision)
        second=Request()
        with patch.object(QMessageBox,'question',return_value=QMessageBox.StandardButton.Yes) as prompt:
            self.permission.requested(second)
        prompt.assert_called_once();self.assertEqual(second.decision,'granted')

    def test_expired_gate_can_retry(self):
        self.permission.until=-1;request=Request()
        with patch.object(QMessageBox,'question') as prompt:self.permission.requested(request)
        prompt.assert_not_called();self.assertEqual(request.decision,'denied')
        self.permission.arm();self.assertEqual(request.resets,1)

    def test_other_origin_or_permission_is_never_reset(self):
        self.permission.arm()
        for request in (Request('https://example.com/'),Request(kind='ClipboardReadWrite')):
            with patch.object(QMessageBox,'question') as prompt:self.permission.requested(request)
            prompt.assert_not_called();self.assertEqual(request.decision,'denied')
            self.permission.arm();self.assertEqual(request.resets,0)

    def test_invalidated_page_request_is_not_reset(self):
        self.permission.arm();request=Request()
        with patch.object(QMessageBox,'question',return_value=QMessageBox.StandardButton.Yes):
            self.permission.requested(request)
        request.valid=False;self.permission.arm();self.assertEqual(request.resets,0)


if __name__=='__main__':unittest.main()
