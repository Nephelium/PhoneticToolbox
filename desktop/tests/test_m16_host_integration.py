"""Boundary checks for the independent native recording route; no devices or DB."""
import json
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch
from ptb_desktop.host import Bridge,Workbench


class NativeRecordingRouteTest(unittest.TestCase):
    def test_stop_is_not_blocked_by_scientific_task_lock(self):
        values=[];done=threading.Event();task_lock=threading.Lock();task_lock.acquire()
        route=SimpleNamespace(dispatch=Mock(return_value={'stopped':True}))
        def emit(id,raw):values.append((id,json.loads(raw)));done.set()
        owner=SimpleNamespace(recording_bridge=lambda:route,recording_lock=threading.Lock(),task_lock=task_lock,recordingReady=SimpleNamespace(emit=emit))
        Bridge.recording(owner,'stop-1','{"op":"stop"}')
        self.assertTrue(done.wait(2));self.assertTrue(values[0][1]['ok'])
        route.dispatch.assert_called_once_with({'op':'stop'})
        self.assertTrue(task_lock.locked())

    def test_invalid_and_busy_recording_requests_do_not_dispatch(self):
        values=[];route=SimpleNamespace(dispatch=Mock())
        owner=SimpleNamespace(recording_bridge=lambda:route,recording_lock=threading.Lock(),recordingReady=SimpleNamespace(emit=lambda id,raw:values.append(json.loads(raw))))
        Bridge.recording(owner,'one','[]');self.assertFalse(values[-1]['ok'])
        owner.recording_lock.acquire();Bridge.recording(owner,'two','{"op":"stop"}')
        self.assertFalse(values[-1]['ok']);route.dispatch.assert_not_called()

    def test_close_failure_keeps_window_and_owned_services_alive(self):
        owner=SimpleNamespace(closing=True,bridge=SimpleNamespace(close_recording=lambda:False,vocal_files=Mock()),vocal=Mock(),provider=Mock(),service=Mock(),update_coordinator=Mock())
        event=Mock()
        with patch('ptb_desktop.host.QMessageBox.warning'):
            Workbench.closeEvent(owner,event)
        self.assertFalse(owner.closing);event.ignore.assert_called_once();event.accept.assert_not_called()
        owner.vocal.close.assert_not_called();owner.provider.close.assert_not_called();owner.service.close.assert_not_called()
        owner.update_coordinator.cancel.assert_called_once()

    def test_recording_close_exception_is_failure(self):
        owner=SimpleNamespace(_recording_bridge=SimpleNamespace(close=Mock(side_effect=OSError('disk unavailable'))))
        self.assertFalse(Bridge.close_recording(owner))

    def test_m05_cannot_arm_media_while_m16_recording_or_probe_owns_device(self):
        media=SimpleNamespace(arm=Mock(return_value=True))
        owner=SimpleNamespace(_recording_bridge=SimpleNamespace(capturing=True),window=SimpleNamespace(m05_media=media))
        value=json.loads(Bridge.invoke(owner,'{"op":"m05_media"}'))
        self.assertFalse(value['ok']);self.assertIn('录前检测',value['error']);media.arm.assert_not_called()
        owner._recording_bridge.capturing=False
        self.assertTrue(json.loads(Bridge.invoke(owner,'{"op":"m05_media"}'))['ok'])
        media.arm.assert_called_once()


if __name__=='__main__':unittest.main()
