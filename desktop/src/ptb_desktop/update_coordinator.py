"""Queue update-close requests on Qt's GUI thread and preserve close cancellation."""
import threading
import uuid
from PyQt6.QtCore import QObject, QTimer, Qt, pyqtSignal, pyqtSlot
from .updates import UpdateError
from .update_apply import prepare_handoff, launch_handoff
from .updates import _atomic_json


class UpdateCoordinator(QObject):
    requested = pyqtSignal(object)

    def __init__(self, window, *, prepare=prepare_handoff, launch=launch_handoff):
        super().__init__(window)
        self.window, self.prepare, self.launch = window, prepare, launch
        self._lock = threading.Lock()
        self.pending = None
        self.requested.connect(self._on_requested, Qt.ConnectionType.QueuedConnection)

    def native_busy(self, *, jobs=True):
        bridge = self.window.bridge
        if bridge.task_lock.locked() or bridge.preview_lock.locked() or bridge.recording_lock.locked():
            return '本机操作尚未完成，请等待后重试更新。'
        recording = bridge._recording_bridge
        if recording and (recording.capturing or recording.service.job):
            return '请先停止录音、录前检测和后台处理，再更新。'
        if getattr(self.window.vocal, 'pending', None):
            return '声道任务尚未完成，请等待后更新。'
        if jobs and self.window.service.local_files_root:
            from .task_bridge import PROJECT
            try:
                values = self.window.service.get('/api/v1/jobs?project_id=' + PROJECT)['jobs']
            except Exception:
                return '无法确认任务状态，暂不退出更新。'
            if any(j.get('state') in ('queued', 'running', 'cancel_requested') for j in values):
                return '仍有正在运行或排队的任务，请完成或取消后更新。'
        return ''

    def apply(self, path, kind):
        if not self._lock.acquire(blocking=False):
            raise UpdateError('APPLY_BUSY', '已有退出更新请求，请先处理。')
        try:
            reason = self.native_busy()
            if reason:
                raise UpdateError('APPLY_BUSY', reason)
            service = self.window.updates_bridge.service
            entries = [(key, package) for key, (candidate, package) in service._downloads.items() if candidate == path]
            if len(entries) != 1:
                raise UpdateError('DOWNLOAD_UNKNOWN', '更新包身份已失效。')
            key, package = entries[0]
            version = service._download_versions.get(key)
            request, plan = self.prepare(service.root, path, kind, package.size, package.sha256, version, service.current.value)
            entry = {'request': request, 'plan': plan, 'done': threading.Event(), 'allowed': False, 'message': ''}
            self.requested.emit(entry)
            if not entry['done'].wait(300):
                entry['expired'] = True
                _atomic_json(request.parent/'status.json', {'state':'failed','code':'APPLY_TIMEOUT'})
                raise UpdateError('APPLY_TIMEOUT', '退出更新等待超时，程序和更新包保留。')
            if not entry['allowed']:
                raise UpdateError('APPLY_CANCELLED', entry['message'] or '已取消退出更新，当前编辑保留。')
            return {'started': True, 'message': '关闭保护已通过，等待窗口退出后更新。'}
        finally:
            self._lock.release()

    def clear_caches(self):
        if self.pending or self._lock.locked():
            raise UpdateError('APPLY_BUSY', '请先处理当前退出请求。')
        reason=self.native_busy()
        if reason:raise UpdateError('APPLY_BUSY',reason.replace('更新','清理缓存'))
        entry={'action':'clear-caches','plan':{'token':uuid.uuid4().hex},'done':threading.Event(),'allowed':False,'message':''}
        self._on_requested(entry)
        return {'started':self.pending is entry}

    @staticmethod
    def _status(entry, value):
        if entry.get('action') != 'clear-caches':
            _atomic_json(entry['request'].parent/'status.json',value)

    @pyqtSlot(object)
    def _on_requested(self, entry):
        if self.pending or entry.get('expired') or self.window.closing:
            entry['message'] = '窗口正在关闭，请稍后重试。'
            entry['done'].set()
            return
        reason = self.native_busy(jobs=False)
        if reason:
            entry['message'] = reason
            entry['done'].set()
            return
        self.pending = entry
        QTimer.singleShot(300000, lambda:self.cancel('退出更新等待超时，程序保留。') if self.pending is entry and not entry['done'].is_set() else None)
        self.window.updates_bridge.prepareClose.emit(entry['plan']['token'])

    def reply(self, token, allowed, message):
        entry = self.pending
        if not entry or entry.get('expired') or token != entry['plan']['token']:
            return
        reason = self.native_busy(jobs=False)
        entry['allowed'] = allowed is True and not reason
        entry['message'] = reason or message[:240]
        entry['done'].set()
        if entry['allowed']:
            # The existing RequestClose/beforeunload path still has the final say.
            QTimer.singleShot(80, self.window.request_update_close)
        else:
            self._status(entry, {'state':'cancelled','message':entry['message']})
            self.pending = None

    def closed(self):
        entry = self.pending
        if not entry or not entry['allowed'] or entry.get('expired'):
            return False
        reason = self.native_busy(jobs=False)
        if reason:
            raise UpdateError('APPLY_BUSY', reason)
        if entry.get('action') == 'clear-caches':
            # The page's save/close guards and recording finalization have now
            # accepted exit. No deletion is requested before this point.
            result=self.window.service.request('/api/v1/jobs/local-storage/clear-all','POST')
            if not result.get('complete',False):
                raise UpdateError('CACHE_PARTIAL','部分处理结果缓存暂时被占用，请稍后重试。')
            from .startup_cache import request_clear
            request_clear()
            self.pending=None
            return True
        self.launch(entry['request'], entry['plan'])
        self.pending = None
        return True

    def cancel(self, message='已取消退出，当前编辑保留。'):
        if self.pending:
            self._status(self.pending, {'state':'cancelled','message':message})
            self.window.updates_bridge.closeCancelled.emit(self.pending['plan']['token'])
            self.pending.update(allowed=False, message=message)
            self.pending['done'].set()
            self.pending = None
