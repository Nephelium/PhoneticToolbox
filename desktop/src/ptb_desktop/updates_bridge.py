"""Independent QWebChannel updater capability; the local-only web policy stays closed."""
from concurrent.futures import ThreadPoolExecutor
import json
import re
import threading
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from .updates import UpdateError, UpdateService


class UpdatesBridge(QObject):
    ready = pyqtSignal(str, str)
    progress = pyqtSignal(str, str)
    prepareClose = pyqtSignal(str)
    closeCancelled = pyqtSignal(str)

    def __init__(self, parent=None, *, service=None, **service_options):
        super().__init__(parent)
        self.service = service or UpdateService(**service_options)
        self._pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix='ptb-updater')
        self._jobs = {}
        self._lock = threading.RLock()
        self._closed = False
        self.close_reply = None

    @pyqtSlot(str, bool, str)
    def closeReply(self, token, allowed, message):
        if not self._closed and self.close_reply is not None:
            self.close_reply(token, allowed, message)

    def _emit(self, request_id, value):
        with self._lock:
            if not self._closed:
                self.ready.emit(request_id, json.dumps(value, ensure_ascii=False))

    @pyqtSlot(str, str)
    def request(self, request_id, payload):
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,96}', request_id):
            return
        try:
            if len(payload.encode('utf-8')) > 8192:
                raise UpdateError('REQUEST_INVALID', '更新请求超过大小限制。')
            value = json.loads(payload)
            if not isinstance(value, dict):
                raise UpdateError('REQUEST_INVALID', '更新请求无效。')
            with self._lock:
                if self._closed:
                    return
                if request_id in self._jobs:
                    # Never overwrite the first operation's cancellation capability.
                    return
                if len(self._jobs) >= 3:
                    raise UpdateError('BUSY', '更新操作繁忙，请稍后重试。')
                cancel = threading.Event()
                self._jobs[request_id] = cancel
                self._pool.submit(self._execute, request_id, value, cancel)
        except (ValueError, UpdateError) as error:
            self._emit(request_id, {'ok': False, 'error': {'code': getattr(error, 'code', 'REQUEST_INVALID'),
                                                        'message': str(error) if isinstance(error, UpdateError) else '更新请求无法解析。'}})

    def _execute(self, request_id, value, cancel):
        def progress(result):
            with self._lock:
                if not self._closed and not cancel.is_set():
                    self.progress.emit(request_id, json.dumps(result, ensure_ascii=False))
        try:
            operation = value.get('operation')
            args = value.get('args', {})
            if set(value) - {'operation', 'args'} or not isinstance(args, dict):
                raise UpdateError('REQUEST_INVALID', '更新请求字段无效。')
            allowed = {'preferences': set(), 'configure': {'source', 'channel', 'autoCheck'},
                       'check': {'manual', 'source', 'channel'}, 'acknowledge': {'releaseId'},
                       'download': {'releaseId', 'packageKind', 'confirmed'}, 'apply': {'downloadId', 'confirmed'}}
            if operation not in allowed or set(args) - allowed[operation]:
                raise UpdateError('REQUEST_INVALID', '更新操作无法识别。')
            if operation == 'preferences':
                result = self.service.preferences()
            elif operation == 'configure':
                result = self.service.set_preferences(args)
            elif operation == 'check':
                result = self.service.check(manual=args.get('manual', False), source=args.get('source'), channel=args.get('channel'), cancel=cancel)
            elif operation == 'acknowledge':
                result = self.service.acknowledge_notice(args.get('releaseId'))
            elif operation == 'download':
                result = self.service.download(args.get('releaseId'), package_kind=args.get('packageKind'), confirmed=args.get('confirmed', False), cancel=cancel, progress=progress)
            else:
                result = self.service.apply(args.get('downloadId'), confirmed=args.get('confirmed', False))
            if cancel.is_set():
                raise UpdateError('CANCELLED', '更新操作已取消。')
            response = {'ok': True, 'value': result}
        except UpdateError as error:
            response = {'ok': False, 'error': {'code': error.code, 'message': str(error)}}
        except Exception:
            # Do not expose exception repr, URL trace bodies or host paths to the page.
            response = {'ok': False, 'error': {'code': 'UPDATE_FAILED', 'message': '更新操作失败，请稍后重试。'}}
        finally:
            with self._lock:
                self._jobs.pop(request_id, None)
        self._emit(request_id, response)

    @pyqtSlot(str)
    def cancel(self, request_id):
        with self._lock:
            event = self._jobs.get(request_id)
            if event:
                event.set()

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for event in self._jobs.values():
                event.set()
        self.service.stop_maintenance()
        self._pool.shutdown(wait=False, cancel_futures=True)
