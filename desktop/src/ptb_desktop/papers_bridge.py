"""M18 serialized native network/PDF capability."""
from concurrent.futures import ThreadPoolExecutor
import json
import re
import threading
from PyQt6.QtCore import QObject, QTimer, pyqtSignal, pyqtSlot
from .papers import PaperService, PaperError


class PapersBridge(QObject):
    ready = pyqtSignal(str, str)
    progress = pyqtSignal(str, str)
    statusChanged = pyqtSignal(str)
    scheduleChanged = pyqtSignal()

    def __init__(self, root, parent=None, *, export_picker=None, root_picker=None, import_picker=None):
        super().__init__(parent)
        try: self.service = PaperService(root)
        except Exception: self.service = None
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='ptb-papers')
        self.network_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='ptb-paper-download')
        self.jobs = {}; self.futures = {}; self.lock = threading.RLock(); self.closed = False
        self.export_picker = export_picker
        self.root_picker = root_picker; self.import_picker = import_picker
        self.timer = QTimer(self); self.timer.timeout.connect(self.automatic_check)
        self.scheduleChanged.connect(self.update_schedule)
        self.auto_cancel = None; self.maintenance_started = False

    def start_maintenance(self):
        if self.maintenance_started or not self.service: return
        self.maintenance_started = True; self.update_schedule()
        QTimer.singleShot(1200, lambda: self.automatic_check(startup=True))

    def update_schedule(self):
        if not self.maintenance_started or self.closed or not self.service: return
        self.timer.stop()
        try:
            pref = self.service.preferences()
            if pref['autoReceive']: self.timer.start(round(pref.get('checkIntervalHours', 6) * 3600000))
            elif self.auto_cancel: self.auto_cancel.set()
        except Exception: self.timer.stop()

    def automatic_check(self, *, startup=False):
        if self.closed or not self.service: return
        try:
            if not startup and not self.service.preferences()['autoReceive']: return
        except Exception: return
        with self.lock:
            if self.auto_cancel is not None: return
            cancel = threading.Event(); self.auto_cancel = cancel
        self.network_pool.submit(self.background_check, cancel)

    def background_check(self, cancel):
        try:
            result = self.service.refresh(cancel)
            if cancel.is_set(): return
            if self.service.preferences()['autoReceive']:
                result = self.service.download(self.service.first_launch, cancel, lambda _: None)
            if not self.closed: self.statusChanged.emit(json.dumps(result, ensure_ascii=False))
        except Exception as exc:
            if not self.closed and not cancel.is_set():
                result = self.service.status(); result['warning'] = '自动检查未完成：' + str(exc)
                self.statusChanged.emit(json.dumps(result, ensure_ascii=False))
        finally:
            with self.lock: self.auto_cancel = None

    @pyqtSlot(str, str)
    def request(self, request_id, payload):
        if not re.fullmatch(r'[a-zA-Z0-9_-]{1,96}', request_id): return
        with self.lock:
            if self.closed or request_id in self.jobs: return
            if len(self.jobs) >= 8 or len(payload) > 128000:
                self.ready.emit(request_id, json.dumps({'ok': False, 'error': '论文服务繁忙，请稍后重试。'})); return
            destination = None
            try:
                value = json.loads(payload)
                if value.get('operation') == 'chooseLibrary':
                    if value.get('args', {}) != {} or self.root_picker is None: raise ValueError()
                    destination = self.root_picker()
                    if not destination:
                        self.ready.emit(request_id, json.dumps({'ok': True, 'value': {'cancelled': True}})); return
                if value.get('operation') == 'importLocal':
                    args = value.get('args', {})
                    if set(args) != {'folder'} or not isinstance(args['folder'], str) or self.import_picker is None: raise ValueError()
                    destination = self.import_picker()
                    if not destination:
                        self.ready.emit(request_id, json.dumps({'ok': True, 'value': {'cancelled': True}})); return
                if value.get('operation') == 'export':
                    args = value.get('args', {})
                    if set(args) != {'id','language','annotated'} or args['language'] not in ('original','translation','analysis') or type(args['annotated']) is not bool or not re.fullmatch(r'[a-zA-Z0-9_-]{1,80}',args['id']): raise ValueError()
                    if self.export_picker is None: raise ValueError()
                    name = args['id'] + {'translation':'-中文译文','original':'-原文','analysis':'-论文解读'}[args['language']] + ('-含批注' if args['annotated'] else '') + '.pdf'
                    destination = self.export_picker(name)
                    if not destination:
                        self.ready.emit(request_id,json.dumps({'ok':True,'value':{'cancelled':True}}));return
            except Exception:
                self.ready.emit(request_id,json.dumps({'ok':False,'error':'无法打开 PDF 保存窗口。'}));return
            cancel = threading.Event(); self.jobs[request_id] = cancel
            pool = self.network_pool if value.get('operation') in ('refresh', 'download', 'downloadSelected') else self.pool
            self.futures[request_id] = pool.submit(self.execute, request_id, payload, cancel, destination)

    def execute(self, request_id, payload, cancel, destination=None):
        def progress(value):
            with self.lock:
                if not self.closed and not cancel.is_set(): self.progress.emit(request_id, json.dumps(value))
        try:
            if self.service is None: raise PaperError('论文存储初始化失败，首次日期未重置。请检查本机存储权限或日期记录后重启软件。')
            if cancel.is_set(): raise PaperError('操作已取消。')
            value = json.loads(payload); op = value['operation']; args = value.get('args', {})
            allowed = {'status': set(), 'refresh': set(), 'download': {'since'}, 'downloadSelected': {'ids'}, 'removeDownload': {'id'}, 'removeLocal': {'id'}, 'autoReceive': {'enabled'}, 'interval': {'hours'}, 'createRecommendationFolder': {'name'}, 'deleteRecommendationFolder': {'folder'}, 'collectRecommendation': {'id','folder'}, 'library': set(), 'chooseLibrary': set(), 'importLocal': {'folder'}, 'createFolder': {'name'}, 'deleteFolder': {'folder'}, 'moveLocal': {'id','folder'}, 'render': {'id', 'language', 'page', 'width'}, 'annotations':{'id','language'}, 'saveAnnotation':{'id','language','revision','item','remove'}, 'export':{'id','language','annotated'}}
            if op not in allowed or not isinstance(args, dict) or set(args) != allowed[op]: raise PaperError('论文请求无效。')
            if op == 'status': result = self.service.status()
            elif op == 'refresh': result = self.service.refresh(cancel)
            elif op == 'download': result = self.service.download(args['since'], cancel, progress)
            elif op == 'downloadSelected': result = self.service.download('0001-01-01', cancel, progress, args['ids'])
            elif op == 'removeDownload': result = self.service.remove_download(args['id'])
            elif op == 'removeLocal':
                self.service.close_documents()
                result = self.service.library().remove(args['id'])
            elif op == 'autoReceive': result = self.service.set_auto_receive(args['enabled'])
            elif op == 'interval': result = self.service.set_interval(args['hours'])
            elif op == 'createRecommendationFolder': result = self.service.recommendation_folder(args['name'])
            elif op == 'deleteRecommendationFolder': result = self.service.delete_recommendation_folder(args['folder'])
            elif op == 'collectRecommendation': result = self.service.collect_recommendation(args['id'], args['folder'])
            elif op == 'library': result = self.service.library_status()
            elif op == 'chooseLibrary': result = self.service.set_library_root(destination)
            elif op == 'importLocal': result = self.service.library().import_files(destination, args['folder'])
            elif op == 'createFolder': result = self.service.library().create_folder(args['name'])
            elif op == 'deleteFolder':
                self.service.close_documents()
                result = self.service.library().delete_folder(args['folder'])
            elif op == 'moveLocal':
                self.service.close_documents()
                result = self.service.library().move(args['id'], args['folder'])
            elif op == 'render': result = self.service.render(args['id'], args['language'], args['page'], args['width'])
            elif op == 'annotations': result = self.service.annotations(args['id'],args['language'])
            elif op == 'saveAnnotation': result = self.service.save_annotation(args['id'],args['language'],args['revision'],args['item'],args['remove'])
            elif op == 'export':
                if destination is None: raise PaperError('请选择 PDF 保存位置。')
                result = self.service.export(args['id'],args['language'],args['annotated'],destination)
            if op in ('autoReceive', 'interval'):
                if op == 'autoReceive' and not args['enabled'] and self.auto_cancel: self.auto_cancel.set()
                self.scheduleChanged.emit()
            response = {'ok': True, 'value': result}
        except PaperError as error:
            response = {'ok': False, 'error': str(error)}
        except Exception:
            response = {'ok': False, 'error': '论文服务暂时不可用，请检查网络后重试。已下载内容仍可离线阅读。'}
        with self.lock:
            self.jobs.pop(request_id, None)
            self.futures.pop(request_id, None)
            if not self.closed and not cancel.is_set(): self.ready.emit(request_id, json.dumps(response, ensure_ascii=False))

    @pyqtSlot(str)
    def cancel(self, request_id):
        with self.lock:
            if request_id in self.jobs:
                self.jobs[request_id].set()
                future = self.futures.get(request_id)
                if future and future.cancel():
                    # Queued work will never reach execute's completion cleanup.
                    # Keep a running native call counted until it really finishes.
                    self.jobs.pop(request_id, None)
                    self.futures.pop(request_id, None)

    def close(self):
        self.timer.stop()
        with self.lock:
            self.closed = True
            if self.auto_cancel: self.auto_cancel.set()
            for event in self.jobs.values(): event.set()
            for future in self.futures.values(): future.cancel()
        self.pool.submit(self.service.close_documents) if self.service else None
        self.pool.shutdown(wait=False, cancel_futures=False)
        self.network_pool.shutdown(wait=False, cancel_futures=True)
