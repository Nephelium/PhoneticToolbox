"""M18 serialized native network/PDF capability."""
from concurrent.futures import ThreadPoolExecutor
import json
import re
import threading
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
from .papers import PaperService, PaperError


class PapersBridge(QObject):
    ready = pyqtSignal(str, str)
    progress = pyqtSignal(str, str)

    def __init__(self, root, parent=None, *, export_picker=None):
        super().__init__(parent)
        try: self.service = PaperService(root)
        except Exception: self.service = None
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='ptb-papers')
        self.jobs = {}; self.lock = threading.RLock(); self.closed = False
        self.export_picker = export_picker

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
                if value.get('operation') == 'export':
                    args = value.get('args', {})
                    if set(args) != {'id','language','annotated'} or args['language'] not in ('original','translation') or type(args['annotated']) is not bool or not re.fullmatch(r'[a-zA-Z0-9_-]{1,80}',args['id']): raise ValueError()
                    if self.export_picker is None: raise ValueError()
                    name = args['id'] + ('-中文译文' if args['language']=='translation' else '-原文') + ('-含批注' if args['annotated'] else '') + '.pdf'
                    destination = self.export_picker(name)
                    if not destination:
                        self.ready.emit(request_id,json.dumps({'ok':True,'value':{'cancelled':True}}));return
            except Exception:
                self.ready.emit(request_id,json.dumps({'ok':False,'error':'无法打开 PDF 保存窗口。'}));return
            cancel = threading.Event(); self.jobs[request_id] = cancel
            self.pool.submit(self.execute, request_id, payload, cancel, destination)

    def execute(self, request_id, payload, cancel, destination=None):
        def progress(value):
            with self.lock:
                if not self.closed and not cancel.is_set(): self.progress.emit(request_id, json.dumps(value))
        try:
            if self.service is None: raise PaperError('论文存储初始化失败，首次日期未重置。请检查本机存储权限或日期记录后重启软件。')
            if cancel.is_set(): raise PaperError('操作已取消。')
            value = json.loads(payload); op = value['operation']; args = value.get('args', {})
            allowed = {'status': set(), 'refresh': set(), 'download': {'since'}, 'render': {'id', 'language', 'page', 'width'}, 'annotations':{'id','language'}, 'saveAnnotation':{'id','language','revision','item','remove'}, 'export':{'id','language','annotated'}}
            if op not in allowed or not isinstance(args, dict) or set(args) != allowed[op]: raise PaperError('论文请求无效。')
            if op == 'status': result = self.service.status()
            elif op == 'refresh': result = self.service.refresh(cancel)
            elif op == 'download': result = self.service.download(args['since'], cancel, progress)
            elif op == 'render': result = self.service.render(args['id'], args['language'], args['page'], args['width'])
            elif op == 'annotations': result = self.service.annotations(args['id'],args['language'])
            elif op == 'saveAnnotation': result = self.service.save_annotation(args['id'],args['language'],args['revision'],args['item'],args['remove'])
            elif op == 'export':
                if destination is None: raise PaperError('请选择 PDF 保存位置。')
                result = self.service.export(args['id'],args['language'],args['annotated'],destination)
            response = {'ok': True, 'value': result}
        except PaperError as error:
            response = {'ok': False, 'error': str(error)}
        except Exception:
            response = {'ok': False, 'error': '论文服务暂时不可用，请检查网络后重试。已下载内容仍可离线阅读。'}
        with self.lock:
            self.jobs.pop(request_id, None)
            if not self.closed and not cancel.is_set(): self.ready.emit(request_id, json.dumps(response, ensure_ascii=False))

    @pyqtSlot(str)
    def cancel(self, request_id):
        with self.lock:
            if request_id in self.jobs: self.jobs[request_id].set()

    def close(self):
        with self.lock:
            self.closed = True
            for event in self.jobs.values(): event.set()
        self.pool.shutdown(wait=False, cancel_futures=True)
