"""Bounded feedback QWebChannel, separate from scientific task bridges."""
from concurrent.futures import ThreadPoolExecutor
import json
import re
import threading
from PyQt6.QtCore import QObject,pyqtSignal,pyqtSlot
from .feedback import FeedbackClient,FeedbackClientError


class FeedbackBridge(QObject):
    ready = pyqtSignal(str,str)
    notification = pyqtSignal(str)
    windowAction = pyqtSignal(str)

    def __init__(self, root, parent=None, *, admin=False, client=None):
        super().__init__(parent)
        self.error = None
        try:
            self.client = client or FeedbackClient(root,admin=admin)
        except FeedbackClientError as exc:
            self.client = None
            self.error = str(exc)
        self.admin = admin
        self.pool = ThreadPoolExecutor(max_workers=1,thread_name_prefix='ptb-feedback')
        self.pending = set()
        self.lock = threading.Lock()
        self.closed = False
        if admin and self.client:
            self.client.subscribe(lambda event:self.notification.emit(json.dumps(event,ensure_ascii=False)) if not self.closed else None)

    @pyqtSlot(str,str)
    def request(self, request_id, payload):
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,96}',request_id):
            return
        with self.lock:
            if self.closed or request_id in self.pending:
                return
            if len(self.pending)>=4 or len(payload)>14_000_000:
                self.ready.emit(request_id,json.dumps({'ok':False,'error':'反馈请求繁忙或图片过大，请稍后再试。'}))
                return
            self.pending.add(request_id)
        self.pool.submit(self.execute,request_id,payload)

    def execute(self, request_id, payload):
        try:
            if self.client is None:
                raise FeedbackClientError(self.error or '反馈服务初始化失败。')
            body = json.loads(payload)
            op,args = body['operation'],body.get('args',{})
            allowed = {'status':set(),'health':set(),'unread':set(),'openDevice':{'email'},'verifyStart':{'email'},'verifyFinish':{'challenge','code'},
                       'connect':{'baseUrl','token'},'conversations':set(),'messages':{'after'},
                       'send':{'text','images','requestId'},'read':{'through'},'resolve':{'owner','status'},
                       'retryMail':{'id'},'resetIdentity':set(),'setMailNotifications':{'mailNotifications'},'createPairing':set(),'listDevices':set(),'revokeDevice':{'deviceId'},'windowAction':{'action'}}
            expected = allowed.get(op)
            if self.admin and op in ('messages','send','read'):
                expected = expected | {'owner'}
            if expected is None or not isinstance(args,dict) or set(args)!=expected:
                raise FeedbackClientError('反馈请求无效。')
            if op=='windowAction' and self.admin and args.get('action')=='quit':
                value={'ok':True}
                self.windowAction.emit('quit')
            else:
                value = self.client.request(op,args)
            result = {'ok':True,'value':value}
        except FeedbackClientError as exc:
            result = {'ok':False,'error':str(exc)}
        except Exception:
            result = {'ok':False,'error':'反馈操作未完成，请稍后重试。'}
        with self.lock:
            self.pending.discard(request_id)
            if not self.closed:
                self.ready.emit(request_id,json.dumps(result,ensure_ascii=False))

    def close(self):
        with self.lock:
            self.closed = True
        if self.client:
            self.client.close()
        self.pool.shutdown(wait=False,cancel_futures=True)
