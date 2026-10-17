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
    inboxState = pyqtSignal(int,int)

    def __init__(self, root, parent=None, *, admin=False, client=None, client_info=None):
        super().__init__(parent)
        self.error = None
        try:
            self.client = client or FeedbackClient(root,admin=admin,client_info=client_info)
        except FeedbackClientError as exc:
            self.client = None
            self.error = str(exc)
        self.admin = admin
        self.pool = ThreadPoolExecutor(max_workers=1,thread_name_prefix='ptb-feedback')
        self.pending = set()
        self.lock = threading.Lock()
        self.event_batch = []
        self.event_scheduled = False
        self.closed = False
        if admin and self.client:
            self.client.subscribe(self.receive_event)

    def update_inbox(self, value):
        rows = [row for row in value['conversations'] if not row.get('trashed')]
        self.inboxState.emit(sum(row['unread'] for row in rows),sum(row['unread']>0 for row in rows))
        return rows

    def receive_event(self, event):
        if self.closed:
            return
        if event.get('kind') in ('daily_feedback','new_feedback','conversation_changed','conversation_trashed','conversation_restored','conversation_deleted'):
            with self.lock:
                self.event_batch.append(event)
                if self.event_scheduled:
                    return
                self.event_scheduled = True
            try:
                self.pool.submit(self.publish_pending_events)
            except RuntimeError:
                pass  # The owning window has closed its worker.
        else:
            self.notification.emit(json.dumps(event,ensure_ascii=False))

    def publish_pending_events(self):
        with self.lock:
            events,self.event_batch = self.event_batch,[]
        try:
            rows = self.update_inbox(self.client.request('conversations',{})) if not self.closed else None
        except Exception:
            rows = None
        for event in events:
            self.publish_event(event,rows,refresh=False)
        with self.lock:
            self.event_scheduled = False
            if self.event_batch and not self.closed:
                self.event_scheduled = True
                self.pool.submit(self.publish_pending_events)

    def publish_event(self, event, rows=None, *, refresh=True):
        if self.closed:
            return
        value = dict(event)
        try:
            if refresh:
                rows = self.update_inbox(self.client.request('conversations',{}))
            if rows is None:
                raise FeedbackClientError('反馈状态暂不可用。')
            if value.get('kind') in ('daily_feedback','new_feedback'):
                relevant = [row for row in rows if not value.get('owner') or row['id']==value['owner']]
                value.update(count=sum(row['unread'] for row in relevant),conversations=sum(row['unread']>0 for row in relevant))
        except Exception:
            # Historical event counts cannot establish current unread state.
            value['count'] = 0
        if not self.closed:
            self.notification.emit(json.dumps(value,ensure_ascii=False))

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
            if self.admin:allowed.update(trashConversation={'owner','through'},restoreConversation={'owner'},webStatus=set(),webRequests={'state','before'},webDecision={'request_id','granted_bytes','duration','note'},webQuota={'quota_bytes','owner_id'})
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
                if self.admin and op in ('conversations','connect'):
                    self.update_inbox(value)
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
