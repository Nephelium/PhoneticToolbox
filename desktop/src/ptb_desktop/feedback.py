"""Native feedback networking and OS-protected credentials."""
import base64
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import re
import secrets
import sys
import threading
import time
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, build_opener, HTTPRedirectHandler
import uuid

BASE_URL = 'https://www.phonetictoolbox.com/api/v1/feedback'
ERRORS = {
    'conversation_changed':'会话有新消息，请刷新后重新确认。',
    'conversation_trashed':'会话已移入回收站，请恢复后再回复。',
    'conversation_mail_busy':'此会话正在发送邮件，请稍后再移入回收站。',
    'conversation_unavailable':'会话已不存在，请刷新会话列表。',
    'feedback_trash_unavailable':'回收站尚未在此服务器启用，请先更新反馈服务。',
    'web_operations_unavailable':'网页版管理暂未连接，反馈功能仍可使用。',
    'capacity_request_closed':'这条申请已处理，请刷新查看结果。',
    'capacity_not_increase':'批准额度应高于用户当前额度。',
    'disk_space_low':'服务器可用空间不足，请先清理或降低批准额度。',
    'web_server_busy':'服务器繁忙，请稍后重试。',
    "mail_notifications_disabled":"用户已关闭邮件提醒，请在应用会话中回复。",
    'invalid_email':'请输入有效邮箱地址。',
    'mail_not_configured':'服务器的邮件服务尚未配置，请稍后再试。',
    'verification_rate_limited':'验证码发送过于频繁，请稍后再试。',
    'verification_expired':'验证码已过期或尝试次数已用完，请重新获取。',
    'verification_invalid':'验证码不正确，请检查邮件。',
    'feedback_unauthorized':'反馈身份已过期，请重新验证邮箱。',
    'admin_unauthorized':'管理员凭据无效，请重新连接。',
    'invalid_image':'图片无法读取或超过限制，请选择 PNG、JPEG 或 WebP，每张不超过 2 MB。',
    'invalid_message':'请输入文字或添加图片，文字最多 12000 字，每次最多 4 张图片。',
    'message_key_conflict':'这条消息的重试内容发生变化，请重新发送。',
    'message_rate_limited':'消息发送过于频繁，请稍后再试。',
    'conversation_budget':'会话图片已达到存储限额，请联系维护者。',
    'email_verification_required':'服务器要求先验证邮箱，请获取验证码。',
    'identity_rate_limited':'首次反馈过于频繁，请稍后再试。',
    'identity_key_conflict':'本机待开通身份与邮箱不一致，请保留现有凭据并重试。',
    'email_identity_conflict':'该邮箱已有经过验证的其他会话。当前会话仍保留，请联系维护者处理。',
}


class FeedbackClientError(Exception):
    pass


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):
        raise FeedbackClientError('反馈服务地址发生重定向，已停止请求。')


def protected(raw, decrypt=False):
    if sys.platform != 'win32':
        raise FeedbackClientError('此版本的反馈凭据保护仅完成 Windows 实现。')
    class Blob(ctypes.Structure):
        _fields_ = [('size',wintypes.DWORD),('data',ctypes.POINTER(ctypes.c_ubyte))]
    buffer = ctypes.create_string_buffer(raw)
    source = Blob(len(raw),ctypes.cast(buffer,ctypes.POINTER(ctypes.c_ubyte)))
    destination = Blob()
    crypt = ctypes.WinDLL('crypt32',use_last_error=True)
    kernel = ctypes.WinDLL('kernel32',use_last_error=True)
    function = crypt.CryptUnprotectData if decrypt else crypt.CryptProtectData
    function.argtypes = [ctypes.POINTER(Blob),ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,wintypes.DWORD,ctypes.POINTER(Blob)]
    function.restype = wintypes.BOOL
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.LocalFree.restype = ctypes.c_void_p
    if not function(ctypes.byref(source),None,None,None,None,1,ctypes.byref(destination)):
        raise FeedbackClientError('无法使用系统保护反馈凭据。')
    try:
        return ctypes.string_at(destination.data,destination.size)
    finally:
        kernel.LocalFree(destination.data)


def valid_base(value, *, test=False):
    part = urlsplit(value)
    secure = part.scheme == 'https' or test and part.scheme=='http' and part.hostname in ('127.0.0.1','localhost')
    if not secure or not part.netloc or part.username or part.password or part.query or part.fragment or part.path.rstrip('/')!='/api/v1/feedback':
        raise FeedbackClientError('服务地址必须是 HTTPS 的 /api/v1/feedback 地址。')
    return value.rstrip('/')


class FeedbackClient:
    def __init__(self, root, *, admin=False, base_url=BASE_URL, test=False, client_info=None):
        self.root = Path(root)
        self.root.mkdir(parents=True,exist_ok=True)
        self.admin, self.test = admin,test
        self.client_info = dict(client_info or {'source':'desktop'})
        self.base_url = valid_base(base_url,test=test)
        self.state = {'email':'','token':'','cursor':0}
        self.path = self.root / ('manager-identity.bin' if admin else 'identity.bin')
        self.lock = threading.RLock()
        self.opener = build_opener(NoRedirect())
        self.stop = threading.Event()
        self.event_thread = None
        self.event_response = None
        if self.path.exists():
            try:
                self.state = json.loads(protected(self.path.read_bytes(),True))
                self.base_url = valid_base(self.state.get('baseUrl',base_url),test=test)
            except Exception:
                raise FeedbackClientError('反馈凭据无法读取，请保留现有文件并检查 Windows 用户身份。') from None

    def save(self):
        raw = json.dumps({**self.state,'baseUrl':self.base_url}).encode()
        temp = self.path.with_suffix('.'+uuid.uuid4().hex+'.tmp')
        temp.write_bytes(protected(raw))
        os.replace(temp,self.path)

    def status(self):
        return dict(connected=bool(self.state.get('token')),verified=bool(self.state.get('emailVerified',False)),owner=self.state.get('owner',''),email=self.state.get('email',''),baseUrl=self.base_url,admin=self.admin,platform='windows')

    def call(self, path, body=None, *, authenticate=True):
        if not re.fullmatch(r'/[A-Za-z0-9_/?=&.-]+',path) or '..' in path:
            raise FeedbackClientError('反馈请求无效。')
        data = json.dumps(body,ensure_ascii=False).encode() if body is not None else None
        if data and len(data)>14_000_000:
            raise FeedbackClientError('本次图片总大小超过限制。')
        headers = {'Accept':'application/json','Content-Type':'application/json','User-Agent':'PhoneticToolbox-Feedback/1'}
        if path == '/messages' and not self.admin:
            for key, name in [('source','Source'),('version','Version'),('packageKind','Package')]:
                if self.client_info.get(key):
                    headers['X-PTB-Feedback-'+name] = self.client_info[key]
        if authenticate:
            headers['Authorization'] = 'Bearer '+self.state.get('token','')
        request = Request(self.base_url+path,data=data,headers=headers)
        try:
            with self.opener.open(request,timeout=20) as response:
                raw = response.read(18_000_001)
                if response.geturl()!=request.full_url or len(raw)>18_000_000:
                    raise FeedbackClientError('反馈响应无效或过大。')
                return json.loads(raw)
        except HTTPError as exc:
            try:
                code = json.loads(exc.read(4096)).get('detail','')
            except Exception:
                code = ''
            if exc.code in (404,405) and path.startswith('/admin/conversations/') and path.endswith(('/trash','/restore')) and code in ('','Not Found','Method Not Allowed'):
                code = 'feedback_trash_unavailable'
            if path.startswith('/admin/web/') and exc.code in (404,502,503,504):
                code='web_operations_unavailable'
            raise FeedbackClientError(ERRORS.get(code,'反馈服务暂时不可用，请稍后再试。')) from None
        except FeedbackClientError:
            raise
        except Exception:
            if path.startswith('/admin/web/'):
                raise FeedbackClientError('无法读取网页管理数据，请检查连接后刷新。反馈会话可单独使用。') from None
            raise FeedbackClientError('无法连接反馈服务器，请检查网络。未发送内容仍保留在此设备。') from None

    def request(self, op, args):
        with self.lock:
            if op == 'status':
                return self.status()
            if op == 'health':
                return self.call('/health',authenticate=False)
            if op == 'unread' and not self.admin:
                return self.call('/unread') if self.state.get('token') else {'unread':0}
            if op == 'verifyStart' and not self.admin:
                return self.call('/verification',{'email':args['email']},authenticate=False)
            if op == 'openDevice' and not self.admin:
                email=args['email'].strip().lower()
                pending=self.state.get('pendingIdentity')
                if not pending or pending['email']!=email:
                    pending={'email':email,'deviceToken':secrets.token_urlsafe(32)}
                    self.state['pendingIdentity']=pending
                    self.save()
                result=self.call('/identity',pending,authenticate=False)
                self.state.update(result)
                self.state.pop('pendingIdentity',None)
                self.save()
                return self.status()
            if op == 'verifyFinish' and not self.admin:
                result = self.call('/verification/complete',args,authenticate=bool(self.state.get('token')))
                self.state.update(result)
                self.save()
                return self.status()
            if op == 'connect' and self.admin:
                base = valid_base(args['baseUrl'],test=self.test)
                token = args['token'].strip()
                if not 32<=len(token)<=128:
                    raise FeedbackClientError('请输入完整的管理员凭据。')
                previous = self.base_url,self.state.copy()
                self.base_url = base
                self.state = {'token':token,'cursor':0,'email':''}
                try:
                    result = self.call('/admin/conversations')
                    self.save()
                except Exception:
                    self.base_url,self.state = previous
                    raise
                return result
            if op == 'conversations' and self.admin:
                return self.call('/admin/conversations?includeTrash=true')
            if self.admin and op in ('webStatus','webRequests','webDecision','webQuota'):
                if op=='webStatus':return self.call('/admin/web/status')
                if op=='webRequests':
                    state=args.get('state','pending');before=args.get('before')
                    if state not in ('pending','approved','declined'):raise FeedbackClientError('申请筛选无效。')
                    if before is not None:before=str(uuid.UUID(before))
                    return self.call('/admin/web/requests?state='+state+('&before='+before if before else ''))
                return self.call('/admin/web/'+('decision' if op=='webDecision' else 'quota'),args)
            if op == 'createPairing' and self.admin:
                return self.call('/admin/devices/pairing',{})
            if op == 'listDevices' and self.admin:
                return self.call('/admin/devices')
            if op == 'revokeDevice' and self.admin:
                device=args['deviceId']
                if not isinstance(device,str) or not re.fullmatch(r'[A-Za-z0-9_-]{24}',device):
                    raise FeedbackClientError('设备身份无效。')
                return self.call('/admin/devices/'+device+'/revoke',{})
            owner = args.get('owner','')
            if owner and not re.fullmatch(r'[A-Za-z0-9_-]{8,80}',owner):
                raise FeedbackClientError('会话身份无效。')
            prefix = '/admin/conversations/'+owner if self.admin else ''
            if self.admin and op in ('messages','send','read','resolve','trashConversation','restoreConversation') and not owner:
                raise FeedbackClientError('请选择用户会话。')
            if op == 'messages':
                after = args.get('after',0)
                if type(after) is not int or after<0:
                    raise FeedbackClientError('会话位置无效。')
                return self.call((prefix if self.admin else '/conversation')+'?after='+str(after))
            if op == 'setMailNotifications' and not self.admin and type(args.get('mailNotifications')) is bool:
                return self.call('/preferences',{'mailNotifications':args['mailNotifications']})
            if op == 'send':
                body = {key:args[key] for key in ('text','images','requestId')}
                return self.call(prefix+'/messages' if self.admin else '/messages',body)
            if op == 'read':
                return self.call(prefix+'/read' if self.admin else '/read',{'through':args['through']})
            if op == 'resolve' and self.admin:
                return self.call(prefix+'/status',{'status':args['status']})
            if op == 'trashConversation' and self.admin:
                if type(args.get('through')) is not int or args['through'] < 1:
                    raise FeedbackClientError('会话位置无效。')
                return self.call(prefix+'/trash',{'through':args['through']})
            if op == 'restoreConversation' and self.admin:
                return self.call(prefix+'/restore',{})
            if op == 'retryMail' and self.admin and type(args['id']) is int and args['id']>0:
                return self.call('/admin/mail/'+str(args['id'])+'/retry',{})
            if op == 'resetIdentity' and not self.admin:
                self.state = {'token':'','email':'','cursor':0}
                self.save()
                return self.status()
            raise FeedbackClientError('反馈操作无效。')

    def subscribe(self, callback):
        if not self.admin or self.event_thread and self.event_thread.is_alive():
            return
        def run():
            while not self.stop.is_set():
                if not self.state.get('token'):
                    self.stop.wait(2)
                    continue
                try:
                    cursor = int(self.state.get('cursor',0))
                    base,token=self.base_url,self.state['token']
                    request = Request(base+'/admin/events?after='+str(cursor),headers={
                        'Authorization':'Bearer '+token,'Accept':'text/event-stream'})
                    with self.opener.open(request,timeout=25) as response:
                        self.event_response=response
                        callback({'kind':'connection','online':True})
                        while not self.stop.is_set():
                            line = response.readline(8193)
                            if base!=self.base_url or token!=self.state.get('token'):
                                break
                            if not line or len(line)>8192:
                                break
                            if line.startswith(b'data: '):
                                event = json.loads(line[6:])
                                if type(event.get('id')) is not int or event['id']<=cursor:
                                    continue
                                callback(event)
                                with self.lock:
                                    self.state['cursor'] = event['id']
                                    self.save()
                                cursor = event['id']
                except Exception:
                    if not self.stop.is_set():
                        callback({'kind':'connection','online':False})
                finally:
                    self.event_response=None
                self.stop.wait(3)
        self.event_thread = threading.Thread(target=run,name='ptb-feedback-events',daemon=True)
        self.event_thread.start()

    def close(self):
        self.stop.set()
        response=self.event_response
        if response is not None:
            try:
                import socket
                response.fp.raw._sock.shutdown(socket.SHUT_RDWR)
            except (AttributeError,OSError):
                pass
