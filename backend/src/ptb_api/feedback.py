"""Private feedback conversations with optional email verification and delivery."""
from __future__ import annotations

import asyncio
import base64
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
import hashlib
import hmac
from io import BytesIO
import json
from pathlib import Path
import re
import secrets
import sqlite3
import time
import warnings

from fastapi import APIRouter, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ConfigDict, Field
from starlette.responses import JSONResponse, StreamingResponse

MAX_IMAGE = 2_000_000
MAX_REQUEST = 14_000_000
SHANGHAI = timezone(timedelta(hours=8))


class FeedbackError(Exception):
    def __init__(self, code, status=422):
        self.code, self.status = code, status


def email_address(value):
    value = value.strip().lower()
    if not re.fullmatch(r'[a-z0-9.!#$%&\x27*+/=?^_`{|}~-]{1,64}@[a-z0-9](?:[a-z0-9.-]{0,188}[a-z0-9])?\.[a-z]{2,63}', value):
        raise FeedbackError('invalid_email')
    return value


def image_data(item):
    """Decode and reencode rather than serving untrusted image containers."""
    from PIL import Image
    try:
        raw = base64.b64decode(item['data'], validate=True)
        if not 0 < len(raw) <= MAX_IMAGE:
            raise ValueError()
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(BytesIO(raw)) as image:
                if image.format not in ('PNG', 'JPEG', 'WEBP') or image.width * image.height > 12_000_000:
                    raise ValueError()
                image.load()
                out = BytesIO()
                if image.mode in ('RGBA', 'LA') or 'transparency' in image.info:
                    image.convert('RGBA').save(out, format='PNG')
                    mime = 'image/png'
                else:
                    image.convert('RGB').save(out, format='JPEG', quality=88)
                    mime = 'image/jpeg'
                body = out.getvalue()
                if len(body) > MAX_IMAGE:
                    raise ValueError()
                return mime, body
    except Exception:
        raise FeedbackError('invalid_image') from None


class FeedbackService:
    def __init__(self, database, *, secret, admin_token, mailer=None, clock=time.time, daily_hour=9, allow_unverified=False):
        if len(secret) < 32 or len(admin_token) < 32 or not 0 <= daily_hour <= 23:
            raise ValueError('Feedback requires private keys and a valid schedule')
        self.database = Path(database)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self.secret = secret.encode()
        self.admin_hash = hashlib.sha256(admin_token.encode()).hexdigest()
        self.mailer, self.clock, self.daily_hour = mailer, clock, daily_hour
        self.allow_unverified = allow_unverified
        self.initialize()
        from .feedback_admin import AdminDevices
        self.admin_devices = AdminDevices(self.database.parent/'feedback-admin-devices.json', secret=self.secret, clock=self.clock, error=FeedbackError)

    @contextmanager
    def db(self):
        conn = sqlite3.connect(self.database, timeout=15)
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA foreign_keys=ON')
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def initialize(self):
        if self.database.exists():
            # Never add feedback tables to an existing research or unknown DB.
            with sqlite3.connect(self.database.resolve().as_uri()+'?mode=ro',uri=True) as existing:
                tables={row[0] for row in existing.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                if 'feedback_meta' not in tables:
                    raise ValueError('Selected existing database is not a feedback database')
                version=existing.execute('SELECT version FROM feedback_meta').fetchone()
                if version!=(2,):
                    raise ValueError('Feedback database version requires review')
        with self.db() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.executescript('''
            CREATE TABLE IF NOT EXISTS feedback_meta(version INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS identities(id TEXT PRIMARY KEY,email TEXT NOT NULL,email_verified INTEGER NOT NULL DEFAULT 0,created REAL NOT NULL,status TEXT NOT NULL DEFAULT 'open');
            CREATE TABLE IF NOT EXISTS sessions(hash TEXT PRIMARY KEY,owner TEXT NOT NULL REFERENCES identities(id),expires REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS challenges(id TEXT PRIMARY KEY,email TEXT NOT NULL,hash TEXT NOT NULL,expires REAL NOT NULL,attempts INTEGER NOT NULL DEFAULT 0,created REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS messages(id INTEGER PRIMARY KEY AUTOINCREMENT,owner TEXT NOT NULL REFERENCES identities(id),role TEXT NOT NULL,text TEXT NOT NULL,created REAL NOT NULL,request_id TEXT NOT NULL,seen INTEGER NOT NULL DEFAULT 0,UNIQUE(owner,role,request_id));
            CREATE TABLE IF NOT EXISTS attachments(id TEXT PRIMARY KEY,message INTEGER NOT NULL REFERENCES messages(id),mime TEXT NOT NULL,data BLOB NOT NULL);
            CREATE TABLE IF NOT EXISTS outbox(id INTEGER PRIMARY KEY AUTOINCREMENT,key TEXT UNIQUE NOT NULL,recipient TEXT NOT NULL,subject TEXT NOT NULL,body TEXT NOT NULL,state TEXT NOT NULL DEFAULT 'pending',attempts INTEGER NOT NULL DEFAULT 0,next_at REAL NOT NULL,lease_until REAL NOT NULL DEFAULT 0,message INTEGER REFERENCES messages(id),expires REAL,error TEXT);
            CREATE TABLE IF NOT EXISTS checks(day TEXT PRIMARY KEY,highwater INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY AUTOINCREMENT,payload TEXT NOT NULL,created REAL NOT NULL);
            CREATE INDEX IF NOT EXISTS feedback_messages_owner ON messages(owner,id);
            CREATE INDEX IF NOT EXISTS feedback_outbox_due ON outbox(state,next_at);
            CREATE UNIQUE INDEX IF NOT EXISTS feedback_verified_email ON identities(email) WHERE email_verified=1;
            ''')
            row = db.execute('SELECT version FROM feedback_meta').fetchone()
            if row is None:
                db.execute('INSERT INTO feedback_meta VALUES(2)')
            elif row['version'] != 2:
                raise ValueError('Feedback database version requires review')
        try:
            self.database.chmod(0o600)
        except OSError:
            pass

    def digest(self, value):
        return hmac.new(self.secret, value.encode(), hashlib.sha256).hexdigest()

    def open_device(self, email, token):
        if not self.allow_unverified:
            raise FeedbackError('email_verification_required',403)
        email=email_address(email)
        now=self.clock()
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            existing=db.execute('SELECT i.id,i.email,i.email_verified,s.expires FROM sessions s JOIN identities i ON i.id=s.owner WHERE s.hash=?',(self.digest(token),)).fetchone()
            if existing:
                if existing['email_verified'] and existing['expires']<=now:
                    raise FeedbackError('feedback_unauthorized',401)
                if existing['email']!=email:
                    raise FeedbackError('identity_key_conflict',409)
                owner=existing['id']
            else:
                minute=db.execute('SELECT COUNT(*) FROM identities WHERE created>?',(now-60,)).fetchone()[0]
                daily=db.execute('SELECT COUNT(*) FROM identities WHERE created>?',(now-86400,)).fetchone()[0]
                if minute>=20 or daily>=1000:
                    raise FeedbackError('identity_rate_limited',429)
                owner=secrets.token_urlsafe(18)
                db.execute('INSERT INTO identities(id,email,created) VALUES(?,?,?)',(owner,email,now))
                db.execute('INSERT INTO sessions VALUES(?,?,?)',(self.digest(token),owner,now+180*86400))
        return dict(token=token,email=email,owner=owner,emailVerified=bool(existing['email_verified']) if existing else False)

    def start_verification(self, email):
        if self.mailer is None:
            raise FeedbackError('mail_not_configured', 503)
        email = email_address(email)
        now = self.clock()
        code = f'{secrets.randbelow(1_000_000):06d}'
        challenge = secrets.token_urlsafe(24)
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            recent = db.execute('SELECT MAX(created) last,COUNT(*) count FROM challenges WHERE email=? AND created>?', (email, now-3600)).fetchone()
            total = db.execute('SELECT COUNT(*) FROM challenges WHERE created>?', (now-60,)).fetchone()[0]
            daily = db.execute('SELECT COUNT(*) FROM challenges WHERE created>?', (now-86400,)).fetchone()[0]
            if recent['count'] >= 5 or recent['last'] is not None and now-recent['last'] < 60 or total >= 20 or daily>=1000:
                raise FeedbackError('verification_rate_limited', 429)
            db.execute('INSERT INTO challenges(id,email,hash,expires,created) VALUES(?,?,?,?,?)', (challenge,email,self.digest(challenge+':'+code),now+600,now))
            body = f'PhoneticToolbox 问题反馈邮箱验证码：{code}\n\n10 分钟内有效。验证后可查看并继续自己的反馈会话。\n如果您没有请求验证，请忽略此邮件。'
            db.execute('INSERT INTO outbox(key,recipient,subject,body,next_at,expires) VALUES(?,?,?,?,?,?)', ('verify-'+challenge,email,'PhoneticToolbox 反馈邮箱验证',body,now,now+600))
        return {'challenge':challenge,'expiresIn':600,'retryAfter':60}

    def verify(self, challenge, code, current_owner=None):
        now = self.clock()
        result = None
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT * FROM challenges WHERE id=?', (challenge,)).fetchone()
            if row is None or row['expires'] <= now or row['attempts'] >= 5:
                raise FeedbackError('verification_expired')
            if not hmac.compare_digest(row['hash'], self.digest(challenge+':'+code)):
                db.execute('UPDATE challenges SET attempts=attempts+1 WHERE id=?', (challenge,))
            else:
                owner = db.execute('SELECT id FROM identities WHERE email=? AND email_verified=1', (row['email'],)).fetchone()
                candidate = db.execute('SELECT id FROM identities WHERE id=? AND email=?',(current_owner,row['email'])).fetchone() if current_owner else None
                if owner and candidate and owner['id']!=candidate['id']:
                    raise FeedbackError('email_identity_conflict',409)
                owner = owner['id'] if owner else candidate['id'] if candidate else secrets.token_urlsafe(18)
                db.execute('INSERT OR IGNORE INTO identities(id,email,email_verified,created) VALUES(?,?,1,?)', (owner,row['email'],now))
                db.execute('UPDATE identities SET email_verified=1 WHERE id=?',(owner,))
                token = secrets.token_urlsafe(32)
                db.execute('INSERT INTO sessions VALUES(?,?,?)', (self.digest(token),owner,now+180*86400))
                db.execute('DELETE FROM challenges WHERE email=?', (row['email'],))
                db.execute("UPDATE outbox SET state='expired',body='' WHERE recipient=? AND key LIKE 'verify-%' AND state='pending'", (row['email'],))
                result = {'token':token,'email':row['email'],'owner':owner,'emailVerified':True}
        if result is None:
            raise FeedbackError('verification_invalid')
        return result

    def authenticate(self, token, admin=False):
        if admin:
            if not token or not (hmac.compare_digest(hashlib.sha256(token.encode()).hexdigest(), self.admin_hash) or self.admin_devices.authenticate(token)):
                raise FeedbackError('admin_unauthorized', 401)
            return None
        with self.db() as db:
            row = db.execute('SELECT s.owner,s.expires,i.email_verified FROM sessions s JOIN identities i ON i.id=s.owner WHERE s.hash=?', (self.digest(token),)).fetchone()
            # Device-only identities cannot recover by email while SMTP is paused.
            # Keep the original device credential usable until explicit revocation.
            if row and (not row['email_verified'] or row['expires']>self.clock()):
                db.execute('UPDATE sessions SET expires=? WHERE hash=?',(self.clock()+180*86400,self.digest(token)))
            else:
                row=None
        if row is None:
            raise FeedbackError('feedback_unauthorized', 401)
        return row['owner']

    def send(self, owner, role, text, images, request_id):
        text = text.strip()
        if len(text) > 12000 or not text and not images or len(images) > 4 or not re.fullmatch(r'[A-Za-z0-9_-]{8,96}', request_id):
            raise FeedbackError('invalid_message')
        converted = [image_data(item) for item in images]
        if sum(len(data) for _,data in converted) > 6_000_000:
            raise FeedbackError('attachment_budget')
        now = self.clock()
        with self.admin_devices.transaction() as preferences, self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            identity = db.execute('SELECT * FROM identities WHERE id=?', (owner,)).fetchone()
            if identity is None:
                raise FeedbackError('conversation_unavailable',404)
            existing = db.execute('SELECT id,text FROM messages WHERE owner=? AND role=? AND request_id=?', (owner,role,request_id)).fetchone()
            if existing:
                old = db.execute('SELECT mime,data FROM attachments WHERE message=? ORDER BY id', (existing['id'],)).fetchall()
                if existing['text'] != text or sorted((r['mime'],bytes(r['data'])) for r in old) != sorted(converted):
                    raise FeedbackError('message_key_conflict',409)
                return {'id':existing['id'],'duplicate':True}
            recent = db.execute('SELECT COUNT(*) FROM messages WHERE owner=? AND role=? AND created>?', (owner,role,now-60)).fetchone()[0]
            used = db.execute('SELECT COALESCE(SUM(length(a.data)),0) FROM attachments a JOIN messages m ON m.id=a.message WHERE m.owner=?', (owner,)).fetchone()[0]
            text_used = db.execute('SELECT COALESCE(SUM(length(CAST(text AS BLOB))),0) FROM messages WHERE owner=?',(owner,)).fetchone()[0]
            if recent >= 15:
                raise FeedbackError('message_rate_limited',429)
            if used+text_used+len(text.encode())+sum(len(body) for _,body in converted) > 100_000_000:
                raise FeedbackError('conversation_budget',413)
            mid = db.execute('INSERT INTO messages(owner,role,text,created,request_id) VALUES(?,?,?,?,?)', (owner,role,text,now,request_id)).lastrowid
            for mime,body in converted:
                db.execute('INSERT INTO attachments VALUES(?,?,?,?)', (secrets.token_urlsafe(18),mid,mime,body))
            db.execute("UPDATE identities SET status='open' WHERE id=?", (owner,))
            if role == 'admin' and self.mailer is not None and identity['email_verified'] and self.mail_notifications(preferences, owner):
                body = f'您的 PhoneticToolbox 反馈收到回复：\n\n{text or "（图片回复）"}\n\n'+('本次回复的图片附在邮件中，也可在应用的关于 → 问题反馈中查看。\n\n' if converted else '')+'打开 PhoneticToolbox → 关于 → 问题反馈，可查看完整会话并继续交流。\n本邮件用于反馈通知，直接回复邮箱暂不自动进入应用会话。'
                db.execute('INSERT INTO outbox(key,recipient,subject,body,next_at,message) VALUES(?,?,?,?,?,?)', ('reply-'+owner+'-'+str(mid),identity['email'],'PhoneticToolbox：您的反馈收到回复',body,now,mid))
            event = {'kind': 'new_feedback' if role == 'user' else 'conversation_changed', 'owner': owner, 'messageId': mid, 'count': 1, 'conversations': 1}
            db.execute('INSERT INTO events(payload,created) VALUES(?,?)', (json.dumps(event,separators=(',',':')),now))
        return {'id':mid,'duplicate':False}

    def messages(self, owner, after=0):
        with self.admin_devices.transaction() as preferences, self.db() as db:
            notifications = self.mail_notifications(preferences, owner)
            identity = db.execute('SELECT * FROM identities WHERE id=?', (owner,)).fetchone()
            if identity is None:
                raise FeedbackError('conversation_unavailable',404)
            rows = db.execute('SELECT * FROM messages WHERE owner=? AND id>? ORDER BY id LIMIT 100', (owner,after)).fetchall()
            result = []
            for row in rows:
                delivery = db.execute('SELECT state,error FROM outbox WHERE message=?', (row['id'],)).fetchone()
                images = [dict(id=a['id'],mime=a['mime'],data=base64.b64encode(a['data']).decode()) for a in db.execute('SELECT * FROM attachments WHERE message=? ORDER BY id', (row['id'],))]
                result.append(dict(id=row['id'],role=row['role'],text=row['text'],created=row['created'],images=images,
                                   mailState=delivery['state'] if delivery else ('manual' if notifications else 'disabled') if row['role']=='admin' else None,mailError=delivery['error'] if delivery else None))
                if sum(len(i['data']) for m in result for i in m['images']) > 9_000_000:
                    break
        return dict(owner=owner,email=identity['email'],emailVerified=bool(identity['email_verified']),mailConfigured=self.mailer is not None,mailNotifications=notifications,status=identity['status'],messages=result,
                    cursor=result[-1]['id'] if result else after,hasMore=bool(rows and rows[-1]['id']> (result[-1]['id'] if result else after)) or len(rows)==100)

    def conversations(self):
        with self.db() as db:
            rows = db.execute('''SELECT i.*,MAX(m.id) lastId,MAX(m.created) updated,
                SUM(CASE WHEN m.role='user' AND m.seen=0 THEN 1 ELSE 0 END) unread
                FROM identities i JOIN messages m ON m.owner=i.id GROUP BY i.id ORDER BY lastId DESC LIMIT 1000''').fetchall()
            return [dict(row) for row in rows]

    @staticmethod
    def mail_notifications(state, owner):
        # Additive private JSON metadata keeps the existing SQLite schema intact.
        value = state.get('mailPreferences', {}).get(owner, True)
        if type(value) is not bool:
            raise ValueError('Invalid feedback mail preference')
        return value

    def set_mail_notifications(self, owner, enabled):
        if type(enabled) is not bool:
            raise FeedbackError('invalid_request')
        with self.admin_devices.transaction(write=True) as state, self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            if not db.execute('SELECT 1 FROM identities WHERE id=?', (owner,)).fetchone():
                raise FeedbackError('conversation_unavailable',404)
            state.setdefault('mailPreferences', {})[owner] = enabled
            if not enabled:
                db.execute("UPDATE outbox SET state='skipped',body='',error='mail_notifications_disabled' WHERE message IN (SELECT id FROM messages WHERE owner=?) AND state IN ('pending','failed')", (owner,))
            db.execute('INSERT INTO events(payload,created) VALUES(?,?)', (json.dumps({'kind':'conversation_changed','owner':owner}),self.clock()))
        return {'mailNotifications':enabled}

    def mark_read(self, owner, role, through):
        with self.db() as db:
            db.execute('UPDATE messages SET seen=1 WHERE owner=? AND role=? AND id<=?', (owner,role,through))
        return {'ok':True}

    def unread(self, owner):
        with self.db() as db:
            count=db.execute("SELECT COUNT(*) FROM messages WHERE owner=? AND role='admin' AND seen=0",(owner,)).fetchone()[0]
        return {'unread':count}

    def set_status(self, owner, status):
        if status not in ('open','resolved'):
            raise FeedbackError('invalid_status')
        with self.db() as db:
            changed = db.execute('UPDATE identities SET status=? WHERE id=?', (status,owner)).rowcount
            if not changed:
                raise FeedbackError('conversation_unavailable',404)
        return {'status':status}

    def daily_check(self):
        now = self.clock()
        local = datetime.fromtimestamp(now, SHANGHAI)
        if local.hour < self.daily_hour:
            return None
        day = local.date().isoformat()
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute('SELECT 1 FROM checks WHERE day=?', (day,)).fetchone():
                return None
            last = db.execute('SELECT COALESCE(MAX(highwater),0) FROM checks').fetchone()[0]
            high = db.execute("SELECT COALESCE(MAX(id),0) FROM messages WHERE role='user'").fetchone()[0]
            rows = db.execute("SELECT owner,COUNT(*) count FROM messages WHERE role='user' AND seen=0 AND id>? AND id<=? GROUP BY owner", (last,high)).fetchall()
            db.execute('INSERT INTO checks VALUES(?,?)', (day,high))
            if not rows:
                return None
            payload = dict(kind='daily_feedback',day=day,count=sum(r['count'] for r in rows),conversations=len(rows))
            eid = db.execute('INSERT INTO events(payload,created) VALUES(?,?)', (json.dumps(payload),now)).lastrowid
            return dict(id=eid,**payload)

    def events(self, after):
        with self.db() as db:
            return [dict(id=r['id'],**json.loads(r['payload'])) for r in db.execute('SELECT * FROM events WHERE id>? ORDER BY id LIMIT 100', (after,))]

    def deliver_one(self):
        if self.mailer is None:
            return False
        now = self.clock()
        with self.admin_devices.transaction() as preferences, self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            db.execute("UPDATE outbox SET state='pending' WHERE state='sending' AND lease_until<?", (now,))
            db.execute("UPDATE outbox SET state='expired',body='' WHERE state='pending' AND expires IS NOT NULL AND expires<=?", (now,))
            for owner, enabled in preferences.get('mailPreferences', {}).items():
                if enabled is False:
                    db.execute("UPDATE outbox SET state='skipped',body='',error='mail_notifications_disabled' WHERE message IN (SELECT id FROM messages WHERE owner=?) AND state='pending'", (owner,))
            row = db.execute("SELECT * FROM outbox WHERE state='pending' AND next_at<=? ORDER BY id LIMIT 1", (now,)).fetchone()
            if row is None:
                return False
            db.execute("UPDATE outbox SET state='sending',attempts=attempts+1,lease_until=? WHERE id=?", (now+120,row['id']))
            attachments=[(a['mime'],bytes(a['data'])) for a in db.execute('SELECT mime,data FROM attachments WHERE message=? ORDER BY id',(row['message'],))] if row['message'] else []
        try:
            self.mailer.send(row['recipient'],row['subject'],row['body'],row['key'],attachments)
        except Exception:
            with self.db() as db:
                state = 'failed' if row['attempts']+1 >= 12 else 'pending'
                db.execute('UPDATE outbox SET state=?,next_at=?,error=? WHERE id=?', (state,now+min(3600,30*2**row['attempts']),'smtp_delivery_failed',row['id']))
        else:
            with self.db() as db:
                db.execute("UPDATE outbox SET state='sent',body='',error=NULL WHERE id=?", (row['id'],))
        return True

    def retry_mail(self, mid):
        with self.admin_devices.transaction() as preferences, self.db() as db:
            message = db.execute('SELECT owner FROM messages WHERE id=?', (mid,)).fetchone()
            if message and not self.mail_notifications(preferences, message['owner']):
                raise FeedbackError('mail_notifications_disabled',409)
            changed = db.execute("UPDATE outbox SET state='pending',attempts=0,next_at=?,error=NULL WHERE message=? AND state='failed'", (self.clock(),mid)).rowcount
        return {'retried':bool(changed)}


class Input(BaseModel):
    model_config = ConfigDict(extra='forbid')


class MailPreferences(Input):
    mailNotifications: bool = Field(strict=True)


class VerifyStart(Input):
    email: str = Field(max_length=254)


class DeviceStart(VerifyStart):
    deviceToken: str = Field(pattern=r'^[A-Za-z0-9_-]{43,64}$')


class VerifyFinish(Input):
    challenge: str = Field(min_length=16,max_length=80)
    code: str = Field(pattern=r'^\d{6}$')


class ImageInput(Input):
    data: str = Field(max_length=2_666_668)


class MessageInput(Input):
    text: str = Field(default='',max_length=12000)
    images: list[ImageInput] = Field(default_factory=list,max_length=4)
    requestId: str = Field(pattern=r'^[A-Za-z0-9_-]{8,96}$')


class ReadInput(Input):
    through: int = Field(ge=0)


class StatusInput(Input):
    status: str = Field(pattern=r'^(open|resolved)$')


class ManagerPairing(Input):
    code: str = Field(pattern=r'^[0-9]{8}$')
    name: str = Field(default='安卓管理端',max_length=40)


def create_feedback_router(service):
    if service is None:
        class Unavailable:
            mailer=None
            daily_hour=9
            allow_unverified=False
            def __getattr__(self,name):
                def unavailable(*args,**kwargs):
                    raise FeedbackError('feedback_service_unavailable',503)
                return unavailable
        service=Unavailable()
    router = APIRouter(prefix='/api/v1/feedback',tags=['feedback'])

    def auth(request, admin=False):
        header = request.headers.get('authorization','')
        if not header.startswith('Bearer ') or len(header) > 256:
            raise FeedbackError('admin_unauthorized' if admin else 'feedback_unauthorized',401)
        return service.authenticate(header[7:],admin)

    @router.get('/health')
    def health():
        return dict(schema='ptb-feedback/2',mailConfigured=service.mailer is not None,allowUnverified=service.allow_unverified,dailyHour=service.daily_hour,timezone='Asia/Shanghai',instantNotifications=True,adminPairing=True)

    @router.post('/identity')
    def device_start(body: DeviceStart):
        return service.open_device(body.email,body.deviceToken)

    @router.post('/verification')
    def start(body: VerifyStart):
        return service.start_verification(body.email)

    @router.post('/verification/complete')
    def complete(body: VerifyFinish, request: Request):
        owner=auth(request) if request.headers.get('authorization') else None
        return service.verify(body.challenge,body.code,owner)

    @router.get('/conversation')
    def conversation(request: Request, after: int = 0):
        return service.messages(auth(request),max(0,after))

    @router.post('/preferences')
    def preferences(request: Request, body: MailPreferences):
        return service.set_mail_notifications(auth(request),body.mailNotifications)

    @router.get('/unread')
    def unread(request: Request):
        return service.unread(auth(request))

    @router.post('/messages')
    def send(request: Request, body: MessageInput):
        return service.send(auth(request),'user',body.text,[x.model_dump() for x in body.images],body.requestId)

    @router.post('/read')
    def read(request: Request, body: ReadInput):
        return service.mark_read(auth(request),'admin',body.through)

    @router.get('/admin/conversations')
    def admin_conversations(request: Request):
        auth(request,True)
        return {'conversations':service.conversations()}

    @router.post('/admin/devices/pairing')
    def pairing(request: Request):
        auth(request,True)
        return service.admin_devices.start_pairing()

    @router.post('/admin/devices/pairing/complete')
    def pair_complete(body: ManagerPairing):
        return service.admin_devices.complete_pairing(body.code,body.name)

    @router.get('/admin/devices')
    def devices(request: Request):
        auth(request,True)
        return service.admin_devices.list_devices()

    @router.post('/admin/devices/{device_id}/revoke')
    def revoke(device_id: str,request: Request):
        auth(request,True)
        if not re.fullmatch(r'[A-Za-z0-9_-]{24}',device_id):
            raise FeedbackError('device_unavailable',404)
        return service.admin_devices.revoke(device_id)

    @router.get('/admin/conversations/{owner}')
    def admin_conversation(owner: str, request: Request, after: int = 0):
        auth(request,True)
        return service.messages(owner,max(0,after))

    @router.post('/admin/conversations/{owner}/messages')
    def admin_send(owner: str,request: Request,body: MessageInput):
        auth(request,True)
        return service.send(owner,'admin',body.text,[x.model_dump() for x in body.images],body.requestId)

    @router.post('/admin/conversations/{owner}/read')
    def admin_read(owner: str,request: Request,body: ReadInput):
        auth(request,True)
        return service.mark_read(owner,'user',body.through)

    @router.post('/admin/conversations/{owner}/status')
    def status(owner: str,request: Request,body: StatusInput):
        auth(request,True)
        return service.set_status(owner,body.status)

    @router.post('/admin/mail/{mid}/retry')
    def retry(mid: int,request: Request):
        auth(request,True)
        return service.retry_mail(mid)

    @router.get('/admin/events')
    async def stream(request: Request,after: int = 0):
        auth(request,True)
        async def generate():
            cursor = max(0,after)
            heartbeat = 0
            while not await request.is_disconnected():
                try:
                    auth(request,True)
                except FeedbackError:
                    break
                events = await asyncio.to_thread(service.events,cursor)
                for event in events:
                    cursor = event['id']
                    yield 'id: '+str(cursor)+'\ndata: '+json.dumps(event,ensure_ascii=False)+'\n\n'
                heartbeat += 1
                if heartbeat >= 15:
                    heartbeat = 0
                    yield ': keepalive\n\n'
                await asyncio.sleep(1)
        return StreamingResponse(generate(),media_type='text/event-stream',headers={'Cache-Control':'no-store','X-Accel-Buffering':'no'})

    return router


class FeedbackBoundary:
    """Bound all feedback bodies before validation, with no request logging."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or not scope['path'].startswith('/api/v1/feedback'):
            return await self.app(scope,receive,send)
        size = 0
        async def bounded():
            nonlocal size
            value = await receive()
            size += len(value.get('body',b''))
            if size > MAX_REQUEST:
                raise FeedbackError('request_too_large',413)
            return value
        async def private(value):
            if value['type'] == 'http.response.start':
                value['headers'] = list(value.get('headers',[]))+[(b'cache-control',b'no-store'),(b'x-content-type-options',b'nosniff')]
            await send(value)
        await self.app(scope,bounded,private)


def configure_feedback(app, service=None):
    app.include_router(create_feedback_router(service))
    app.add_middleware(FeedbackBoundary)

    @app.exception_handler(FeedbackError)
    async def error(request, exc):
        return JSONResponse({'detail':exc.code},status_code=exc.status)


async def maintain_feedback(service):
    while True:
        try:
            await asyncio.to_thread(service.daily_check)
            await asyncio.to_thread(service.deliver_one)
        except Exception:
            import logging
            logging.getLogger('ptb.feedback').warning('feedback_maintenance_unavailable')
        await asyncio.sleep(2)


def create_feedback_app(service, *, background=True):
    from contextlib import asynccontextmanager
    @asynccontextmanager
    async def lifespan(app):
        task = asyncio.create_task(maintain_feedback(service)) if background else None
        try:
            yield
        finally:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
    app = FastAPI(title='PhoneticToolbox Feedback',version='1',lifespan=lifespan)
    configure_feedback(app,service)

    @app.exception_handler(RequestValidationError)
    async def invalid(request,exc):
        return JSONResponse({'detail':'invalid_request'},status_code=422)

    @app.exception_handler(sqlite3.Error)
    @app.exception_handler(OSError)
    async def unavailable(request,exc):
        return JSONResponse({'detail':'feedback_service_unavailable'},status_code=503)
    return app
