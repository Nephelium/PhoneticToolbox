"""Web-only presence and manually reviewed capacity. No public approval endpoint."""
from datetime import datetime, timedelta, timezone
from typing import Literal
from uuid import UUID, uuid4
from pydantic import Field
from .models import WireModel
from .quota import StorageError
from .web_status import ServerLoadView


class PresenceView(WireModel):
    online_accounts: int
    window_seconds: int = 90
    measured_at: float
    server: ServerLoadView | None = None


class CapacityRequest(WireModel):
    requested_bytes: int = Field(gt=0, le=10_000_000_000, strict=True)
    reason: str = Field(min_length=1, max_length=500)
    duration: Literal['today','7d','30d'] = 'today'


class CapacityRequestView(WireModel):
    id: str
    requested_bytes: int
    state: Literal['pending', 'approved', 'declined']
    created_at: datetime
    granted_bytes: int | None = None
    decision_note: str | None = None
    duration: Literal['today','7d','30d']
    granted_duration: Literal['today','7d','30d'] | None = None
    expires_at: datetime | None = None


def public(row):
    return None if not row else CapacityRequestView(**{k: str(row[k]) if k=='id' else row[k]
        for k in CapacityRequestView.model_fields})


def presence(service):
    with service._read() as c:
        now=service.now(c)
        count=c.execute('''SELECT count(DISTINCT w.owner_id) AS n FROM ptb_web.workspaces w
            JOIN ptb_accounts.sessions s ON s.token_hash=w.session_hash
            JOIN ptb_accounts.users u ON u.id=w.owner_id
            WHERE w.state='active' AND w.last_seen>%s AND w.expires_at>%s
            AND s.revoked_at IS NULL AND s.expires_at>clock_timestamp() AND u.active''',(now-90,now)).fetchone()['n']
        return PresenceView(online_accounts=count,measured_at=now)


def latest_request(service,owner):
    with service._read() as c:
        return public(c.execute('SELECT * FROM ptb_web.capacity_requests WHERE owner_id=%s ORDER BY created_at DESC,id DESC LIMIT 1',(owner,)).fetchone())


def request_capacity(service,owner,body):
    reason=body.reason.strip()
    if not reason:raise StorageError('capacity_reason_required',422)
    with service.storage._locked() as c, c.transaction():
        old=c.execute("SELECT * FROM ptb_web.capacity_requests WHERE owner_id=%s AND state='pending'",(owner,)).fetchone()
        if old:return public(old)  # one pending request per account, including concurrent/replayed forms
        quota=c.execute('SELECT quota_bytes FROM ptb_storage.quota_accounts WHERE owner_id=%s',(owner,)).fetchone()
        if body.requested_bytes <= (quota['quota_bytes'] if quota else service.default_quota(c)):
            raise StorageError('capacity_not_increase',422)
        if c.execute("SELECT count(*) AS n FROM ptb_web.capacity_requests WHERE owner_id=%s AND created_at>clock_timestamp()-interval '1 day'",(owner,)).fetchone()['n']>=3:
            raise StorageError('capacity_request_limited',429)
        return public(c.execute('INSERT INTO ptb_web.capacity_requests(id,owner_id,requested_bytes,reason,duration) VALUES(%s,%s,%s,%s,%s) RETURNING *',
            (uuid4(),owner,body.requested_bytes,reason,body.duration)).fetchone())


def decide(service,request_id,*,granted_bytes=None,note='',duration=None):
    """Operator-only entry invoked locally with the private service configuration.

    Account ceilings do not reserve the physical disk. Every actual write still
    checks global headroom. Approval reports and checks headroom under the same lock.
    """
    with service.storage._locked() as c, c.transaction():
        row=c.execute('SELECT * FROM ptb_web.capacity_requests WHERE id=%s FOR UPDATE',(UUID(str(request_id)),)).fetchone()
        if not row or row['state']!='pending':raise StorageError('capacity_request_closed',409)
        expires=None
        chosen=duration or row['duration']
        if chosen not in ('today','7d','30d'):raise StorageError('invalid_capacity_duration',422)
        if granted_bytes is not None:
            if type(granted_bytes) is not int or not 100_000_000<=granted_bytes<=10_000_000_000:
                raise StorageError('invalid_capacity_grant',422)
            c.execute('INSERT INTO ptb_storage.quota_accounts(owner_id,quota_bytes) VALUES(%s,%s) ON CONFLICT DO NOTHING',(row['owner_id'],service.default_quota(c)))
            quota=c.execute('SELECT * FROM ptb_storage.quota_accounts WHERE owner_id=%s FOR UPDATE',(row['owner_id'],)).fetchone()
            if granted_bytes<=quota['quota_bytes']:raise StorageError('capacity_not_increase',422)
            service.storage._disk_budget(c,granted_bytes-quota['quota_bytes'])
            now=c.execute('SELECT clock_timestamp() AS t').fetchone()['t']
            expires=(now.astimezone(timezone(timedelta(hours=8))).replace(hour=0,minute=0,second=0,microsecond=0)+timedelta(days=1)
                     if chosen=='today' else now+timedelta(days=7 if chosen=='7d' else 30))
            c.execute('UPDATE ptb_storage.quota_accounts SET quota_bytes=%s,web_override=true,web_override_expires_at=%s WHERE owner_id=%s',(granted_bytes,expires,row['owner_id']))
        return public(c.execute('''UPDATE ptb_web.capacity_requests SET state=%s,decided_at=clock_timestamp(),granted_bytes=%s,decision_note=%s,
            granted_duration=%s,expires_at=%s WHERE id=%s RETURNING *''',
            ('approved' if granted_bytes is not None else 'declined',granted_bytes,note[:500],chosen if granted_bytes is not None else None,expires,row['id'])).fetchone())


def set_quota(service,amount,owner=None):
    """Operator changes default or one account; reductions never delete data."""
    if type(amount) is not int or not 100_000_000<=amount<=10_000_000_000:
        raise StorageError('invalid_capacity_grant',422)
    with service.storage._locked() as c, c.transaction():
        if owner is None:
            old=service.default_quota(c)
            if amount>old:service.storage._disk_budget(c,amount-old)
            c.execute('UPDATE ptb_web.capacity_policy SET default_quota_bytes=%s WHERE version=1',(amount,))
            c.execute('UPDATE ptb_storage.quota_accounts SET quota_bytes=%s WHERE NOT web_override',(amount,))
        else:
            owner=UUID(str(owner))
            if not c.execute('SELECT 1 FROM ptb_accounts.users WHERE id=%s',(owner,)).fetchone():
                raise StorageError('account_not_found',404)
            old=c.execute('SELECT quota_bytes FROM ptb_storage.quota_accounts WHERE owner_id=%s',(owner,)).fetchone()
            previous=old['quota_bytes'] if old else service.default_quota(c)
            if amount>previous:service.storage._disk_budget(c,amount-previous)
            c.execute('''INSERT INTO ptb_storage.quota_accounts(owner_id,quota_bytes,web_override,web_base_quota_bytes) VALUES(%s,%s,true,%s)
                ON CONFLICT(owner_id) DO UPDATE SET quota_bytes=excluded.quota_bytes,web_base_quota_bytes=excluded.web_base_quota_bytes,
                web_override=true,web_override_expires_at=NULL''',(owner,amount,amount))
        return dict(quota_bytes=amount,scope='account' if owner else 'default',existing_data_deleted=False)
