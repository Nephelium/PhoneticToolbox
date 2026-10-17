"""Synthetic manager UI fixture. Real transactions are tested in test_web_admin."""
import time
from copy import deepcopy

class OperationsFixture:
    def __init__(self):
        self.default=1_000_000_000
        self.rows=[dict(id='00000000-0000-0000-0000-000000000001',owner_id='00000000-0000-0000-0000-000000000002',
            username='synthetic_user',reason='需要处理合成音频，请申请30天临时扩容。',requested_bytes=2_000_000_000,
            quota_bytes=1_000_000_000,used_bytes=200_000_000,reserved_bytes=100_000_000,duration='30d',state='pending',
            created_at='2026-10-16T10:00:00+00:00',granted_bytes=None,granted_duration=None,decision_note=None,expires_at=None)]
    def snapshot(self):
        return dict(measured_at=time.time(),online_accounts=3,pending_requests=sum(r['state']=='pending' for r in self.rows),
            default_quota_bytes=self.default,disk_total_bytes=50_000_000_000,disk_free_bytes=30_000_000_000,
            disk_safety_bytes=10_000_000_000,used_bytes=200_000_000,reserved_bytes=100_000_000,cleanup_pending=0,cleanup_failed=0,
            load=dict(running_tasks=1,waiting_tasks=2,task_slots=1,cpu_percent=42.,memory_percent=36.))
    def requests(self,state='pending',before=None):return dict(requests=deepcopy([r for r in self.rows if r['state']==state]),next_cursor=None)
    def decision(self,body):
        row=next(r for r in self.rows if r['id']==str(body.request_id))
        row.update(state='approved' if body.granted_bytes else 'declined',granted_bytes=body.granted_bytes,
            granted_duration=body.duration,decision_note=body.note,expires_at='2026-10-23T10:00:00+00:00')
        return deepcopy(row)
    def quota(self,body):self.default=body.quota_bytes;return dict(quota_bytes=self.default)
