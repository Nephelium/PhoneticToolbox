"""M01-F1 deterministic batch policy, shared by future PG/SQLite coordinators.

No submission, database initialization, file reads, or scientific computation.
"""
from dataclasses import dataclass
import hashlib
import json
from ptb_api.acoustic_models import AcousticBatchSummary,AcousticBatchItem
from ptb_api.acoustic_batch_models import BatchRequest

MAX_BATCH_INPUTS=1000
ACTIVE={'queued','running','cancel_requested'}


@dataclass(frozen=True)
class FrozenBatch:
    serialized: str
    sha256: str


def freeze_request(body):
    body=BatchRequest.model_validate_json(body.model_dump_json())
    raw=json.dumps(body.model_dump(exclude={'idempotency_key'}),sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)
    if len(raw.encode())>4_000_000:raise ValueError('Batch snapshot budget exceeded')
    return FrozenBatch(raw,hashlib.sha256(raw.encode()).hexdigest())


def child_snapshot(frozen,index,*,batch_id,attempt=0):
    if type(index)!=int or type(attempt)!=int or not 0<=attempt<=99:raise ValueError('Invalid batch item attempt')
    value=json.loads(frozen.serialized)
    if not 0<=index<len(value['inputs']):raise ValueError('Invalid batch item index')
    return {'operation':value['operation'],'project_id':value['project_id'],'inputs':value['inputs'][index],
            'config':value['config'],'layer':value['layer'],'batch_sha256':frozen.sha256,
            'idempotency_key':f'm01-{batch_id}-{index}-{attempt}'}


def summarize(batch_id,items,*,cancel_requested=False):
    items=[AcousticBatchItem.model_validate(x) for x in items]
    counts={state:sum(i.state==state for i in items) for state in
            ('not_started','queued','running','cancel_requested','succeeded','failed','cancelled','interrupted')}
    active=any(i.state in ACTIVE for i in items)
    closed=not active and (cancel_requested or counts['not_started']==0)
    return AcousticBatchSummary(batch_id=batch_id,total=len(items),closed=closed,
        complete=closed and counts['succeeded']==len(items),counts=counts,items=items)


def next_item(summary,*,cancel_requested=False):
    if cancel_requested or summary.closed or any(i.state in ACTIVE for i in summary.items):return None
    return next((i.index for i in summary.items if i.state=='not_started'),None)


def check_capacity(existing_jobs,pending_items,incoming):
    if any(type(n)!=int or n<0 for n in (existing_jobs,pending_items,incoming)):raise ValueError('Invalid capacity counts')
    if incoming<1 or incoming>MAX_BATCH_INPUTS or existing_jobs+pending_items+incoming>1000:
        raise ValueError('job_limit_reached')
