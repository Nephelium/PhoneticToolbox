import json
import pytest
from pydantic import ValidationError
from ptb_worker.batch_policy import BatchRequest,freeze_request,child_snapshot,summarize,next_item,check_capacity


def uid(n):return f'00000000-0000-4000-8000-{n:012d}'


def request(n=17):
    return BatchRequest(operation='acoustic_analysis',project_id=uid(1),idempotency_key='batch-test-01',
        inputs=[{'audio':{'asset_id':uid(i+10),'sha256':'a'*64}} for i in range(n)],config={})


def items(states):
    return [dict(index=i,audio_asset_id=uid(i+10),job_id=None if s=='not_started' else uid(i+100),state=s) for i,s in enumerate(states)]


def test_17_files_snapshot_is_immutable_and_child_keys_are_repeatable():
    body=request();frozen=freeze_request(body)
    before=child_snapshot(frozen,16,batch_id=uid(3));body.inputs.reverse();body.config.settings.frameshift_ms=10
    assert child_snapshot(frozen,16,batch_id=uid(3))==before
    assert before['inputs']['audio']['asset_id']==uid(26)
    assert before['config']['settings']['frameshift_ms']==5
    assert child_snapshot(frozen,16,batch_id=uid(3),attempt=1)['idempotency_key']!=before['idempotency_key']
    equivalent=request();equivalent.idempotency_key='batch-test-02';assert freeze_request(equivalent)==frozen


def test_failure_continues_but_cancel_never_schedules_unstarted_items():
    summary=summarize(uid(3),items(['succeeded','failed',*(['not_started']*15)]))
    assert not summary.closed and not summary.complete and next_item(summary)==2
    closed=summarize(uid(3),summary.items,cancel_requested=True)
    assert closed.closed and not closed.complete and closed.counts.not_started==15
    assert next_item(closed,cancel_requested=True) is None
    active=summarize(uid(3),items(['succeeded','cancel_requested',*(['not_started']*15)]),cancel_requested=True)
    assert not active.closed and next_item(active,cancel_requested=True) is None
    assert summarize(uid(3),items(['succeeded']*17)).complete


def test_no_next_item_while_current_child_queued_or_running():
    for state in ('queued','running','cancel_requested'):
        assert next_item(summarize(uid(3),items([state,'not_started']))) is None


def test_admission_counts_reserved_items_and_rejects_duplicates_or_large_list():
    check_capacity(900,83,17)
    with pytest.raises(ValueError):check_capacity(900,84,17)
    with pytest.raises(ValidationError):request(1001)
    value=request().model_dump();value['inputs'][1]=value['inputs'][0]
    with pytest.raises(ValidationError):BatchRequest.model_validate(value)


def test_cut_requires_paired_grid_and_layer_and_rejects_arbitrary_paths():
    value=request(1).model_dump();value.update(operation='textgrid_segment',layer='音节',config=None)
    with pytest.raises(ValidationError):BatchRequest.model_validate(value)
    value['inputs'][0]['textgrid']={'asset_id':uid(9),'sha256':'b'*64}
    assert BatchRequest.model_validate(value).layer=='音节'
    value['path']='C:/Users'
    with pytest.raises(ValidationError):BatchRequest.model_validate(value)
