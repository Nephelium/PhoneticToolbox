"""P16: transport budgets must agree with the file role, including Base64 overhead."""
import base64
import json
import pytest
from ptb_desktop.task_requests import decode_task_request


def test_mfa_file_larger_than_old_shared_limit_is_accepted():
    raw=json.dumps(dict(op='m11_import',role='audio',name='test.wav',base64=base64.b64encode(b'0'*6_100_000).decode()))
    assert decode_task_request(raw)['name']=='test.wav'


@pytest.mark.parametrize('body',[
    dict(op='m11_import',role='unexpected',base64='AA=='),
    dict(op='m11_import',role='transcript',base64='A'*2_666_672),
    dict(op='other',padding='x'*1_000_001),
])
def test_large_or_unknown_operation_cannot_borrow_mfa_budget(body):
    with pytest.raises(ValueError):decode_task_request(json.dumps(body))


def test_global_ceiling_is_checked_before_decoding():
    with pytest.raises(ValueError):decode_task_request('x'*86_000_001)
