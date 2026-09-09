"""P06 transition expectations independent of any persistence implementation."""
import pytest
from ptb_worker.policy import cancel_state, fenced, TERMINAL


def test_cancellation_race_has_no_success_after_request():
    assert cancel_state('queued') == 'cancelled'
    assert cancel_state('running') == 'cancel_requested'
    assert cancel_state('cancel_requested') == 'cancel_requested'
    for state in TERMINAL:
        assert cancel_state(state) == state


def test_lease_identity_generation_and_expiry_are_all_required():
    row = dict(state='running', worker_id='worker-a', generation=3, lease_until=20, deadline=40)
    assert fenced(row,'worker-a',3,19)
    for worker,generation,now in [('worker-b',3,19),('worker-a',2,19),('worker-a',3,20)]:
        assert not fenced(row,worker,generation,now)
    assert not fenced({**row,'state':'interrupted'},'worker-a',3,19)
    assert not fenced({**row,'deadline':19},'worker-a',3,19)
