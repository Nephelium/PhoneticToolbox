import hashlib
import pytest
from phonetic_core.pipeline_check import blocks


def test_probe_known_bytes_and_partial_last_block():
    rows=list(blocks(257,0))
    assert [r['progress'] for r in rows[:-1]] == [128/257,256/257,1]
    assert rows[-1]['result'] == {'sha256':hashlib.sha256(bytes(range(256))+b'\x00').hexdigest(),'sample_count':257}


@pytest.mark.parametrize('count,seed',[(0,0),(4097,0),(1,-1),(1,256),(True,0)])
def test_probe_rejects_invalid_bounds(count,seed):
    with pytest.raises(ValueError):list(blocks(count,seed))
