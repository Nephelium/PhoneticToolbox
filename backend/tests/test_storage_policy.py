"""P07 independent boundary examples; no SQL, disk deletion or schema creation."""
import pytest
from pydantic import ValidationError
from ptb_api.quota import QUOTA_BYTES, StorageError, reserve, settle, content_range, expiry
from ptb_api.storage_models import UploadInput


def test_q01_last_bytes_and_settlement_are_not_double_counted():
    assert QUOTA_BYTES == 1_000_000_000
    assert reserve(QUOTA_BYTES - 100, 60, 40) == 100
    with pytest.raises(StorageError, match='quota_exceeded'):
        reserve(QUOTA_BYTES - 100, 60, 41)
    assert settle(120, 80, 30) == (150, 50)
    with pytest.raises(StorageError):
        settle(120, 20, 21)


def test_q02_invalid_budgets_do_not_expand_quota():
    for amount in (-1, 1_000_000_001, True, 1.5):
        with pytest.raises(StorageError): reserve(0, 0, amount)


def test_q06_range_boundaries_and_empty_resources():
    assert content_range(None, 10) == (0, 10, False)
    assert content_range('bytes=2-4', 10) == (2, 5, True)
    assert content_range('bytes=-3', 10) == (7, 10, True)
    assert content_range('bytes=7-', 10) == (7, 10, True)
    assert content_range(None, 0) == (0, 0, False)
    for header in ('bytes=10-', 'bytes=4-2', 'bytes=0-1,3-4', 'bytes=-0', 'bytes=x-y'):
        with pytest.raises(StorageError): content_range(header, 10)


def test_q19_q20_ttl_is_fixed_and_archive_cannot_extend_inputs():
    assert expiry(100) == 259300
    assert expiry(100, [300, 250]) == 250
    with pytest.raises(StorageError): expiry(100, [100])


def test_filename_is_display_only_and_owner_cannot_be_selected():
    data=dict(project_id='00000000-0000-4000-8000-000000000001', name='元音 ɑ.wav', idempotency_key='a'*32)
    assert UploadInput(**data).name == '元音 ɑ.wav'
    for name in ('../x', 'a/b', 'a\\b', 'C:x', 'x\r\nSet-Cookie:bad', '', '  '):
        with pytest.raises(ValidationError): UploadInput(**(data | {'name':name}))
    with pytest.raises(ValidationError): UploadInput(**data, owner_id='another-user')
