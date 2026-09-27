"""Q21-Q24 policy boundaries and historical wire compatibility; no database."""
import pytest
from pydantic import ValidationError
from ptb_api.acoustic_models import AcousticFileManifest, AcousticMetadata, AcousticConfigSnapshot, config_digest
from ptb_api.acoustic_boundary import output_expiry
from ptb_api.quota import reserve, settle, expiry, StorageError
from ptb_api.storage_models import UploadInput
from ptb_api.storage_policy import retention_seconds, INDEPENDENT_RESULT_OPERATIONS
from ptb_api.job_models import JobView


def uid(n):
    return f'00000000-0000-4000-8000-{n:012d}'


def manifest_data(ttl=604800):
    config = AcousticConfigSnapshot()
    metadata = AcousticMetadata(core_version='3.0.0a1', project_id=uid(1),
        inputs=[dict(asset_id=uid(2), sha256='a'*64, role='audio', expires_at=200.)],
        decoded=dict(sample_rate_hz=16000, channels=1, sample_count=16000, sample_dtype='int16'),
        config=config, config_sha256=config_digest(config), source_ids=['SRC-PRAAT'], backends=[])
    return dict(job_id=uid(3), metadata=metadata.model_dump(), completed_at=100., retention='server',
        expires_at=100.+ttl, row_count=1, files=[dict(asset_id=uid(n), format=fmt,
            size_bytes=200, sha256='b'*64, expires_at=100.+ttl) for n, fmt in ((4,'xlsx'),(5,'sqlite'))])


def test_old_manifest_seven_days_and_large_results_remain_readable():
    raw = manifest_data()
    raw['files'][0]['size_bytes'] = 1_000_000_001
    restored = AcousticFileManifest.model_validate(raw)
    assert restored.policy_version == 1 and restored.expires_at == 604900
    assert AcousticFileManifest.model_validate_json(restored.model_dump_json()) == restored


@pytest.mark.parametrize('ttl,valid', [(259199,True),(259200,True),(259201,False),(0,False)])
def test_new_manifest_exact_seconds(ttl, valid):
    raw = manifest_data(ttl) | {'policy_version':2}
    if valid:
        assert AcousticFileManifest.model_validate(raw).expires_at == 100+ttl
    else:
        with pytest.raises(ValidationError): AcousticFileManifest.model_validate(raw)


def test_new_manifest_cannot_use_legacy_size_allowance():
    raw = manifest_data(259200) | {'policy_version':2}
    raw['files'][0]['size_bytes'] = 999_999_800
    AcousticFileManifest.model_validate(raw)
    raw['files'][0]['size_bytes'] += 1
    with pytest.raises(ValidationError): AcousticFileManifest.model_validate(raw)
    with pytest.raises(ValueError): retention_seconds(3)
    with pytest.raises(ValidationError): AcousticFileManifest.model_validate(raw | {'policy_version':3})


def test_old_job_managed_manifest_has_no_ttl_revalidation_or_new_byte_ceiling():
    raw = dict(id=uid(3), project_id=uid(1), operation='archive_zip', state='succeeded', progress=1,
        generation=1, created_at=0., updated_at=100., error_code=None,
        result_manifest=dict(kind='managed_files',complete=True, core_version='3.0.0a1',
            files=[dict(id=uid(4),name='旧结果.zip',kind='archive',size_bytes=5_000_000_000,
                sha256='b'*64,expires_at=604900.)]))
    job = JobView.model_validate(raw)
    assert job.result_manifest.files[0].expires_at == 604900.


def test_over_quota_blocks_even_pre_reserved_append_but_allows_recovery_settlement():
    with pytest.raises(StorageError, match='quota_exceeded'): reserve(1_000_000_001,0,0)
    with pytest.raises(StorageError, match='quota_exceeded'): reserve(2,1_000_000_000,0)
    assert settle(2,1_000_000_000,1) == (3,999_999_999)
    assert reserve(999_999_999,0,1) == 1


def test_new_upload_byte_ceiling_and_no_display_rounding():
    raw = dict(project_id=uid(1),name='合成.wav',idempotency_key='policy-boundary-test')
    assert UploadInput(**raw,expected_bytes=1_000_000_000).expected_bytes == 1_000_000_000
    with pytest.raises(ValidationError): UploadInput(**raw,expected_bytes=1_000_000_001)


def test_analysis_and_derived_expiry_keep_distinct_semantics():
    inputs = AcousticFileManifest.model_validate(manifest_data()).metadata.inputs
    assert output_expiry(mode='server',operation='analysis',completed_at=100.,inputs=inputs) == 259300.
    assert output_expiry(mode='server',operation='segment',completed_at=100.,inputs=inputs) == 200.
    assert output_expiry(mode='local',operation='analysis',completed_at=100.,inputs=inputs) is None
    assert expiry(100, [800000]) == 259300
    assert expiry(100, [101,200]) == 101


def test_operation_policy_distinguishes_recomputation_from_copying():
    assert {'acoustic_analysis','egg_analysis','lpc_analysis','spectrogram_to_audio','pitch_manipulation'} <= INDEPENDENT_RESULT_OPERATIONS
    assert not {'archive_zip','extract_zip','textgrid_segment'} & INDEPENDENT_RESULT_OPERATIONS
