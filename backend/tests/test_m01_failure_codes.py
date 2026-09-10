"""M01-G: public task failures preserve actionable reasons, never exception text."""
import json
import struct
import pytest
from ptb_worker.io.limits import FormatError, LimitError
from ptb_worker.acoustic_errors import AcousticFailure, public_error
from ptb_worker.segmentation import unpack_bundle


@pytest.mark.parametrize('code', ['invalid_audio', 'invalid_textgrid', 'invalid_lip',
    'analysis_sample_limit', 'analysis_output_limit', 'analysis_resource_limit',
    'no_parameter_frames', 'missing_or_invalid_tier', 'parent_result_source_mismatch'])
def test_safe_child_reason_survives_bundle(code):
    raw=json.dumps({'error':code}).encode()
    with pytest.raises(AcousticFailure) as result:
        unpack_bundle(struct.pack('<Q',len(raw))+raw,10000)
    assert result.value.code==code
    assert isinstance(result.value,FormatError)


def test_private_exception_is_not_a_public_reason():
    secret='C:/private/corpus/name.TextGrid: secret annotation'
    assert public_error(ValueError(secret))=='execution_failed'
    with pytest.raises(ValueError):AcousticFailure(secret)
    raw=json.dumps({'error':secret}).encode()
    with pytest.raises(FormatError) as result:
        unpack_bundle(struct.pack('<Q',len(raw))+raw,10000)
    assert secret not in str(result.value)


@pytest.mark.parametrize('error,expected',[
    (LimitError('sample_limit_exceeded'),'analysis_sample_limit'),
    (LimitError('export_cell_limit'),'analysis_output_limit'),
    (LimitError('native_timeout'),'deadline_exceeded'),
    (MemoryError(),'analysis_resource_limit'),
    (LimitError('private detail'),'analysis_resource_limit')])
def test_resource_failures_are_classified_without_raw_text(error,expected):
    assert public_error(error)==expected
