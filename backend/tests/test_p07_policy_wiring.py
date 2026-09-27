"""Publication policy regression; synthetic metadata, no scientific claims."""
from uuid import uuid4

import pytest
from pydantic import ValidationError
from ptb_api.job_models import FileConfig, ResultFile, FileManifest
from ptb_api.acoustic_batch_models import AcousticTaskManifest
from ptb_api.egg_models import EggManifest, EggTaskConfig, expected_names
from ptb_api.lpc_models import LpcManifest, LPC_NAMES
from ptb_api.m08_models import M08Manifest
from ptb_api.m14_models import M14Manifest
from ptb_api.spec2wav_models import Spec2WavManifest
from ptb_worker.acoustic_files import AcousticFiles
from phonetic_core.transcription.phonology.models import NAMES as M14_NAMES


CASES = [
    ('storage_check', FileManifest, ['result.bin']),
    ('archive_zip', FileManifest, ['result.zip']),
    ('extract_zip', FileManifest, ['result.bin']),
    ('acoustic_analysis', AcousticTaskManifest, ['result.ptb.json', 'result.ptb.sqlite', 'result.xlsx']),
    ('textgrid_segment', AcousticTaskManifest, ['segment.wav']),
    ('egg_analysis', EggManifest, expected_names(EggTaskConfig())),
    ('lpc_analysis', LpcManifest, LPC_NAMES),
    ('spectrogram_to_audio', Spec2WavManifest, ['calibrated.png', 'reconstructed.png', 'reconstructed.wav', 'reconstruction.ptb.json']),
    ('pitch_manipulation', M08Manifest, ['result.wav', 'm08.ptb.json']),
    ('phonology_induction', M14Manifest, ['m14-preview.json']),
    ('phonology_induction', M14Manifest, M14_NAMES),
]


def test_request_ceiling_does_not_restrict_historical_result_reads():
    assert FileConfig(max_output_bytes=1_000_000_000).max_output_bytes == 1_000_000_000
    with pytest.raises(ValidationError):
        FileConfig(max_output_bytes=1_000_000_001)
    assert ResultFile(id=str(uuid4()), name='old', kind='result', size_bytes=5_000_000_000,
                      sha256='a'*64, expires_at=604800.).size_bytes == 5_000_000_000


@pytest.mark.parametrize('operation,model,names', CASES)
def test_published_manifest_explicit_version_and_legacy_missing_field(operation, model, names):
    files = [dict(id=str(uuid4()), name=name, kind='result', size_bytes=1,
                  sha256='a'*64, expires_at=259300.) for name in names]
    publisher = object.__new__(AcousticFiles)
    manifest = publisher.manifest(operation, files)
    assert manifest['policy_version'] == 2
    assert model.model_validate(manifest).policy_version == 2
    del manifest['policy_version']
    assert model.model_validate(manifest).policy_version == 1


@pytest.mark.parametrize('operation,independent', [
    ('storage_check', True), ('acoustic_analysis', True), ('egg_analysis', True),
    ('lpc_analysis', True), ('spectrogram_to_audio', True), ('pitch_manipulation', True),
    ('phonology_induction', True), ('archive_zip', False), ('extract_zip', False),
    ('textgrid_segment', False),
])
def test_publication_uses_snapshot_classification(operation, independent):
    publisher = object.__new__(AcousticFiles)
    assert publisher.independent_expiry({'operation': operation}) is independent


@pytest.mark.parametrize('copy_fields', [
    {'saved_copy': True}, {'source_ref': {'asset_id': str(uuid4()), 'sha256': 'a'*64}},
    {'copy_result': {}}, {'saved_copy': True, 'source_ref': {}, 'copy_result': {}},
])
def test_m08_copy_markers_never_grant_a_new_deadline(copy_fields):
    publisher = object.__new__(AcousticFiles)
    assert not publisher.independent_expiry({'operation': 'pitch_manipulation', **copy_fields})
