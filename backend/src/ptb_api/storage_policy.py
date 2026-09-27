"""Versioned server storage policy; legacy limits are READ compatibility only.

The database switch is separately reviewed. Local research files have no server
quota/TTL. Resource limits (memory, per-task output, block sizes) live elsewhere.
"""
from typing import Literal

PolicyVersion = Literal[1, 2]
LEGACY_POLICY_VERSION = 1
POLICY_VERSION = 2
LEGACY_QUOTA_BYTES = 5_000_000_000
LEGACY_RETENTION_SECONDS = 604_800
QUOTA_BYTES = 1_000_000_000
RETENTION_SECONDS = 259_200
TEMP_SECONDS = 86_400
# Pure copies (including segmentation without recomputation) inherit deadlines.
# M14 preview/export both re-run phonological analysis from the input table.
INDEPENDENT_RESULT_OPERATIONS = frozenset({
    'storage_check', 'acoustic_analysis', 'egg_analysis', 'lpc_analysis',
    'mfa_alignment', 'spectrogram_to_audio', 'pitch_manipulation', 'speech_synthesis', 'phonation_synthesis', 'phonology_induction',
})


def independent_result_expiry(snapshot: dict) -> bool:
    """Classify trusted task semantics, including M08's byte-copy save path."""
    if snapshot['operation'] == 'pitch_manipulation' and (
        snapshot.get('saved_copy') or 'source_ref' in snapshot or 'copy_result' in snapshot
    ):
        return False
    return snapshot['operation'] in INDEPENDENT_RESULT_OPERATIONS


def retention_seconds(version: PolicyVersion) -> int:
    if version == LEGACY_POLICY_VERSION:
        return LEGACY_RETENTION_SECONDS
    if version == POLICY_VERSION:
        return RETENTION_SECONDS
    raise ValueError('Unknown storage policy version')
