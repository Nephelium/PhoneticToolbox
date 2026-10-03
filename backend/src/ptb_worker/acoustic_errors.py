"""Fixed M01 public reasons. This module must remain free of scientific imports."""
from .io.limits import Cancelled, FormatError, LimitError
from phonetic_core.ports.errors import ACOUSTIC_STAGES

from .m14_errors import M14_ERRORS

from .mfa.errors import M11_ERRORS

ACOUSTIC_ERRORS=M11_ERRORS | M14_ERRORS | frozenset(f'analysis_{stage}_failed' for stage in ACOUSTIC_STAGES) | frozenset({
    'resource_queue_full','resource_queue_timeout','resource_cleanup_required',
    'resource_admission_corrupt','resource_admission_unsafe','resource_profile_invalid',
    'trusted_worker_unavailable','server_resource_profile_required',
    'scientific_platform_unavailable',
    'm08_audio_decode_failed','m08_input_changed','m08_input_budget','m08_output_budget',
    'm08_praat_error','m08_curve_length','m08_invalid_range','m08_execution_failed',
    'm08_transform_requires_whole','m08_nonfinite_audio','m08_controls_required',
    'reaper_runtime_failed',
    'font_unavailable',
    'lpc_runtime_unavailable','lpc_runtime_mismatch','lpc_input_budget','lpc_sample_rate',
    'lpc_invalid_roi','lpc_roi_budget','lpc_solver_failed','lpc_segment_too_short',
    'lpc_textgrid_range','lpc_label_budget',
    'egg_runtime_unavailable','egg_runtime_mismatch','egg_input_budget','egg_stereo_required',
    'egg_sample_rate','egg_invalid_roi','egg_inverse_budget','egg_incomplete_export',
    'egg_filter_failed','egg_inverse_unavailable','egg_pitch_failed','egg_reaper_unavailable','egg_reaper_failed',
    'invalid_spectrogram_image','invalid_spectrogram_config','invalid_spectrogram_result','spectrogram_budget','invalid_image_corners',
    'invalid_audio','invalid_textgrid','invalid_lip','analysis_sample_limit',
    'analysis_output_limit','analysis_resource_limit','no_parameter_frames',
    'deadline_exceeded','invalid_segment_input','segment_budget_exceeded',
    'parent_result_source_mismatch','no_labelled_segments','missing_or_invalid_tier',
    'legacy_parameter_invalid','legacy_parameter_budget','legacy_parameter_time_mismatch',
})


class AcousticFailure(FormatError):
    def __init__(self,code):
        if code not in ACOUSTIC_ERRORS:raise ValueError('Unsafe acoustic error code')
        self.code=code
        super().__init__(code)


def public_error(error):
    if isinstance(error,AcousticFailure):return error.code
    if isinstance(error,Cancelled):return 'cancelled'
    if str(error) in ACOUSTIC_ERRORS:return str(error)
    if isinstance(error,MemoryError):return 'analysis_resource_limit'
    if isinstance(error,LimitError):
        code=str(error)
        if code in ('sample_limit_exceeded','converted_sample_limit_exceeded'):return 'analysis_sample_limit'
        if code in ('native_timeout','deadline_exceeded'):return 'deadline_exceeded'
        if code in ('scientific_result_budget','scientific_output_budget','export_cell_limit',
                    'export_text_limit','export_column_limit','sqlite_pair_budget_exceeded',
                    'output_bytes_exceeded','output_budget_exceeded'):return 'analysis_output_limit'
        return 'analysis_resource_limit'
    code=getattr(error,'code',None)
    return code if code in {'cancelled','input_unavailable','quota_exceeded','disk_space_low',
        'storage_write_failed','output_count_exceeded','stale_worker'} else 'execution_failed'

# M06 fixed child/domain errors; paths and exception text are not exposed.
ACOUSTIC_ERRORS |= {'m06_invalid_f0_method','m06_reaper_unavailable','m06_reaper_failed'}
ACOUSTIC_ERRORS |= {'m06_invalid_render','m06_method_action_mismatch','m06_world_unavailable',
    'm06_world_version','m06_world_input_range','m06_psola_f0_range','m06_resynthesis_pitch_range',
    'm06_resynthesis_duration','m06_world_matrix_budget','m06_invalid_output_length','m06_resynthesis_source_mismatch','m06_psola_no_voiced'}
ACOUSTIC_ERRORS |= {"m06_invalid_config","m06_invalid_sequence","m06_invalid_ipa","m06_empty_sequence", "m06_admission_budget","m06_input_budget","m06_input_changed","m06_audio_decode_failed","m06_execution_failed","m06_invalid_output","m06_invalid_bundle","m06_output_hash","m06_incomplete_output","m06_output_budget","m06_timeout"}

from .m07_errors import ERRORS as M07_ERRORS
ACOUSTIC_ERRORS |= M07_ERRORS
