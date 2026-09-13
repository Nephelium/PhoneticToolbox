"""Fixed M01 public reasons. This module must remain free of scientific imports."""
from .io.limits import Cancelled, FormatError, LimitError

ACOUSTIC_ERRORS=frozenset({
    'font_unavailable',
    'lpc_runtime_unavailable','lpc_runtime_mismatch','lpc_input_budget','lpc_sample_rate',
    'lpc_invalid_roi','lpc_roi_budget','lpc_solver_failed','lpc_segment_too_short',
    'lpc_textgrid_range','lpc_label_budget',
    'egg_runtime_unavailable','egg_runtime_mismatch','egg_input_budget','egg_stereo_required',
    'egg_sample_rate','egg_invalid_roi','egg_inverse_budget','egg_incomplete_export',
    'egg_filter_failed','egg_inverse_unavailable','egg_pitch_failed',
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
