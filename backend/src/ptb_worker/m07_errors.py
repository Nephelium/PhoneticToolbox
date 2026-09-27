ERRORS=set('m07_invalid_audio m07_input_budget m07_input_changed m07_snapshot_budget m07_invalid_snapshot m07_analysis_stale m07_no_voiced_region m07_insufficient_pulses m07_f0_analysis_failed m07_reaper_unavailable m07_backend_unavailable m07_nonfinite_result m07_nonfinite_parameter m07_invalid_f0 m07_invalid_controls m07_invalid_time_order m07_generation_budget m07_output_budget m07_invalid_bundle m07_output_hash m07_incomplete_output m07_timeout m07_execution_failed m07_invalid_analysis_parameters'.split())
def public_error(exc):
    code=getattr(exc,'code',str(exc))
    if code=='native_timeout':return 'm07_timeout'
    return code if code in ERRORS|{'cancelled','quota_exceeded','output_budget_exceeded','input_unavailable','stale_worker'} else 'm07_execution_failed'
