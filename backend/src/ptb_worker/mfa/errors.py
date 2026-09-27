"""Fixed public MFA reasons. No raw paths, transcript contents, or native logs."""
M11_ERRORS=frozenset({
    'm11_bootstrap_missing','m11_runtime_missing','m11_version_mismatch','m11_dependency_missing','m11_model_missing',
    'm11_dictionary_mismatch','m11_model_mismatch','m11_alignment_failed','m11_no_alignments',
    'm11_corpus_invalid','m11_empty_alignment','m11_invalid_textgrid','m11_incomplete_output',
    'm11_transcript_tiers','m11_audio_budget','m11_oov_words','m11_input_changed','m11_model_changed','m11_runtime_changed',
    'm11_component_changed','m11_self_test_required','m11_temp_budget','m11_temp_space',
    'm11_timeout','m11_process_crashed','m11_response_missing','m11_output_budget',
    'm11_attempt_link','m11_execution_failed','m11_waiting_verified_node','m11_feature_failed',
})
