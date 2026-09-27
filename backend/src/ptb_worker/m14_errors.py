"""Fixed public error codes; no input text or paths are exposed."""
M14_ERRORS=frozenset({
 'm14_empty_input','m14_input_budget','m14_unsupported_format','m14_row_budget',
 'm14_invalid_cell','m14_expanded_budget','m14_formula_input','m14_missing_columns',
 'm14_column_budget','m14_decode_failed','m14_no_valid_rows','m14_output_shape_budget',
 'm14_output_budget','m14_ipa_font','m14_unknown_symbol','m14_cyclic_merge',
 'm14_order_mismatch','m14_tone_mismatch','m14_invalid_tone_name','m14_execution_failed',
 'm14_input_changed','m14_invalid_bundle','m14_incomplete_output','m14_output_hash',
 'm14_runtime_unavailable','m14_cell_output_budget','m14_timeout',
})
