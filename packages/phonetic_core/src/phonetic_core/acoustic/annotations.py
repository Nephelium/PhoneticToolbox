"""M01-B: preserve original service ceil boundaries and last same-name tier wins.

Source mapping is recorded in docs/modules/evidence/M01-core-migration.json.
File parsing and stricter public contract validation belong to M01-C/D.
"""
import numpy as np


def align_annotations(tiers, target_len, frameshift_ms):
    result = {}
    for tier in tiers:
        labels = np.full(target_len, '', dtype=object)
        for interval in tier.intervals:
            start_idx = int(np.ceil(interval.xmin * 1000.0 / frameshift_ms))
            end_idx = int(np.ceil(interval.xmax * 1000.0 / frameshift_ms))
            start_idx = max(0, start_idx)
            end_idx = min(target_len, end_idx)
            if end_idx > start_idx:
                labels[start_idx:end_idx] = interval.text
        result[f'text_{tier.name}'] = labels
    return result
