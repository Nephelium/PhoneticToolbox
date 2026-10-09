"""Read-only F0 display from an immutable M07 generation snapshot.

Uses the inherited synthesis interpolation, including its FFT resampling.
source_ids: SRC-ZAIWA, REF-ZAIWA. No new synthesis or pitch estimator.
"""
from types import SimpleNamespace
import numpy as np
from .m07_legacy import interpolate_target_f0, sample_f0_on_axis, voiced_range
from .m07_models import F0AlignmentMode


def synthesis_f0_display(metadata):
    if metadata.get('action') != 'generate' or metadata.get('algorithm') != 'm07-v2-lpc-residual/1':
        raise ValueError('m07_display_snapshot')
    steps = metadata['generation']['step_count']
    if type(steps) is not int or not 2 <= steps <= 50:
        raise ValueError('m07_display_snapshot')
    pair = []
    for role in ('source', 'target'):
        f0 = np.asarray(metadata[role + '_f0'], dtype=float)
        bounds = metadata[role + '_bounds']
        rate = metadata['sample_rate_hz']
        if (f0.ndim != 1 or not 2 <= len(f0) <= 10002 or not np.isfinite(f0).all()
                or np.any(f0 < 0) or np.any(f0 > 1000) or rate != 11025
                or len(bounds) != 2 or any(type(v) is not int for v in bounds)
                or not 0 <= bounds[0] <= bounds[1] < metadata[role + '_samples'] <= 110250):
            raise ValueError('m07_display_snapshot')
        pair.append(SimpleNamespace(f0_hz=f0.copy(), sample_rate=rate,
                                    start_sample=bounds[0], end_sample=bounds[1]))
    source, target = pair[::-1] if metadata['reverse_direction'] else pair
    kind = metadata['continuum_type']
    if kind not in (1, 2, 3):
        raise ValueError('m07_display_snapshot')
    values = (np.repeat(source.f0_hz[:, None], steps, axis=1) if kind == 1
              else interpolate_target_f0(source, target, steps))
    mode = F0AlignmentMode(metadata['alignment'])
    curves = []
    for i in range(steps):
        track = values[:, i]
        bounds = voiced_range(track)
        if bounds is None:
            axis = np.array([0., 100. if mode == F0AlignmentMode.NORMALIZED else 1.])
        else:
            end = 100. if mode == F0AlignmentMode.NORMALIZED else float(max(1, bounds[1] - bounds[0]))
            axis = np.linspace(0., end, min(600, max(2, bounds[1] - bounds[0] + 1)))
        curves.append(dict(name=f'step{i + 1:02d}', axis=axis.tolist(),
                           values=sample_f0_on_axis(track, axis, mode).tolist()))
    return dict(kind='synthesis-control-f0', alignment=mode.value, curves=curves)
