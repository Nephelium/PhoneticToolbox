"""M08 boundary rules, separate from the unmodified Praat numerical paths.

Source: V2 PitchManipulationWidget / ImportF0Dialog; SRC-PRAAT.
Invalid nonfinite inputs and runaway combination counts are rejected, never clipped.
"""
import math
import re
import numpy as np


def track(sound):
    pitch = sound.to_pitch()  # V2 defaults, real Praat frame times
    return pitch.xs(), pitch.selected_array['frequency'].copy()


def validate_range(sound, start, end):
    if not all(math.isfinite(x) for x in (start, end)) or not sound.xmin <= start < end <= sound.xmax:
        raise ValueError('m08_invalid_range')


def validate_controls(points, xmin, xmax, *, max_outputs=256):
    if len(points) < 2 or len(points) > 64:
        raise ValueError('m08_control_count')
    times = [p['time'] for p in points]
    if not all(math.isfinite(t) for t in times) or not xmin <= times[0] < times[-1] <= xmax:
        raise ValueError('m08_control_range')
    if any(a >= b for a, b in zip(times, times[1:])):
        raise ValueError('m08_control_order')
    diagonal = None
    count = 1
    for p in points:
        mode, values = p['mode'], p['freqs']
        if mode not in ('full', 'order', 'reverse', 'constant') or not values or not all(math.isfinite(v) for v in values):
            raise ValueError('m08_control_values')
        if mode in ('order', 'reverse'):
            if diagonal is not None and diagonal != len(values):
                raise ValueError('m08_diagonal_length')
            diagonal = len(values)
        if mode == 'full':
            count *= len(values)
    count *= diagonal or 1
    if count > max_outputs:
        raise ValueError('m08_output_budget')
    return count


def import_sequence(times, modified, xmin, xmax, text):
    values = []
    for line in re.sub(r'[,，;；]', ' ', text.strip()).splitlines():
        parts = line.split()
        if parts:
            values.append(float(parts[1] if len(parts) >= 2 else parts[0]))
    if not values or not np.isfinite(values).all():
        raise ValueError('m08_import_values')
    indices = np.flatnonzero((times >= xmin) & (times <= xmax))
    voiced = np.flatnonzero(modified[indices] > 0)
    if not len(voiced):
        raise ValueError('m08_import_unvoiced')
    if np.any(np.diff(voiced) > 1):
        raise ValueError('m08_import_discontinuous')
    targets = indices[voiced[0]:voiced[-1] + 1]
    result = modified.copy()
    result[targets] = (np.full(len(targets), values[0]) if len(values) < 2 else
                       np.interp(np.linspace(0, 1, len(targets)), np.linspace(0, 1, len(values)), values))
    return result


def next_name(stem, xmin, xmax, names):
    prefix = f'{stem}_{xmin:.2f}_{xmax:.2f}_modified'
    numbers = [int(m.group(1)) for n in names if n.startswith(prefix + '_')
               and (m := re.search(r'_(\d+)\.wav$', n))]
    return f'{prefix}_{max(numbers, default=0) + 1}.wav'
