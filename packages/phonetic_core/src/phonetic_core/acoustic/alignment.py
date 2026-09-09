# M01-B source_ids: SRC-VOICESAUCE; migration evidence: docs/modules/evidence/M01-core-migration.json
import numpy as np
import pandas as pd

def align_track_to_grid(
    source_times: np.ndarray,
    source_values: np.ndarray,
    target_times: np.ndarray,
) -> np.ndarray:
    """Interpolate a track onto a target grid without extrapolation.

    NaN values are intentionally retained in the interpolation input so
    unvoiced gaps are not bridged by neighbouring voiced samples.
    """
    times = np.asarray(source_times, dtype=float).reshape(-1)
    values = np.asarray(source_values, dtype=float).reshape(-1)
    targets = np.asarray(target_times, dtype=float)
    if times.size != values.size:
        raise ValueError("轨迹时间轴与数值长度不一致")

    output = np.full(targets.shape, np.nan, dtype=float)
    valid_times = np.isfinite(times)
    times = times[valid_times]
    values = values[valid_times]
    if times.size == 0:
        return output

    order = np.argsort(times, kind="stable")
    times = times[order]
    values = values[order]
    times, unique_indices = np.unique(times, return_index=True)
    values = values[unique_indices]

    if times.size == 1:
        exact = np.isclose(targets, times[0], rtol=0.0, atol=1e-12)
        output[exact] = values[0]
        return output

    return np.interp(
        targets,
        times,
        values,
        left=np.nan,
        right=np.nan,
    )

def smooth_preserving_gaps(
    values: np.ndarray,
    window_size: int,
) -> np.ndarray:
    """Apply a centred rolling mean within each finite segment."""
    data = np.asarray(values, dtype=float)
    if data.size == 0 or window_size <= 1:
        return data.copy()

    output = np.full(data.shape, np.nan, dtype=float)
    valid_indices = np.flatnonzero(np.isfinite(data))
    if valid_indices.size == 0:
        return output

    cuts = np.flatnonzero(np.diff(valid_indices) > 1)
    starts = np.concatenate(([0], cuts + 1))
    ends = np.concatenate((cuts, [valid_indices.size - 1]))
    for start_pos, end_pos in zip(starts, ends):
        start = int(valid_indices[start_pos])
        end = int(valid_indices[end_pos]) + 1
        output[start:end] = (
            pd.Series(data[start:end])
            .rolling(
                window=int(window_size),
                center=True,
                min_periods=1,
            )
            .mean()
            .to_numpy()
        )
    return output
