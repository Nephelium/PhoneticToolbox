import numpy as np
import pytest
from phonetic_core.acoustic.alignment import align_track_to_grid, smooth_preserving_gaps


def test_alignment_preserves_gaps_and_first_duplicate_without_mutation():
    times = np.array([.3, .1, .2, .1, np.nan])
    values = np.array([3., 1., np.nan, 99., 0.])
    original = values.copy()
    out = align_track_to_grid(times, values, [0., .1, .2, .3, .4])
    np.testing.assert_array_equal(out, [np.nan, 1., np.nan, 3., np.nan])
    np.testing.assert_array_equal(values, original)
    with pytest.raises(ValueError):
        align_track_to_grid([0], [], [0])


def test_smoothing_never_bridges_gaps_or_changes_input():
    values = np.array([1., 3., np.nan, 10., 20., 30., np.nan])
    out = smooth_preserving_gaps(values, 3)
    np.testing.assert_array_equal(out, [2., 2., np.nan, 15., 20., 25., np.nan])
    np.testing.assert_array_equal(values, [1., 3., np.nan, 10., 20., 30., np.nan])
