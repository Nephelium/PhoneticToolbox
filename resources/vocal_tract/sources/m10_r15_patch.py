"""Runtime longitudinal resolution with fixed-capacity native arrays.

SPDX-License-Identifier: GPL-3.0-or-later
The 257-point option retains the R14 calculation exactly. Lower resolutions
use the upstream power-of-two centerline algorithm, including its exponent.
"""
import re


def apply(header, source):
    marker = '  CenterLinePoint roughCenterLine[NUM_CENTERLINE_POINTS];'
    assert header.count(marker) == 1
    header = header.replace(marker, '''  int m10SectionCount = NUM_CENTERLINE_POINTS >= 129 ? 129 : NUM_CENTERLINE_POINTS;
  int m10SectionExponent = NUM_CENTERLINE_POINTS >= 129 ? 7 : NUM_CENTERLINE_POINTS_EXPONENT;
''' + marker)
    # Local arrays retain compile-time capacity on MSVC. Loops, indices,
    # spacing and tube integrals use the active instance count.
    arrays = ['Point2D nuLine[NUM_CENTERLINE_POINTS+2]',
              'double nu[NUM_CENTERLINE_POINTS+2]']
    for i, array in enumerate(arrays):
        assert source.count(array) == 1
        source = source.replace(array, f'M10_FIXED_ARRAY_{i}')
    source = re.sub(r'\bNUM_CENTERLINE_POINTS_EXPONENT\b', 'm10SectionExponent', source)
    source = re.sub(r'\bNUM_CENTERLINE_POINTS\b', 'm10SectionCount', source)
    for i, array in enumerate(arrays):
        source = source.replace(f'M10_FIXED_ARRAY_{i}', array)
    return header, source
