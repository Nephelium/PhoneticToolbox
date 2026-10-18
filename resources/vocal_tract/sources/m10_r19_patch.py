"""M10-R19: tangent-continuous blade and wider local dorsal depression.

SPDX-License-Identifier: GPL-3.0-or-later
The remaining 8 mm support bound is a numerical regularizer, not anatomy.
Native floor, teeth and palate contact are still applied after deformation.
"""


def apply(header, source):
    old = 'params[TBY].x=std::max(params[TBY].x,support-0.3);'
    assert source.count(old) == 1
    source = source.replace(old, 'params[TBY].x=std::max(params[TBY].x,support-0.8);')
    old = '''  const double flexibleBlend=std::min(1.0,std::max(0.0,bladeDepression/0.2));
  const double flexibleWeight=flexibleBlend*flexibleBlend*(3.0-2.0*flexibleBlend);'''
    assert source.count(old) == 1
    # The old quadratic moves its middle control off the two tangent lines.
    # Mixing it with the smooth curve leaves a kink for small depressions.
    source = source.replace(old, '''  const double flexibleWeight=1.0;
  // Use the explicit body and tip tangents throughout the complete range.''')
    old = '2.0*tipResolution*(samplingTurn[k-1]+samplingTurn[k])*0.5;'
    assert source.count(old) == 1
    # Curvature-dependent redistribution moved distant ribs when the blade
    # folded. Keep extra anterior resolution tied to material stations.
    source = source.replace(old, 'tipResolution*((k>=64 ? 2.0/63.0 : 0.0)+(k>=96 ? 6.0/31.0 : 0.0)+0.6*exp(-pow((k-94.0)/8.0,2.0)));')
    return header, source
