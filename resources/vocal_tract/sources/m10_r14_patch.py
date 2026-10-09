"""M10-R14 raised-tip thickness and a material blade surface anchor.

SPDX-License-Identifier: GPL-3.0-or-later
Applied after R13 to the pinned VTL 2.4 source.
"""


def apply(source):
    def replace(old, new, count=1):
        nonlocal source
        assert source.count(old) == count, old[:100]
        source = source.replace(old, new)

    replace('void VocalTract::calcTongue()', '''double VocalTract::m10TipRadius()
{
  // Continuous model thickness compensation for a raised/retracted tip.
  // Use the same radius in construction and hard-tissue clearance.
  double raised=std::max(0.0,std::min(1.0,(params[TTY].x-params[TCY].x-1.2)/1.0));
  raised=raised*raised*(3.0-2.0*raised);
  double curled=std::max(0.0,std::min(1.0,(params[TBX].x-params[TTX].x+0.5)/1.25));
  curled=curled*curled*(3.0-2.0*curled);
  double folded=std::max(0.0,std::min(1.0,(params[TTY].x-params[TBY].x)/0.5));
  return anatomy.tongueTipRadius_cm+0.18*raised*curled*folded;
}

void VocalTract::calcTongue()''')
    replace('double r1 = anatomy.tongueTipRadius_cm;', 'double r1 = m10TipRadius();')
    replace('double tongueTipRadius = anatomy.tongueTipRadius_cm;', 'double tongueTipRadius = m10TipRadius();', 3)
    replace('  alpha[2] = getCircleTangent(tipTangent, C1, r1, true);', '''  alpha[2] = getCircleTangent(tipTangent, C1, r1, true);
  // A curled blade joins the posterior arc, not almost at its apex. This
  // preserves a rounded tissue cross-section instead of a thin forward strip.
  const double roundTip=std::max(0.0,std::min(1.0,(r1-anatomy.tongueTipRadius_cm)/0.18));
  alpha[2]=(1.0-roundTip)*alpha[2]+roundTip*M_PI;''')
    replace('1.5*tipResolution*(samplingTurn[k-1]+samplingTurn[k])*0.5;',
            '2.0*tipResolution*(samplingTurn[k-1]+samplingTurn[k])*0.5;')
    replace('  const int samplingCount=targetCurve.getNumPoints();', '''  // Follow the middle of the blade segment, including after hull clipping.
  // Store its fractional native rib, not its off-surface Bezier control point.
  const double bladeStation=targetCurve.getCurveParam(64+16);
  double previousStation=0.0;
  m10BladeRib=0.0;
  const int samplingCount=targetCurve.getNumPoints();''')
    replace('    rib[i].point = targetCurve.getPoint(t);', '''    if(i>0 && previousStation<=bladeStation && t>=bladeStation)
      m10BladeRib=i-1+(bladeStation-previousStation)/std::max(EPSILON,t-previousStation);
    previousStation=t;
    rib[i].point = targetCurve.getPoint(t);''')
    return source
