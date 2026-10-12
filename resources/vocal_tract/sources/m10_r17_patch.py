"""M10-R17 bounded blade depression and shared geometric cross profiles.

SPDX-License-Identifier: GPL-3.0-or-later
"""


def apply(header, source):
    def replace(old, new):
        nonlocal source
        assert source.count(old) == 1, old[:120]
        source = source.replace(old, new)

    header = header.replace('M10_FLOOR_RIB = NUM_LARYNX_RIBS + NUM_THROAT_RIBS + 4;',
                            'M10_FLOOR_RIB = NUM_LARYNX_RIBS + NUM_THROAT_RIBS + 2;')
    token = '  double m10BladeRib = 0.0;'
    assert header.count(token) == 1
    header = header.replace(token, token + '''
  double m10Upper[NUM_CENTERLINE_POINTS][NUM_PROFILE_SAMPLES];
  double m10Lower[NUM_CENTERLINE_POINTS][NUM_PROFILE_SAMPLES];
''')
    replace('    params[TBY].x=std::max(params[TBY].x,floorPoint.y+0.12);', '''    params[TBY].x=std::max(params[TBY].x,floorPoint.y+0.12);
  // Limit inward deformation relative to the moving dorsal/tip support.
  // 3 mm is a model regularizer, not a measured anatomical universal.
  const double bladeFraction=std::max(0.0,std::min(1.0,
    (params[TBX].x-params[TCX].x)/std::max(0.35,params[TTX].x-params[TCX].x)));
  const double support=(1.0-bladeFraction)*(params[TCY].x+ry)+
    bladeFraction*(params[TTY].x+tongueTipRadius);
  params[TBY].x=std::max(params[TBY].x,support-0.3);''')
    replace('  const double flexibleWeight=std::min(1.0,std::max(0.0,bladeDepression/0.2));', '''  const double flexibleBlend=std::min(1.0,std::max(0.0,bladeDepression/0.2));
  const double flexibleWeight=flexibleBlend*flexibleBlend*(3.0-2.0*flexibleBlend);''')
    replace('''    getCrossProfiles(centerLine[i].point, centerLine[i].normal, upperProfile, lowerProfile, true, articulator);
    getCrossSection(upperProfile, lowerProfile, &crossSection[i]);''', '''    // Preserve genuine narrow gaps in both display and acoustic integration.
    // Empirical area corrections below remain explicit and unchanged.
    getCrossProfiles(centerLine[i].point, centerLine[i].normal, upperProfile, lowerProfile, true, articulator, false, true);
    for(int j=0;j<NUM_PROFILE_SAMPLES;j++) {
      m10Upper[i][j]=upperProfile[j];m10Lower[i][j]=lowerProfile[j];
    }
    getCrossSection(upperProfile, lowerProfile, &crossSection[i]);''')
    return header, source
