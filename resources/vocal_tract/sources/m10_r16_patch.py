"""M10-R16 anterior transverse shaping and material floor attachment.

SPDX-License-Identifier: GPL-3.0-or-later
Applied after R15, with exact context checks against the pinned VTL source.
"""


def apply(header, source):
    def replace(old,new,count=1):
        nonlocal source
        assert source.count(old)==count,old[:120]
        source=source.replace(old,new)

    token='  double m10BladeRib = 0.0;'
    assert header.count(token)==1
    header=header.replace(token,token+'''
  double m10SideRibs[3] = {11.0,21.0,30.0};
  double m10FrontWeight[NUM_TONGUE_RIBS] = {0.0};
  static const int M10_FLOOR_RIB = NUM_LARYNX_RIBS + NUM_THROAT_RIBS + 4;
''')
    replace('  if (tongueSideElevation_cm[4] > 0.0) { tongueSideElevation_cm[4] = 0.0; }',
            '  // R16: retain the positive side elevation through the tip.')
    replace('  const double bladeStation=targetCurve.getCurveParam(64+16);','''  const double bladeStation=targetCurve.getCurveParam(64+16);
  double sideStations[3];
  for(int k=0;k<3;k++) sideStations[k]=targetCurve.getCurveParam(tongueSideIndex[k+1]);
  for(int k=0;k<NUM_TONGUE_RIBS;k++) m10FrontWeight[k]=0.0;''')
    replace('    previousStation=t;\n    rib[i].point = targetCurve.getPoint(t);','''    for(int k=0;k<3;k++) if(i>0 && previousStation<=sideStations[k] && t>=sideStations[k])
      m10SideRibs[k]=i-1+(sideStations[k]-previousStation)/std::max(EPSILON,t-previousStation);
    double front=std::max(0.0,std::min(1.0,(t-sideStations[1])/std::max(EPSILON,sideStations[2]-sideStations[1])));
    m10FrontWeight[i]=front*front*(3.0-2.0*front);
    previousStation=t;
    rib[i].point = targetCurve.getPoint(t);''')
    replace('  rib[i].point = finalPoint;', '  rib[i].point = finalPoint;\n  m10FrontWeight[i]=1.0;')
    # Refine a detected width boundary inside its sampling interval. Assigning
    # the first inside sample directly produces a 0.73 mm width jump when a
    # side control crosses a profile sample.
    replace('        rib[i].minX = x;', '''        rib[i].minX = x;
        if(k>0 && upperProfile[i][k-1]!=INVALID && lowerProfile[i][k-1]!=INVALID) {
          const double px=x-PROFILE_SAMPLE_LENGTH,py=a2_left[i]*px*px+a3_left[i]*px*px*px;
          const double before=std::min(px-leftEdge.x,std::min(py-lowerProfile[i][k-1],upperProfile[i][k-1]-py));
          const double now=std::min(x-leftEdge.x,std::min(y-lowerProfile[i][k],upperProfile[i][k]-y));
          if(before<0.0 && now>=0.0) rib[i].minX=px+PROFILE_SAMPLE_LENGTH*(-before)/(now-before);
        }''')
    replace('        rib[i].maxX = x;', '''        rib[i].maxX = x;
        if(k+1<NUM_PROFILE_SAMPLES && upperProfile[i][k+1]!=INVALID && lowerProfile[i][k+1]!=INVALID) {
          const double px=x+PROFILE_SAMPLE_LENGTH,py=a2_right[i]*px*px+a3_right[i]*px*px*px;
          const double before=std::min(rightEdge.x-px,std::min(py-lowerProfile[i][k+1],upperProfile[i][k+1]-py));
          const double now=std::min(rightEdge.x-x,std::min(y-lowerProfile[i][k],upperProfile[i][k]-y));
          if(before<0.0 && now>=0.0) rib[i].maxX=px-PROFILE_SAMPLE_LENGTH*(-before)/(now-before);
        }''')
    # Native contact calculations, rather than a post-contact mesh displacement.
    replace('  static const double SAFETY_MARGIN_CM = 0.02;', '''  static const double SAFETY_MARGIN_CM = 0.02;
  const auto smooth16=[](double x){x=std::max(0.0,std::min(1.0,x));return x*x*(3.0-2.0*x);};
  const double sideControl=params[TS3].x;
  // Positive bracing reaches the oral side walls. Otherwise width filtering
  // narrows the tongue away from its contact shoulders, leaving false side air.
  if(sideControl>0.0) for(i=1;i<=NUM_DYNAMIC_TONGUE_RIBS;i++) {
    double left=rib[i].minX,right=rib[i].maxX;
    for(k=0;k<NUM_PROFILE_SAMPLES;k++) if(upperProfile[i][k]!=INVALID && lowerProfile[i][k]!=INVALID) {
      const double at=k*PROFILE_SAMPLE_LENGTH-0.5*PROFILE_LENGTH;
      if(std::abs(at)<=STANDARD_HALF_WIDTH && upperProfile[i][k]>=lowerProfile[i][k]) {
        left=std::min(left,at);right=std::max(right,at);
      }
    }
    const double w=m10FrontWeight[i]*smooth16(sideControl/0.15);
    rib[i].minX+=w*(left-rib[i].minX);rib[i].maxX+=w*(right-rib[i].maxX);
  }
  // A braced edge can cross the last valid wall sample. Extend the nearest
  // contact value over missing samples instead of turning contact off abruptly
  // when just one of the interpolation neighbours is invalid.
  for(i=1;i<=NUM_DYNAMIC_TONGUE_RIBS;i++) if(m10FrontWeight[i]>0.0) {
    for(double* profile : {upperProfile[i],lowerProfile[i]}) {
      double filled[NUM_PROFILE_SAMPLES];
      for(k=0;k<NUM_PROFILE_SAMPLES;k++) {
        filled[k]=profile[k];
        if(profile[k]==INVALID) for(int d=1;d<NUM_PROFILE_SAMPLES;d++) {
          if(k-d>=0 && profile[k-d]!=INVALID){filled[k]=profile[k-d];break;}
          if(k+d<NUM_PROFILE_SAMPLES && profile[k+d]!=INVALID){filled[k]=profile[k+d];break;}
        }
      }
      for(k=0;k<NUM_PROFILE_SAMPLES;k++) profile[k]=filled[k];
    }
  }''')
    replace('      // Test if the rib is within the hull.','''      // R16: a narrow central trough or a central crest with lowered sides.
      // These are geometric control dimensions, not prescribed phone labels.
      const double front=m10FrontWeight[i];
      if(front>0.0 && sideControl>0.0) {
        const double shoulder=smooth16(std::abs(z)/0.55);
        const double target=(std::min(sideControl/0.3,1.0)+std::max(0.0,sideControl-0.3)*0.4)*shoulder;
        t+=front*smooth16(sideControl/0.15)*(target-t);
      }
      // Test if the rib is within the hull.''')
    # The 96-sample upstream hull mirrors at the half-sample, not z=0.
    # With wider bracing the positive edge could land on an invalid neighbour
    # and bypass contact entirely. Use the same anatomical half-profile on
    # both linked shoulders, while keeping the original signed vertex z.
    replace('      index = (int)((z + 0.5*PROFILE_LENGTH) / PROFILE_SAMPLE_LENGTH);\n      ratio = (z + 0.5*PROFILE_LENGTH - (double)index*PROFILE_SAMPLE_LENGTH) / PROFILE_SAMPLE_LENGTH;', '''      const double contactZ=front>0.0?-std::abs(z):z;
      index = (int)((contactZ + 0.5*PROFILE_LENGTH) / PROFILE_SAMPLE_LENGTH);
      ratio = (contactZ + 0.5*PROFILE_LENGTH - (double)index*PROFILE_SAMPLE_LENGTH) / PROFILE_SAMPLE_LENGTH;''')
    replace('y = upperProfile[i][index] + ratio*(upperProfile[i][index+1]-upperProfile[i][index]) - SAFETY_MARGIN_CM;', '''y = upperProfile[i][index] + ratio*(upperProfile[i][index+1]-upperProfile[i][index]);
        y-=SAFETY_MARGIN_CM*(1.0-front);
        // A braced anterior groove remains open at the centre even when its
        // shoulders contact the palate, mirroring TS3's bracing interpretation.
        if(sideControl>0.2) y-=front*0.25*smooth16((sideControl-0.2)/0.3)*(1.0-smooth16(std::abs(z)/0.35));''')
    return header,source
