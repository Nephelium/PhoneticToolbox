"""M10-R18 local dorsal shape before native hard-tissue contact.

SPDX-License-Identifier: GPL-3.0-or-later
"""


def apply(header,source):
    token='  double m10BladeRib = 0.0;'
    assert header.count(token)==1
    header=header.replace(token,token+'''
  double m10SurfaceX = 0.0, m10SurfaceY = 0.0, m10SurfaceRib = 0.0;
''')
    token='  tongueSideIndex[4] = targetCurve.getNumPoints() - 1;'
    assert source.count(token)==1
    source=source.replace(token,token+'''
  // A material point midway along the root-to-blade surface. Arc length
  // keeps this local control usable even when the upstream body arc collapses.
  const int shapeBegin=24, shapeEnd=80;
  const double shapeStart=targetCurve.getCurveParam(shapeBegin);
  const double shapeStop=targetCurve.getCurveParam(shapeEnd);
  const double shapeMid=(shapeStart+shapeStop)*0.5;
  int surfaceIndex=shapeBegin;
  for(int k=shapeBegin;k<=shapeEnd;k++)
    if(std::abs(targetCurve.getCurveParam(k)-shapeMid)<
       std::abs(targetCurve.getCurveParam(surfaceIndex)-shapeMid))surfaceIndex=k;
  if(std::abs(m10SurfaceX)+std::abs(m10SurfaceY)>0.0) {
    std::vector<Point2D> shaped;
    for(int k=0;k<targetCurve.getNumPoints();k++) {
      Point2D p=targetCurve.getControlPoint(k);
      const double station=targetCurve.getCurveParam(k);
      if(k>shapeBegin && k<shapeEnd) {
        const double t=std::max(0.0,std::min(1.0,(station-shapeStart)/std::max(EPSILON,shapeStop-shapeStart)));
        // Compact, smooth support has zero displacement and slope at both ends.
        const double w=16.0*t*t*(1.0-t)*(1.0-t);
        p.x+=w*m10SurfaceX;p.y+=w*m10SurfaceY;
      }
      shaped.push_back(p);
    }
    targetCurve.reset(0);
    for(const auto& p:shaped)targetCurve.addPoint(p);
  }
  const double surfaceStation=targetCurve.getCurveParam(surfaceIndex);
  m10SurfaceRib=0.0;
''')
    token='    previousStation=t;\n    rib[i].point = targetCurve.getPoint(t);'
    assert source.count(token)==1
    source=source.replace(token,'''    if(i>0 && previousStation<=surfaceStation && t>=surfaceStation)
      m10SurfaceRib=i-1+(surfaceStation-previousStation)/std::max(EPSILON,t-previousStation);
'''+token)
    return header,source
