"""M10-R21: arc-length regularization before native tongue contact.

SPDX-License-Identifier: GPL-3.0-or-later
The 4 mm support is a geometric regularizer, not a tissue mechanics model.
"""


def apply(header, source):
    token = '  const double surfaceStation=targetCurve.getCurveParam(surfaceIndex);'
    assert source.count(token) == 1
    source = source.replace(token, '''  // Round local construction shoulders in physical arc length. Uniform
  // vertex filtering overweights the almost collapsed body arc and can move
  // the blade by millimetres. Keep material indices and both endpoints.
  // Native floor/palate/teeth constraints and area integration follow this.
  double contourLength=0.0;
  for(int k=1;k<targetCurve.getNumPoints();k++)
    contourLength+=(targetCurve.getControlPoint(k)-targetCurve.getControlPoint(k-1)).magnitude();
  if(contourLength>EPSILON) {
    std::vector<Point2D> rounded;
    const double radius=0.4;
    for(int k=0;k<targetCurve.getNumPoints();k++) {
      const double station=targetCurve.getCurveParam(k);
      const double reach=std::min(radius,std::min(station,1.0-station)*contourLength);
      Point2D mean(0.0,0.0);double total=0.0;
      for(int j=-4;j<=4;j++) {
        const double weight=exp(-0.5*j*j/4.0);
        mean=mean+targetCurve.getPoint(station+j*reach/(4.0*contourLength))*weight;
        total+=weight;
      }
      rounded.push_back(mean*(1.0/total));
    }
    targetCurve.reset(0);
    for(const auto& p:rounded)targetCurve.addPoint(p);
  }
''' + token)
    return header, source
