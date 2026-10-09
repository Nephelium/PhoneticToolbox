"""M10-R13 tangent-continuous depressed blade and rounded retroflex tip.

SPDX-License-Identifier: GPL-3.0-or-later
Applied after the context-checked R11 and R12 adaptations.
"""


def apply(source):
    def replace(old, new):
        nonlocal source
        assert source.count(old) == 1, old[:100]
        source = source.replace(old, new)

    # A control-x-dependent circle height has an infinite slope as that x
    # enters the circle. Use its stable top for the auxiliary tangent point;
    # the actual depressed control point remains independent below it.
    replace('  const double tipX=(params[TBX].x-C1.x)/r1;\n', '')
    replace('  const double tipTop=C1.y+r1*sqrt(std::max(0.0,1.0-tipX*tipX));',
            '  const double tipTop=C1.y+r1;')
    replace('  Q[1].set(params[TBX].x,params[TBY].x,0.0);\n  bladeCurve.setPoints(3, &Q[0], &WEIGHT[0]);', '''  const double bladeDepression=Q[1].y-params[TBY].x;
  Q[1].set(params[TBX].x,params[TBY].x,0.0);
  bladeCurve.setPoints(3, &Q[0], &WEIGHT[0]);
  // A flexible blade still joins the body ellipse and tip circle tangentially.
  // Separate interior depression controls from those two endpoint tangents.
  Point3D flexible[6]={Q[0],Q[0],Q[1],Q[1],Q[2],Q[2]};
  Point2D bodyDirection(r0x*sin(alpha[1]),-r0y*cos(alpha[1]));
  bodyDirection.normalize();
  const double bodyHandle=std::min(0.6,(Q[1]-Q[0]).magnitude()*0.3);
  const double tipHandle=std::min(0.3,(Q[2]-Q[1]).magnitude()*0.3);
  flexible[1].x+=bodyDirection.x*bodyHandle;
  flexible[1].y+=bodyDirection.y*bodyHandle;
  flexible[4].x-=sin(alpha[2])*tipHandle;
  flexible[4].y+=cos(alpha[2])*tipHandle;
  BezierCurve3D flexibleBlade(6,flexible);
  const double flexibleWeight=std::min(1.0,std::max(0.0,bladeDepression/0.2));
''')
    replace('    targetCurve.addPoint(bladeCurve.getPoint((double)i/32.0).toPoint2D());', '''    const double bladeT=(double)i/32.0;
    targetCurve.addPoint(((1.0-flexibleWeight)*bladeCurve.getPoint(bladeT)+
      flexibleWeight*flexibleBlade.getPoint(bladeT)).toPoint2D());''')
    replace('  for (i=0; i < 8; i++)\n  {\n    angle = alpha[2] + delta*(double)i / 7.0;', '''  for (i=0; i < 32; i++)
  {
    angle = alpha[2] + delta*(double)i / 31.0;''')
    replace('  // Calculate the tongue ribs.\n', '''  // Calculate the tongue ribs.
  // Resolve the raised tip and concave blade by curvature, using the same
  // native topology. Uniform arc length discarded almost the entire tip arc.
''')
    replace('  for (i=0; i < NUM_DYNAMIC_TONGUE_RIBS; i++)\n  {\n    t = (double)i/(double)(NUM_DYNAMIC_TONGUE_RIBS-1);', '''  const int samplingCount=targetCurve.getNumPoints();
  std::vector<double> samplingDistance(samplingCount,0.0),samplingTurn(samplingCount,0.0);
  const double tipResolution=std::max(0.0,std::min(1.0,(C1.y-C0.y-1.2)/1.0));
  for(int k=1;k<samplingCount-1;k++) {
    Point2D before=targetCurve.getControlPoint(k)-targetCurve.getControlPoint(k-1);
    Point2D after=targetCurve.getControlPoint(k+1)-targetCurve.getControlPoint(k);
    const double norm=before.magnitude()*after.magnitude();
    // The upstream body arc can collapse to its epsilon angle. Its submicron
    // edges have no resolvable curvature and must not steer rib allocation.
    if(before.magnitude()>0.0001 && after.magnitude()>0.0001)
      samplingTurn[k]=acos(std::max(-1.0,std::min(1.0,(before.x*after.x+before.y*after.y)/norm)));
  }
  for(int k=1;k<samplingCount;k++) samplingDistance[k]=samplingDistance[k-1]+
    (targetCurve.getControlPoint(k)-targetCurve.getControlPoint(k-1)).magnitude()+
    1.5*tipResolution*(samplingTurn[k-1]+samplingTurn[k])*0.5;
  for (i=0; i < NUM_DYNAMIC_TONGUE_RIBS; i++)
  {
    t = (double)i/(double)(NUM_DYNAMIC_TONGUE_RIBS-1);''')
    replace('    rib[i].point = targetCurve.getPoint(t);', '''    if(tipResolution>0.0) {
      const double wanted=t*samplingDistance.back();int k=0;
      while(k<samplingCount-2 && samplingDistance[k+1]<wanted)k++;
      const double part=(wanted-samplingDistance[k])/std::max(EPSILON,samplingDistance[k+1]-samplingDistance[k]);
      t=targetCurve.getCurveParam(k)*(1.0-part)+targetCurve.getCurveParam(k+1)*part;
    }
    rib[i].point = targetCurve.getPoint(t);''')
    replace('  finalPoint.x+=curl*0.45;', '  finalPoint.x+=curl*0.08;')
    return source
