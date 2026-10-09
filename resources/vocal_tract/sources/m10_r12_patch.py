"""M10-R12 flexible blade and height-aware tongue-tip clearance.

SPDX-License-Identifier: GPL-3.0-or-later
Applied after m10_r11_patch.py to the pinned VTL 2.4 source.
"""


def apply(source):
    def replace(old, new, count=1):
        nonlocal source
        assert source.count(old) == count, old[:120]
        source = source.replace(old, new)

    replace('  // Tongue tip must be right of the tongue body.\n', '''  // M10-R12: a raised tip may retract above the body. The previous
  // x-only separation treated the full body radius as a vertical wall.
  const auto tipClearance = [&]() {
    const double dy=std::max(0.0,params[TTY].x-params[TCY].x)/(ry+tongueTipRadius);
    return std::max(0.35,(rx+tongueTipRadius)*sqrt(std::max(0.0,1.0-dy*dy)));
  };
  // Tongue tip must be right of the tongue body.
''')
    replace('params[TCX].x + rx + tongueTipRadius', 'params[TCX].x + tipClearance()', 4)
    replace('''  // The tongue body must be left of the tongue tip.
  if (params[TCX].x > params[TTX].x - rx - tongueTipRadius)
  {
    params[TCX].x = params[TTX].x - rx - tongueTipRadius;
  }
''', '''  // M10-R12: the independently dragged tip does not translate the body.
  // Its height-aware clearance is checked again after the body hull limit.
''')
    replace('  if (params[TBX].x > params[TTX].x) { params[TBX].x = params[TTX].x; }',
            '  if (params[TBX].x > params[TTX].x+0.75) { params[TBX].x = params[TTX].x+0.75; }')
    replace('  params[TBY].x=std::max(params[TBY].x,std::max(bodyTop,tipTop)+EPSILON);', '''  // The blade can depress into the flexible tongue body. Keep its control
  // point above the actual floor/dental boundary, not above two rigid circles.
  Point2D bladePoint(params[TBX].x,params[TBY].x),floorPoint;
  double floorDistance;
  if (lowerBorderTB.getClosestIntersection(bladePoint,Point2D(0.0,1.0),floorDistance,floorPoint))
    params[TBY].x=std::max(params[TBY].x,floorPoint.y+0.12);
''')
    replace('  Q[1].set(params[TBX].x, params[TBY].x, 0.0);', '''  // Use feasible tangent endpoints, but the actual flexible control point
  // may lie below the ellipse. This permits a smooth concave blade without
  // asking getEllipseTangent() to solve a point inside its rigid ellipse.
  const double bladeX=(params[TBX].x-C0.x)/r0x;
  const double tipX=(params[TBX].x-C1.x)/r1;
  const double bladeTop=C0.y+r0y*sqrt(std::max(0.0,1.0-bladeX*bladeX));
  const double tipTop=C1.y+r1*sqrt(std::max(0.0,1.0-tipX*tipX));
  Q[1].set(params[TBX].x,std::max(params[TBY].x,std::max(bladeTop,tipTop)+EPSILON),0.0);''')
    replace('  bladeCurve.setPoints(3, &Q[0], &WEIGHT[0]);', '''  Q[1].set(params[TBX].x,params[TBY].x,0.0);
  bladeCurve.setPoints(3, &Q[0], &WEIGHT[0]);''')
    replace('  alpha[2] = getCircleTangent(Q[1].toPoint2D(), C1, r1, true);', '''  // Keep the joining tangent on the posterior/upper semicircle when the
  // tip curls behind the blade. The opposite branch crosses zero and then
  // wraps to nearly 2*pi, spuriously drawing a whole loop through the tongue.
  Point2D tipTangent=Q[1].toPoint2D();
  if (tipTangent.x>C1.x) tipTangent.x=2*C1.x-tipTangent.x;
  alpha[2] = getCircleTangent(tipTangent, C1, r1, true);''')
    replace('  Point2D finalPoint = rib[i].point + rib[i].normal*t;', '''  Point2D finalPoint = rib[i].point + rib[i].normal*t;
  // The terminal underside rib must also clear the returning blade.
  const double curl=std::max(0.0,std::min(1.0,(params[TBX].x-params[TTX].limitedX)/0.75))*
    std::max(0.0,std::min(1.0,(params[TTY].limitedX-params[TBY].x)/0.5));
  finalPoint.x+=curl*0.45;''')
    replace('  D.x-= 2.0*tongueTipRadius;', '''  D.x-= 2.0*tongueTipRadius;
  // A retroflex underside must go around the blade's anterior shoulder.
  // The ordinary rearward 45-degree return crosses a depressed/curled blade.
  D.x+=curl*1.2;''')
    return source
