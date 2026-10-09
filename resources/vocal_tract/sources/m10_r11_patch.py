"""Auditable, context-checked adaptations to the pinned VTL 2.4 source.

SPDX-License-Identifier: GPL-3.0-or-later
"""


def apply(source):
    def replace(old,new,count=1):
        nonlocal source
        assert source.count(old)==count,old[:100]
        source=source.replace(old,new)

    # A separate tongue-tip hull continues through the mouth opening. The
    # body retains the upstream hull. Teeth/palate remain collision surfaces.
    replace('LineStrip2D lowerBorderTT;    // Lower boundary for the tongue tip circle.',
            'LineStrip2D tipUpperBorder;\n  LineStrip2D lowerBorderTT;    // Lower boundary for the tongue tip circle.')
    marker='  // Lower border for the tongue tip.\n'
    replace(marker,marker)
    anchor='  lowerBorderTT.reset(1);'
    replace(anchor,'''  // M10-R11: open the artificial anterior wall, keep the upper incisor.
  for (i=0; i<upperBorder.getNumPoints()-1; i++)
    tipUpperBorder.addPoint(upperBorder.getControlPoint(i));
  P0=tipUpperBorder.getControlPoint(tipUpperBorder.getNumPoints()-1);
  tipUpperBorder.addPoint(Point2D(P0.x+10.0,P0.y));

  lowerBorderTT.reset(1);''')
    replace('  lowerBorderTT.addPoint(P1);','  lowerBorderTT.addPoint(Point2D(P0.x+10.0,P0.y));')
    replace('tongueTipRadius, tongueTipRadius, upperBorder, A);','tongueTipRadius, tongueTipRadius, tipUpperBorder, A);')
    begin=source.index('  // Later, the ellipsoid shape should be considered here !!')
    end=source.index('  // ****************************************************************',begin)
    # Exact ellipsoid clearance replaces the 45-degree construction cones.
    # A tangent construction point must remain outside the body and tip.
    source=source[:begin]+'''  // M10-R11: keep valid upper tangents without the old 45-degree cones.
  double bladeDx=(params[TBX].x-params[TCX].x)/rx;
  double bodyTop=params[TCY].x+ry*sqrt(std::max(0.0,1.0-bladeDx*bladeDx));
  double tipDx=(params[TBX].x-params[TTX].x)/tongueTipRadius;
  double tipTop=params[TTY].x+tongueTipRadius*sqrt(std::max(0.0,1.0-tipDx*tipDx));
  params[TBY].x=std::max(params[TBY].x,std::max(bodyTop,tipTop)+EPSILON);

'''+source[end:]
    # Reuse the upstream contour construction after independent surface edits.
    # Center-line calculations and the fast intersection tiles otherwise retain
    # pre-deformation geometry even though the mesh readout has already moved.
    start=source.index('  int upperCoverPoint = NUM_UPPER_COVER_POINTS-1;')
    end=source.index('  calcTongue();',start)
    covers=source[start:end]
    start=source.index('  tongueOutline.reset(0);')
    end=source.index('  Surface *source, *target;',start)
    mobile=source[start:end]
    source+='\nvoid VocalTract::refreshM10Geometry()\n{\n  int i;\n'+covers+mobile+'''
  for (i=0; i<NUM_SURFACES; i++) intersectionsPrepared[i]=false;
}
'''
    return source
