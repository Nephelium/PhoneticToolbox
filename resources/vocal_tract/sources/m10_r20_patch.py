"""M10-R20: independent lip targets, retaining upstream lip attachment/loft.

SPDX-License-Identifier: GPL-3.0-or-later
Zero offsets preserve the upstream lip points and all earlier geometry.
"""


def apply(header, source):
    old = '  void refreshM10Geometry();'
    assert header.count(old) == 1
    header = header.replace(old, old + '\n  double m10LipShape[4] = {0,0,0,0};')
    old = '  // z-coordinate.\n\n  Point3D lipTangent'
    assert source.count(old) == 1
    source = source.replace(old, '''  // Independent author offsets in centimetres. Keep both gum attachments,
  // common LP/LD and the existing native limits. Contact is resolved after
  // the lip surfaces have been constructed, before areas are measured.
  F0.x = std::max(T0.x + X_OFFSET, F0.x + m10LipShape[0]);
  F1.x = std::max(T1.x + X_OFFSET, F1.x + m10LipShape[2]);
  F0.y = std::max(T1.y, std::min(T0.y, F0.y + m10LipShape[1]));
  F1.y = std::max(T1.y, std::min(T0.y, F1.y + m10LipShape[3]));

  // z-coordinate.

  Point3D lipTangent''')
    old='void VocalTract::refreshM10Geometry()\n{'
    assert source.count(old)==1
    source=source.replace(old,old+'''
  if (std::abs(m10LipShape[0])+std::abs(m10LipShape[1])+
      std::abs(m10LipShape[2])+std::abs(m10LipShape[3]) > 1e-12) {
    Point3D onset, corner, f0, f1; double close;
    getImportantLipPoints(onset, corner, f0, f1, close);
    calcRadiation(corner);
  }
''')
    return header, source
