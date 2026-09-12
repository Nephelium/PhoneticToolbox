"""Font resolution within the owned render child, never in the scientific core."""
import hashlib
from pathlib import Path
from .acoustic_errors import AcousticFailure

def check_fonts(snapshot):
    from matplotlib import font_manager
    from matplotlib.ft2font import FT2Font
    fixed=Path(__file__).with_name('assets')/'DoulosSIL-Regular.ttf'
    fixed_available=True
    try:font_manager.fontManager.addfont(str(fixed))
    except (ValueError,OSError,RuntimeError):fixed_available=False
    result=[]
    for role,requested in [('zh',snapshot.zh),('latin',snapshot.latin),('ipa','Doulos SIL')]:
        try:
            if role=='ipa' and not fixed_available:raise ValueError('Bundled font unavailable')
            path=fixed if role=='ipa' else Path(font_manager.findfont(font_manager.FontProperties(family=[requested]),fallback_to_default=False))
            face=FT2Font(str(path))
            # Matplotlib resolves localized names through its own catalogue. The
            # exact resolved family and content fingerprint are part of evidence.
            result.append(dict(role=role,requested=requested,available=True,family=face.family_name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        except (ValueError,OSError,RuntimeError):result.append(dict(role=role,requested=requested,available=False,family=None,sha256=None))
    return dict(available=all(item['available'] for item in result),fonts=result)


def resolve_fonts(snapshot):
    checked=check_fonts(snapshot)
    if not checked['available']:raise AcousticFailure('font_unavailable')
    return {'schema_version':'font/1','size_px':snapshot.size_px,**{
        item['role']:{key:item[key] for key in ('requested','family','sha256')} for item in checked['fonts']}}
