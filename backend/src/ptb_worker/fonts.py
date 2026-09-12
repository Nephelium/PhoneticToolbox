"""Font resolution within the owned render child, never in the scientific core."""
import hashlib
from pathlib import Path
from .acoustic_errors import AcousticFailure

def resolve_fonts(snapshot):
    from matplotlib import font_manager
    from matplotlib.ft2font import FT2Font
    fixed=Path(__file__).with_name('assets')/'DoulosSIL-Regular.ttf'
    if not fixed.is_file():raise AcousticFailure('font_unavailable')
    font_manager.fontManager.addfont(str(fixed))
    result={'schema_version':'font/1','size_px':snapshot.size_px}
    for role,requested in [('zh',snapshot.zh),('latin',snapshot.latin),('ipa','Doulos SIL')]:
        try:
            path=fixed if role=='ipa' else Path(font_manager.findfont(font_manager.FontProperties(family=[requested]),fallback_to_default=False))
            face=FT2Font(str(path))
            # Matplotlib resolves localized names through its own catalogue. The
            # exact resolved family and content fingerprint are part of evidence.
            result[role]={'requested':requested,'family':face.family_name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        except (ValueError,OSError,RuntimeError):raise AcousticFailure('font_unavailable') from None
    return result
