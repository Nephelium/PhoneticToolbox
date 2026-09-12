import pytest
from pydantic import ValidationError
from ptb_api.font_models import FigureFontSnapshot

def test_font_snapshot_rejects_paths_css_and_ipa_substitution():
    for change in [dict(zh='C:/Windows/Fonts/a.ttf'),dict(latin='bad";color:red'),dict(ipa='Arial'),dict(size_px=100)]:
        with pytest.raises(ValidationError): FigureFontSnapshot(**change)

def test_missing_font_cannot_silently_fall_back():
    from ptb_worker.fonts import resolve_fonts
    from ptb_worker.acoustic_errors import AcousticFailure
    with pytest.raises(AcousticFailure,match='font_unavailable'):
        resolve_fonts(FigureFontSnapshot(latin='PTB nonexistent 123456'))

def test_resolved_font_evidence_and_fixed_doulos():
    from ptb_worker.fonts import resolve_fonts
    value=resolve_fonts(FigureFontSnapshot(zh='SimSun',latin='Times New Roman'))
    assert value['latin']['family']=='Times New Roman'
    assert value['ipa']['family']=='Doulos SIL'
    assert len(value['ipa']['sha256'])==64
    assert all('path' not in item for item in [value['zh'],value['latin'],value['ipa']])
