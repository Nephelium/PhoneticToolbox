"""P16: host-side portable font choice, strict IPA and auditable actual face."""
from pathlib import Path
from types import SimpleNamespace
import pytest
from ptb_api.font_models import FigureFontSnapshot
from ptb_worker.fonts import check_fonts


def test_windows_font_request_can_resolve_on_a_linux_font_catalog(monkeypatch,tmp_path):
    from matplotlib import font_manager,ft2font
    paths={}
    for name in ('Noto Sans CJK SC','DejaVu Sans'):
        p=tmp_path/(name+'.ttf');p.write_bytes(name.encode());paths[name]=p
    def find(props,**kwargs):
        name=props.get_family()[0]
        if name not in paths:raise ValueError('missing')
        return str(paths[name])
    monkeypatch.setattr(font_manager,'findfont',find)
    monkeypatch.setattr(font_manager.fontManager,'addfont',lambda path:None)
    monkeypatch.setattr(ft2font,'FT2Font',lambda path:SimpleNamespace(family_name=Path(path).stem if Path(path).stem!='DoulosSIL-Regular' else 'Doulos SIL'))
    value=check_fonts(FigureFontSnapshot(fallback_policy='portable'))
    assert value['available']
    assert value['fonts'][0]['requested']=='Microsoft YaHei'
    assert value['fonts'][0]['family']=='Noto Sans CJK SC'
    assert value['fonts'][1]['family']=='DejaVu Sans'
    assert all(len(item['sha256'])==64 for item in value['fonts'])


def test_portable_mode_still_rejects_a_missing_ipa_font(monkeypatch):
    from matplotlib import font_manager
    def missing(*a,**kw):raise ValueError('missing IPA')
    monkeypatch.setattr(font_manager.fontManager,'addfont',missing)
    assert not check_fonts(FigureFontSnapshot(fallback_policy='portable'))['available']
