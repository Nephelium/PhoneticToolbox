"""Rendering-only font snapshot. IPA is a fixed bundled resource."""
from typing import Literal, Annotated
from pydantic import Field
from .models import WireModel

Family = Annotated[str, Field(min_length=1, max_length=100, pattern=r'^[^\x00-\x1f"\x27\\/;{}<>]+$')]

class FigureFontSnapshot(WireModel):
    schema_version: Literal['font/1'] = 'font/1'
    zh: Family = 'Microsoft YaHei'
    latin: Family = 'Segoe UI'
    ipa: Literal['Doulos SIL'] = 'Doulos SIL'
    size_px: float = Field(default=12,ge=10,le=24)
