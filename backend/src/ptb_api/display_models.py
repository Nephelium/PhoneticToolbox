"""M02 original-frame, bounded result display contract."""
from typing import Literal
from pydantic import Field, model_validator
from .models import WireModel


class ParameterTable(WireModel):
    schema_version: Literal['m02/1'] = 'm02/1'
    sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    columns: list[str] = Field(min_length=1, max_length=256)
    kinds: list[Literal['number', 'text']] = Field(min_length=1, max_length=256)
    rows: list[list[float | str | None]] = Field(max_length=200000)

    @model_validator(mode='after')
    def shape(self):
        if len(self.columns) != len(self.kinds) or any(len(row) != len(self.columns) for row in self.rows):
            raise ValueError('Invalid table shape')
        if len(self.rows) * len(self.columns) > 200000:
            raise ValueError('Table exceeds display budget')
        return self
