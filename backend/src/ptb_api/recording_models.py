"""M16 web task contract. Frame indices and native defaults match the desktop."""
from typing import Literal
from pydantic import Field, model_validator
from .models import WireModel, Identifier, IdempotencyKey


class RecordingConfig(WireModel):
    inputs: list[Identifier] = Field(min_length=1, max_length=2)
    hashes: dict[str,str]
    action: Literal['noise','denoise','gain','keep','delete','cut','paste','export']
    start: int = Field(default=0, ge=0, strict=True)
    end: int | None = Field(default=None, ge=0, strict=True)
    reference_start: int = Field(default=0, ge=0, strict=True)
    reference_end: int | None = Field(default=None, ge=0, strict=True)
    channel: int = Field(default=0, ge=0, le=7, strict=True)
    roles: list[Literal['microphone','egg','other']] = Field(min_length=1,max_length=8)
    reference_roles: list[Literal['microphone','egg','other']] | None = None
    channels: list[int] = Field(default_factory=list,max_length=8)
    strength: float = Field(default=1.,ge=.1,le=2.,allow_inf_nan=False)
    gain_db: float = Field(default=0.,ge=-60,le=24,allow_inf_nan=False)
    subtype: Literal['FLOAT','PCM_24','PCM_16'] = 'FLOAT'
    max_output_bytes: int = Field(default=310_000_000,ge=1,le=310_000_000)

    @model_validator(mode='after')
    def references(self):
        import re
        if len(set(self.inputs))!=len(self.inputs) or set(self.hashes)!=set(self.inputs):raise ValueError('input references')
        if any(not re.fullmatch('[a-f0-9]{64}',h) for h in self.hashes.values()):raise ValueError('input hashes')
        if self.action in ('paste','denoise') and len(self.inputs)!=2 and self.reference_end is None:raise ValueError('reference required')
        if len(set(self.channels))!=len(self.channels) or any(type(c)!=int or not 0<=c<len(self.roles) for c in self.channels):raise ValueError('channels')
        return self


class RecordingRequest(WireModel):
    project_id: Identifier
    idempotency_key: IdempotencyKey
    operation: Literal['recording_audio'] = 'recording_audio'
    config: RecordingConfig
