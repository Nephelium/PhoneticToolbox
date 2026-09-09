"""Single handwritten wire-model source; scientific algorithms live in core."""
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .protocol_version import API_VERSION

SafeCount = Annotated[int, Field(ge=0, le=9007199254740991)]
Rate = Annotated[int, Field(gt=0, le=9007199254740991)]
Text = Annotated[str, Field(min_length=1)]
Hash = Annotated[str, Field(pattern=r'^[0-9a-f]{64}$')]
Identifier = Annotated[str, Field(pattern=r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')]
IdempotencyKey = Annotated[str, Field(pattern=r'^[A-Za-z0-9_-]{8,80}$')]


class WireModel(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, allow_inf_nan=False)


class Audio(WireModel):
    sample_rate_hz: Rate
    channels: Annotated[int, Field(gt=0)]
    channel_roles: list[Text]
    sample_count: SafeCount
    origin: Literal['file', 'recording', 'generated']

    @model_validator(mode='after')
    def channel_count(self) -> Self:
        if self.channels != len(self.channel_roles):
            raise ValueError('channel_roles must describe every channel')
        return self


class Selection(WireModel):
    """Integer sample-frame interval [start_sample, end_sample)."""
    start_sample: SafeCount
    end_sample: SafeCount
    sample_rate_hz: Rate

    @model_validator(mode='after')
    def order(self) -> Self:
        if self.start_sample > self.end_sample:
            raise ValueError('start_sample must not exceed end_sample')
        return self


class Track(WireModel):
    parameter_key: Text
    backend: Text
    unit: Text = Field(description='Explicit scientific unit, e.g. Hz, dB, s, %, or 1; parameter mapping is audited in P03.')
    times_s: list[Annotated[float, Field(ge=0)]]
    values: list[float | None]
    validity: list[Literal['valid', 'unvoiced', 'missing', 'failed']]
    reason: list[Text | None]
    analysis_config_hash: Hash
    source_ids: list[Text] = Field(min_length=1)

    @model_validator(mode='after')
    def trajectory(self) -> Self:
        n = len(self.times_s)
        if any(len(items) != n for items in [self.values, self.validity, self.reason]):
            raise ValueError('times, values, validity and reason must have equal lengths')
        if any(a >= b for a, b in zip(self.times_s, self.times_s[1:])):
            raise ValueError('times_s must be strictly increasing')
        for value, validity, reason in zip(self.values, self.validity, self.reason):
            if validity == 'valid':
                if value is None or reason is not None:
                    raise ValueError('valid samples require a finite value and null reason')
            elif value is not None or reason is None:
                raise ValueError('invalid samples require null value and a reason')
        return self


class Viewport(WireModel):
    audio: Audio
    selection: Selection
    tracks: list[Track]

    @model_validator(mode='after')
    def bounds(self) -> Self:
        if self.audio.sample_rate_hz != self.selection.sample_rate_hz:
            raise ValueError('selection must use the original audio sample rate')
        if self.selection.end_sample > self.audio.sample_count:
            raise ValueError('selection exceeds audio frame count')
        return self


class Health(WireModel):
    api_version: str = API_VERSION
    app_version: str
    core_version: str
    mode: Literal['local', 'server']
    status: Literal['ok'] = 'ok'


class Capabilities(WireModel):
    api_version: str = API_VERSION
    stage: Literal['P02', 'P05', 'P06', 'P07'] = 'P02'
    algorithms: list[str]
    task_operations: list[str] = []
    storage_operations: list[str] = []
    limitations: list[str]
