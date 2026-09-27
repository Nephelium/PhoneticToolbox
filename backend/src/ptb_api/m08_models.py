"""M08/1 module-owned contract; public endpoint registration is a platform task."""
import math
from typing import Literal
from pydantic import Field, model_validator
from .models import WireModel, Identifier, IdempotencyKey
from .acoustic_models import AcousticAssetRef


class M08Point(WireModel):
    time: float
    freqs: list[float] = Field(min_length=1, max_length=256)
    mode: Literal['full', 'order', 'reverse', 'constant'] = 'order'


class M08Config(WireModel):
    action: Literal['preview', 'synthesize', 'transform', 'linear']
    start: float = 0
    end: float | None = None
    speed: float = Field(default=1., gt=0)
    pitch_ratio: float = Field(default=1., gt=0)
    pitch_hz: float = 0.
    modified_f0: list[float] | None = Field(default=None, max_length=200000)
    points: list[M08Point] = Field(default_factory=list, max_length=64)
    offset: bool = False

    @model_validator(mode='after')
    def finite(self):
        values = [self.start, self.speed, self.pitch_ratio, self.pitch_hz]
        if self.end is not None: values.append(self.end)
        values += self.modified_f0 or []
        for p in self.points: values += [p.time, *p.freqs]
        if not all(math.isfinite(x) for x in values): raise ValueError('m08_nonfinite')
        if self.start < 0 or (self.end is not None and self.end <= self.start): raise ValueError('m08_invalid_range')
        if self.action in ('synthesize', 'linear') and self.end is None: raise ValueError('m08_range_required')
        if self.action == 'synthesize' and self.modified_f0 is None: raise ValueError('m08_curve_required')
        if self.action == 'linear' and len(self.points) < 2: raise ValueError('m08_controls_required')
        return self


class M08Request(WireModel):
    schema_version: Literal['m08/1'] = 'm08/1'
    project_id: Identifier
    idempotency_key: IdempotencyKey
    audio: AcousticAssetRef
    config: M08Config


from .acoustic_batch_models import AcousticManagedFile


from .storage_policy import PolicyVersion, LEGACY_POLICY_VERSION


class M08Manifest(WireModel):
    policy_version: PolicyVersion = LEGACY_POLICY_VERSION
    kind: Literal['managed_m08_files'] = 'managed_m08_files'
    complete: Literal[True] = True
    operation: Literal['pitch_manipulation'] = 'pitch_manipulation'
    core_version: str
    files: list[AcousticManagedFile] = Field(min_length=1, max_length=258)
    # Result management metadata stays in the existing durable manifest, no DDL.
    saved: list[str] = Field(default_factory=list, max_length=256)
    deleted: list[str] = Field(default_factory=list, max_length=256)
    aliases: dict[str, str] = Field(default_factory=dict)


class M08Source(WireModel):
    project_id: Identifier
    source: AcousticAssetRef


class M08Manage(WireModel):
    project_id: Identifier
    source: AcousticAssetRef
    ids: list[Identifier] = Field(min_length=1, max_length=256)
    names: list[str] = Field(default_factory=list, max_length=256)
