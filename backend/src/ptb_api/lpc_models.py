"""M04/1 input snapshots and complete managed outputs; no scientific imports."""
from typing import Literal
from pydantic import Field, model_validator
from .models import WireModel, Identifier, IdempotencyKey
from .acoustic_models import AcousticAssetRef
from .acoustic_batch_models import AcousticManagedFile
from .font_models import FigureFontSnapshot

LPC_NAMES = ['lpc.ptb.json', 'lpc_SPECTRUM.png', 'lpc_AUDIO.wav']


class LpcTaskConfig(WireModel):
    roi_start: float = Field(default=0., ge=0, le=1000)
    roi_end: float = Field(gt=0, le=1000)
    order: int = Field(default=50, ge=1, le=200)
    freq_max_hz: float = Field(default=8000., ge=100, le=48000)
    amp_min_db: float = Field(default=-5., ge=-200, le=100)
    amp_max_db: float = Field(default=35., ge=-200, le=100)
    dynamic_y: bool = False
    tier_name: str | None = Field(default=None, min_length=1, max_length=180)
    font: FigureFontSnapshot = Field(default_factory=FigureFontSnapshot)

    @model_validator(mode='after')
    def ranges(self):
        if self.roi_start >= self.roi_end or self.amp_min_db >= self.amp_max_db:
            raise ValueError('ROI and display bounds must increase')
        return self


class LpcRequest(WireModel):
    schema_version: Literal['m04/1'] = 'm04/1'
    project_id: Identifier
    idempotency_key: IdempotencyKey
    audio: AcousticAssetRef
    textgrid: AcousticAssetRef | None = None
    config: LpcTaskConfig

    @model_validator(mode='after')
    def tier_requires_grid(self):
        if self.config.tier_name is not None and self.textgrid is None:
            raise ValueError('Tier selection requires TextGrid')
        return self


class LpcSpectrumData(WireModel):
    frequencies_hz: list[float] = Field(min_length=1024,max_length=1024)
    magnitude_db: list[float] = Field(min_length=1024,max_length=1024)
    amp_min_db: float
    amp_max_db: float

    @model_validator(mode='after')
    def ordered(self):
        if self.amp_min_db >= self.amp_max_db or self.frequencies_hz[0] != 0 or any(
                a >= b for a,b in zip(self.frequencies_hz,self.frequencies_hz[1:])):
            raise ValueError('Invalid spectrum axes')
        return self


from .storage_policy import PolicyVersion, LEGACY_POLICY_VERSION


class LpcManifest(WireModel):
    policy_version: PolicyVersion = LEGACY_POLICY_VERSION
    kind: Literal['managed_lpc_files'] = 'managed_lpc_files'
    complete: Literal[True] = True
    operation: Literal['lpc_analysis'] = 'lpc_analysis'
    core_version: str
    files: list[AcousticManagedFile] = Field(min_length=3,max_length=3)

    @model_validator(mode='after')
    def complete_set(self):
        if (sorted(f.name for f in self.files) != sorted(LPC_NAMES)
                or len({f.id for f in self.files}) != 3 or sum(f.size_bytes for f in self.files)>8_000_000):
            raise ValueError('Incomplete LPC export')
        return self
