"""M05 fixed method configuration; executable paths never cross the wire."""
from typing import Literal
from pydantic import Field, model_validator
from .models import WireModel, Identifier, IdempotencyKey
from .acoustic_models import AcousticAssetRef
from .acoustic_batch_models import AcousticManagedFile
from .storage_policy import PolicyVersion, LEGACY_POLICY_VERSION

class M05Config(WireModel):
    filter_enabled: bool = True
    cutoff_hz: float = Field(default=15., ge=1, le=240, allow_inf_nan=False)
    animation: Literal['none','mp4','gif'] = 'none'
    quality: Literal['high','standard','small'] = 'standard'
    offset: float = Field(default=0., ge=-2, le=2, allow_inf_nan=False)

class M05Request(WireModel):
    schema_version: Literal['m05/1'] = 'm05/1'
    project_id: Identifier
    idempotency_key: IdempotencyKey
    video: AcousticAssetRef
    config: M05Config = Field(default_factory=M05Config)

class M05Manifest(WireModel):
    policy_version: PolicyVersion = LEGACY_POLICY_VERSION
    kind: Literal['managed_m05_files'] = 'managed_m05_files'
    complete: Literal[True] = True
    operation: Literal['lip_analysis'] = 'lip_analysis'
    core_version: str
    files: list[AcousticManagedFile] = Field(min_length=5,max_length=12)
    @model_validator(mode='after')
    def complete_set(self):
        names=[f.name for f in self.files]
        if len(names)!=len(set(names)) or not {'frames.jsonl','manifest.json','preview.json','measurements.csv','legacy-compatibility.csv'}.issubset(names):
            raise ValueError('m05_incomplete_output')
        return self

class M05Upload(WireModel):
    name: str = Field(min_length=1,max_length=220)
    size: int = Field(ge=1,le=128_000_000,strict=True)

class M05Block(WireModel):
    offset: int = Field(ge=0,le=128_000_000,strict=True)
    base64: str = Field(max_length=349528)
